#!/usr/bin/env python3
"""Evaluate a saved ETMC checkpoint on the untouched prepared test split.

Example:
  !python training/03_evaluate_etmc.py --prepared-root prepared_dataset \
      --checkpoint outputs/best_etmc_model.pth --output-dir evaluation_results
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from etmc_common import ETMC, PreparedAVDataset, checkpoint_model_state, uncertainty, save_json, normalize_frames

def score_metrics(y, probability, u, threshold=.5, seconds=None):
    pred=(probability>=threshold).astype(int); cm=confusion_matrix(y,pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    return {"samples":int(len(y)),"threshold":float(threshold),"accuracy":float(accuracy_score(y,pred)),
      "precision":float(precision_score(y,pred,zero_division=0)),"recall_sensitivity":float(recall_score(y,pred,zero_division=0)),
      "specificity":float(tn/max(1,tn+fp)),"f1":float(f1_score(y,pred,zero_division=0)),"false_positive_rate":float(fp/max(1,fp+tn)),
      "false_negative_rate":float(fn/max(1,fn+tp)),"roc_auc":float(roc_auc_score(y,probability)) if len(set(y))==2 else None,
      "confusion_matrix":{"labels":["REAL","FAKE"],"matrix":cm.tolist()},"uncertainty":{"mean":float(np.mean(u)),"median":float(np.median(u)),"std":float(np.std(u)),"min":float(np.min(u)),"max":float(np.max(u))},
      "inference_seconds_total":float(seconds or 0),"inference_seconds_average":float((seconds or 0)/max(1,len(y)))}

def save_cm(path,cm,title):
    fig,ax=plt.subplots(figsize=(4,4)); ax.imshow(cm,cmap="Blues"); ax.set(title=title,xlabel="Predicted",ylabel="Actual",xticks=[0,1],yticks=[0,1],xticklabels=["REAL","FAKE"],yticklabels=["REAL","FAKE"])
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center")
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)
def save_roc(path,y,p,title):
    if len(set(y))<2:
        fig,ax=plt.subplots(); ax.text(.5,.5,"ROC-AUC unavailable: test labels contain one class",ha="center",va="center",wrap=True); ax.set_axis_off(); ax.set_title(title); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig); return
    from sklearn.metrics import roc_curve
    fpr,tpr,_=roc_curve(y,p); fig,ax=plt.subplots(); ax.plot(fpr,tpr,label=f"AUC={roc_auc_score(y,p):.3f}"); ax.plot([0,1],[0,1],"--",color="gray"); ax.set(xlabel="False positive rate",ylabel="True positive rate",title=title); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-root",default="prepared_dataset"); p.add_argument("--checkpoint",default="outputs/best_etmc_model.pth"); p.add_argument("--output-dir",default="evaluation_results")
    p.add_argument("--batch-size",type=int,default=4); p.add_argument("--workers",type=int,default=2); p.add_argument("--threshold",type=float,default=.5)
    p.add_argument("--ambiguity-quantile",type=float,default=None,help="Optional uncertainty quantile in [0,1], calibrated from validation rows only")
    args=p.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False); model=ETMC(image_pretrained=False).to(device); model.load_state_dict(checkpoint_model_state(ck),strict=True); model.eval()
    ds=PreparedAVDataset(Path(args.prepared_root)/"test.csv"); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.type=="cuda")
    labels={k:[] for k in ("visual","audio","overall")}; scores={k:[] for k in ("visual","audio","pseudo","fused","visual_audio")}; us={k:[] for k in scores}; rows=[]; infer=0.; offset=0
    with torch.inference_mode():
        for batch in loader:
            frames=normalize_frames(batch["frames"].to(device)); audio=batch["audio"].to(device)
            t=time.perf_counter(); va,aa,pa,fa=model(frames,audio); fv=model.fused_without_pseudo(va,aa); infer+=time.perf_counter()-t
            alphas={"visual":va,"audio":aa,"pseudo":pa,"fused":fa,"visual_audio":fv}
            for k,a in alphas.items():
                a=a.float(); scores[k].extend((a/a.sum(1,keepdim=True))[:,1].cpu().numpy()); us[k].extend(uncertainty(a).cpu().numpy())
            labels["visual"].extend(batch["visual_label"].numpy()); labels["audio"].extend(batch["audio_label"].numpy()); labels["overall"].extend(batch["overall_label"].numpy())
            n=len(batch["visual_label"])
            for j in range(n):
                idx=offset+j
                rows.append({"video_path":ds.frame.iloc[idx].video_path,"category":ds.frame.iloc[idx].category,
                  "visual_label":int(batch["visual_label"][j]),"audio_label":int(batch["audio_label"][j]),"overall_label":int(batch["overall_label"][j]),
                  **{f"{k}_fake_probability":float(scores[k][-n+j]) for k in scores},**{f"{k}_prediction":int(scores[k][-n+j]>=args.threshold) for k in scores},
                  **{f"{k}_uncertainty":float(us[k][-n+j]) for k in us}})
            offset+=n
    yv=np.asarray(labels["visual"]); ya=np.asarray(labels["audio"]); yo=np.asarray(labels["overall"]); threshold=args.threshold
    result={"visual":score_metrics(yv,np.asarray(scores["visual"]),np.asarray(us["visual"]),threshold,infer),
            "audio":score_metrics(ya,np.asarray(scores["audio"]),np.asarray(us["audio"]),threshold,infer),
            "pseudo":score_metrics(yo,np.asarray(scores["pseudo"]),np.asarray(us["pseudo"]),threshold,infer),
            "fused":score_metrics(yo,np.asarray(scores["fused"]),np.asarray(us["fused"]),threshold,infer),
            "visual_audio_without_pseudo":score_metrics(yo,np.asarray(scores["visual_audio"]),np.asarray(us["visual_audio"]),threshold,infer)}
    for key in ("visual","audio","fused"): save_json(out/f"{key}_metrics.json",result[key])
    save_json(out/"ablation_results.json",{k:result[k] for k in ("pseudo","fused","visual_audio_without_pseudo")})
    pd.DataFrame(rows).to_csv(out/"predictions.csv",index=False)
    for key,actual in (("visual",yv),("audio",ya),("fused",yo)):
        prob=np.asarray(scores[key]); pred=(prob>=threshold).astype(int); cm=confusion_matrix(actual,pred,labels=[0,1]); save_cm(out/f"confusion_matrix_{key}.png",cm,f"{key.title()} confusion matrix"); save_roc(out/f"roc_{key}.png",actual,prob,f"{key.title()} ROC")
    fig,ax=plt.subplots(); ax.hist(us["visual"],bins=30,alpha=.5,label="visual"); ax.hist(us["audio"],bins=30,alpha=.5,label="audio"); ax.hist(us["fused"],bins=30,alpha=.5,label="fused"); ax.set(xlabel="Dirichlet uncertainty",ylabel="Samples",title="Predictive uncertainty"); ax.legend(); fig.tight_layout(); fig.savefig(out/"uncertainty_analysis.png",dpi=160); plt.close(fig)
    amb_threshold=None; ambiguous=None
    if args.ambiguity_quantile is not None:
        if not 0<=args.ambiguity_quantile<=1:p.error("--ambiguity-quantile must be in [0,1]")
        val=PreparedAVDataset(Path(args.prepared_root)/"validation.csv"); vl=DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
        vu=[]
        with torch.inference_mode():
            for b in vl:
                v=normalize_frames(b["frames"].to(device)); a=b["audio"].to(device); alpha=model(v,a)[-1]; vu.extend(uncertainty(alpha).cpu().numpy())
        amb_threshold=float(np.quantile(vu,args.ambiguity_quantile)); ambiguous=np.where(np.asarray(us["fused"])>amb_threshold,"AMBIGUOUS",np.where(np.asarray(scores["fused"])>=threshold,"FAKE","REAL"))
        for r,d in zip(rows,ambiguous):r["decision_with_ambiguity"]=str(d)
        pd.DataFrame(rows).to_csv(out/"predictions.csv",index=False)
    report={"dataset_size":len(ds),"test_split":"prepared_dataset/test.csv; not used for training, threshold choice, or early stopping","checkpoint":str(Path(args.checkpoint).resolve()),"checkpoint_epoch":ck.get("epoch"),"best_validation_fused_f1":ck.get("best_validation_f1"),"random_seed":ck.get("random_seed"),"model_config":ck.get("model_config"),
      "threshold":threshold,"threshold_source":"CLI fixed default or provided threshold; no test calibration","ambiguity_threshold":amb_threshold,"ambiguity_source":f"validation uncertainty quantile {args.ambiguity_quantile}" if amb_threshold is not None else None,
      "metrics":result,"inference_time_seconds_total":infer,"inference_time_seconds_average":infer/max(1,len(ds)),"split_summary":json.loads((Path(args.prepared_root)/"split_summary.json").read_text()) if (Path(args.prepared_root)/"split_summary.json").exists() else None,
      "limitations":["The pseudo-view is learned from both modalities and is not an independent physical sensor.","Scores and thresholds should be interpreted on the supplied data distribution.","Data preparation identity-group quality depends on source identity metadata."]}
    save_json(out/"evaluation_report.json",report)
    md=["# ETMC test evaluation","",f"Test samples: **{len(ds)}**",f"Checkpoint: `{Path(args.checkpoint).resolve()}`",f"Checkpoint epoch: {report['checkpoint_epoch']}; best validation fused F1: {report['best_validation_fused_f1']}",f"Inference for all modes: {infer:.3f}s total, {infer/max(1,len(ds)):.6f}s/sample",f"Decision threshold: {threshold} ({report['threshold_source']})",f"Ambiguity threshold: {amb_threshold}","","Evaluation used the precomputed test split once. The model was not retrained and no threshold was selected on test outcomes.","","## Metrics",""]
    split_summary=report.get("split_summary") or {}
    md[md.index("## Metrics")-1:md.index("## Metrics")-1]=["","## Split details",f"Grouping column: {split_summary.get('grouping_column','not recorded')} (identity source: {split_summary.get('identity_source','not recorded')}); identity-independent claim supported: {split_summary.get('identity_independent_claim_supported','unknown')}","| Split | Samples | Unique actors/sources | Categories |","|---|---:|---:|---|"]+[f"| {name} | {info.get('samples')} | {info.get('unique_actors_or_sources')} | {info.get('category_distribution')} |" for name,info in (split_summary.get('splits') or {}).items()]+[""]
    md += ["| Mode | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Mean uncertainty |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in result:
        m=result[k]; md.append(f"| {k} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall_sensitivity']:.4f} | {m['specificity']:.4f} | {m['f1']:.4f} | {m['roc_auc'] if m['roc_auc'] is not None else 'N/A'} | {m['uncertainty']['mean']:.4f} |")
    md += ["","## Uncertainty",*[f"- {k}: mean={v['uncertainty']['mean']:.4f}, median={v['uncertainty']['median']:.4f}, range=[{v['uncertainty']['min']:.4f}, {v['uncertainty']['max']:.4f}]" for k,v in result.items()],"","## Ablation","","Compares visual, audio, visual+audio DS fusion without the pseudo-view, the learned pseudo-view, and complete ETMC fusion. The pseudo-view is a joint learned representation, not an independent sensor.","","## Limitations","",*['- '+x for x in report['limitations']]]
    (out/"final_evaluation_report.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print(f"Evaluation artifacts written to {out.resolve()}")
if __name__=="__main__": main()
