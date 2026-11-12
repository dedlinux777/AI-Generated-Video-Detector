#!/usr/bin/env python3
"""Train the complete ETMC audiovisual model in Google Colab.

Setup and Drive mount are documented in training/README.md. Example:
  !python training/02_train_etmc.py --prepared-root prepared_dataset \
      --checkpoint checkpoints/model.pth --output-dir /content/drive/MyDrive/etmc_run
"""
from __future__ import annotations
import argparse, csv, json, time, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from etmc_common import ETMC, PreparedAVDataset, evidential_loss, uncertainty, seed_everything, load_pretrained_encoders, save_json, normalize_frames

def metrics(y, score, alpha, loss):
    pred=(score>=.5).astype(int); cm=confusion_matrix(y,pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    return {"loss":float(loss),"accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),
            "recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),
            "specificity":float(tn/max(1,tn+fp)),"roc_auc":float(roc_auc_score(y,score)) if len(set(y))==2 else None,
            "mean_uncertainty":float(np.mean(alpha.shape[1]/alpha.sum(1)))}

def evaluate(model, loader, device, weights_by_branch, step):
    model.eval(); ys={k:[] for k in ("visual","audio","pseudo","fused")}; scores={k:[] for k in ys}; alphas={k:[] for k in ys}; losses={k:[] for k in ys}
    with torch.no_grad():
        for batch in loader:
            frames=normalize_frames(batch["frames"].to(device))
            audio=batch["audio"].to(device); labels={"visual":batch["visual_label"].to(device),"audio":batch["audio_label"].to(device),"pseudo":batch["overall_label"].to(device),"fused":batch["overall_label"].to(device)}
            with torch.autocast(device_type="cuda",enabled=device.type=="cuda"):
                outs=model(frames,audio)
            weight_keys={"visual":"visual","audio":"audio","pseudo":"overall","fused":"overall"}
            for key,a,y in zip(ys,outs,labels.values()):
                alpha=a.float(); p=alpha/alpha.sum(1,keepdim=True)
                ys[key].extend(y.cpu().numpy()); scores[key].extend(p[:,1].cpu().numpy()); alphas[key].extend(alpha.cpu().numpy())
                losses[key].append(evidential_loss(y,alpha,weights_by_branch[weight_keys[key]],step).item())
    return {k:metrics(np.asarray(ys[k]),np.asarray(scores[k]),np.asarray(alphas[k]),np.mean(losses[k])) for k in ys}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-root",default="prepared_dataset"); p.add_argument("--checkpoint",default="checkpoints/model.pth")
    p.add_argument("--output-dir",default="outputs"); p.add_argument("--epochs",type=int,default=10); p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--workers",type=int,default=2); p.add_argument("--seed",type=int,default=42); p.add_argument("--max-runtime-minutes",type=float,default=120)
    p.add_argument("--max-steps",type=int,default=0); p.add_argument("--gradient-accumulation",type=int,default=1); p.add_argument("--freeze-epochs",type=int,default=2)
    p.add_argument("--lambda-visual",type=float,default=1); p.add_argument("--lambda-audio",type=float,default=1); p.add_argument("--lambda-pseudo",type=float,default=1); p.add_argument("--lambda-fused",type=float,default=1)
    p.add_argument("--patience",type=int,default=4); p.add_argument("--resume",default=None); p.add_argument("--smoke-test",action="store_true")
    args=p.parse_args(); seed_everything(args.seed); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=ETMC(image_pretrained=not args.smoke_test).to(device); compatibility=load_pretrained_encoders(model,args.checkpoint)
    if args.smoke_test:
        model.eval(); frames=torch.rand(2,3,3,256,256,device=device); audio=torch.randn(2,48000,device=device)*.01
        vals=model(frames,audio)
        assert len(vals)==4 and all(x.shape==(2,2) and torch.isfinite(x).all() and (x>0).all() for x in vals)
        loss=sum(evidential_loss(torch.tensor([0,1],device=device),v) for v in vals); loss.backward()
        tmp=out/"smoke_checkpoint.pth"; torch.save({"model_state_dict":model.state_dict()},tmp)
        restored=ETMC(image_pretrained=False).to(device); restored.load_state_dict(torch.load(tmp,map_location=device,weights_only=False)["model_state_dict"])
        print("Smoke test passed: ETMC forward, positive finite alpha, loss/backward, checkpoint save/reload."); return
    train_csv=Path(args.prepared_root)/"train.csv"; val_csv=Path(args.prepared_root)/"validation.csv"
    train_ds=PreparedAVDataset(train_csv); val_ds=PreparedAVDataset(val_csv)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=device.type=="cuda",persistent_workers=args.workers>0)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.type=="cuda",persistent_workers=args.workers>0)
    def weights(column):
        counts=train_ds.frame[column].value_counts().to_dict(); total=sum(counts.values())
        return torch.tensor([total/(2*counts.get(c,1)) for c in (0,1)],device=device,dtype=torch.float32)
    weights_by_branch={"visual":weights("visual_label"),"audio":weights("audio_label"),"overall":weights("overall_label")}
    split_file=Path(args.prepared_root)/"split_summary.json"
    prep_cfg=json.loads(split_file.read_text()).get("preprocessing",{}) if split_file.exists() else {}
    first_audio=np.load(train_ds.frame.iloc[0].cache_audio,mmap_mode="r")
    first_frames=np.load(train_ds.frame.iloc[0].cache_frames,mmap_mode="r")
    cfg={**vars(args),"device":str(device),"class_mapping":{"0":"REAL","1":"FAKE"},"preprocessing":{**prep_cfg,"size":256,"color":"RGB","image_normalization":"timm tf_efficientnet_b4_ns config: mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)","audio_samples":int(len(first_audio)),"cached_frame_count":int(len(first_frames))},"pretrained_compatibility":compatibility,"primary_threshold":0.5,"threshold_source":"fixed default; no test-based tuning"}
    save_json(out/"config.json",cfg)
    start=time.monotonic(); best=-1.; best_epoch=-1; stale=0; global_step=0; history=[]; scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    resume_payload=None
    if args.resume:
        resume_payload=torch.load(args.resume,map_location=device,weights_only=False); model.load_state_dict(resume_payload["model_state_dict"]); start_epoch=resume_payload.get("epoch",-1)+1; best=resume_payload.get("best_validation_f1",-1); global_step=resume_payload.get("global_step",0)
        rng=resume_payload.get("rng_state")
        if rng:
            random.setstate(rng["python"]); np.random.set_state(rng["numpy"]); torch.set_rng_state(rng["torch"])
            if device.type=="cuda" and rng.get("cuda") is not None: torch.cuda.set_rng_state_all(rng["cuda"])
    else: start_epoch=0
    optimizer=None; scheduler=None; active_stage=None
    for epoch in range(start_epoch,args.epochs):
        stage=1 if epoch<args.freeze_epochs else 2; model.set_stage(stage)
        if stage != active_stage:
            optimizer=torch.optim.AdamW([{"params":[p for n,p in model.named_parameters() if p.requires_grad and not (n.startswith("visual_encoder") or n.startswith("audio_encoder"))],"lr":1e-4},
                                         {"params":[p for n,p in model.named_parameters() if p.requires_grad and (n.startswith("visual_encoder") or n.startswith("audio_encoder"))],"lr":1e-5}],weight_decay=1e-4)
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=.5,patience=1)
            if resume_payload is not None and epoch==start_epoch:
                try:
                    optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
                    if "scheduler_state_dict" in resume_payload: scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
                except (ValueError, KeyError) as e: print(f"Resume optimizer state was incompatible with stage {stage}; retaining model weights and restarting optimizer: {e}")
            active_stage=stage
        model.train(); optimizer.zero_grad(set_to_none=True); sums=[]
        for bi,batch in enumerate(train_loader):
            frames=normalize_frames(batch["frames"].to(device,non_blocking=True)); audio=batch["audio"].to(device,non_blocking=True)
            vy=batch["visual_label"].to(device); ay=batch["audio_label"].to(device); oy=batch["overall_label"].to(device)
            with torch.autocast(device_type="cuda",enabled=device.type=="cuda"):
                va,aa,pa,fa=model(frames,audio)
                lv=evidential_loss(vy,va,weights_by_branch["visual"],global_step); la=evidential_loss(ay,aa,weights_by_branch["audio"],global_step)
                lp=evidential_loss(oy,pa,weights_by_branch["overall"],global_step); lf=evidential_loss(oy,fa,weights_by_branch["overall"],global_step)
                loss=(args.lambda_visual*lv+args.lambda_audio*la+args.lambda_pseudo*lp+args.lambda_fused*lf)/args.gradient_accumulation
            scaler.scale(loss).backward(); sums.append(float(loss.item()*args.gradient_accumulation)); global_step+=1
            runtime_hit=(time.monotonic()-start)/60>=args.max_runtime_minutes
            step_limit_hit=bool(args.max_steps and global_step>=args.max_steps)
            if (bi+1)%args.gradient_accumulation==0 or bi+1==len(train_loader) or runtime_hit or step_limit_hit:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if step_limit_hit or runtime_hit: break
        vm=evaluate(model,val_loader,device,weights_by_branch,global_step)
        row={"epoch":epoch+1,"stage":stage,"train_loss":float(np.mean(sums)),"validation":vm,"global_step":global_step}; history.append(row)
        primary=vm["fused"]["f1"]; scheduler.step(primary)
        rng_state={"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if device.type=="cuda" else None}
        state={"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict(),"epoch":epoch,"stage":stage,"best_validation_f1":max(best,primary),"global_step":global_step,"model_config":{"visual_dim":model.visual_dim,"audio_dim":model.audio_dim},"class_mapping":cfg["class_mapping"],"preprocessing":cfg["preprocessing"],"random_seed":args.seed,"rng_state":rng_state,"primary_threshold":0.5,"threshold_source":"fixed default 0.5","validation_metrics":vm}
        torch.save(state,out/"latest_etmc_checkpoint.pth")
        if primary>best:
            best,best_epoch,stale=primary,epoch,0; torch.save(state,out/"best_etmc_model.pth")
        else: stale+=1
        save_json(out/"training_history.json",history)
        pd.json_normalize(history,sep=".").to_csv(out/"training_history.csv",index=False)
        print(f"epoch {epoch+1}/{args.epochs} stage={stage} train_loss={np.mean(sums):.4f} val_fused_f1={primary:.4f}")
        if stale>=args.patience or (args.max_steps and global_step>=args.max_steps) or (time.monotonic()-start)/60>=args.max_runtime_minutes: break
    save_json(out/"training_summary.json",{"epochs_completed":len(history),"best_epoch":best_epoch+1 if best_epoch>=0 else None,"best_validation_fused_f1":best,"runtime_minutes":(time.monotonic()-start)/60,"complete_training_run":len(history)>=args.epochs,"note":"Runtime and hardware can prevent completion; no duration guarantee."})
if __name__=="__main__": main()
