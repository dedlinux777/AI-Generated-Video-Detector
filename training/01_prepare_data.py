#!/usr/bin/env python3
"""Validate, split, and cache FakeAVCeleb audiovisual samples.

Colab setup (run once):
  !git clone https://github.com/dedlinux777/AI-Generated-Video-Detector.git
  %cd AI-Generated-Video-Detector
  !pip install -r requirements-training.txt
  !apt-get -qq update && apt-get -qq install -y ffmpeg
  from google.colab import drive; drive.mount('/content/drive')
  !python training/01_prepare_data.py --dataset-root /content/drive/MyDrive/dataset \
      --manifest /content/drive/MyDrive/dataset/etmc_multimodal_manifest.csv
"""
from __future__ import annotations
import argparse, json, subprocess, hashlib, sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import GroupShuffleSplit

CATEGORY_MAP = {
    "realvideo-realaudio": (0, 0, 0), "fakevideo-fakeaudio": (1, 1, 1),
    "fakevideo-realaudio": (1, 0, 1), "realvideo-fakeaudio": (0, 1, 1),
}
ALIASES = {
    "video_path": ("video_path", "video", "video_file", "videopath", "filepath"),
    "audio_path": ("audio_path", "audio", "audio_file", "audiopath"),
    "visual_label": ("visual_label", "video_label", "visual", "videolabel"),
    "audio_label": ("audio_label", "audiolabel"),
    "overall_label": ("overall_label", "label", "target", "overall"),
    "category": ("category", "type", "manipulation_type"),
    "actor_id": ("actor_id", "actor", "identity", "person_id", "source_id"),
}

def canonicalize(frame):
    norm = {str(c).strip().lower().replace(" ", "_"): c for c in frame.columns}
    result, found = frame.copy(), {}
    for target, aliases in ALIASES.items():
        source = next((norm[a] for a in aliases if a in norm), None)
        if source is not None:
            found[target] = source
            if source != target: result[target] = frame[source]
    if "video_path" not in result: raise ValueError(f"Manifest needs video path column; found: {list(frame.columns)}")
    # A standalone audio field is optional: audio is extracted from video otherwise.
    for c in ("visual_label", "audio_label", "overall_label"):
        if c not in result: result[c] = np.nan
    if "category" not in result: result["category"] = ""
    if "actor_id" not in result: result["actor_id"] = ""
    return result, found

def binary(value):
    if pd.isna(value): return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("real", "0"): return 0
        if value in ("fake", "1"): return 1
        return None
    try:
        n = int(value)
        return n if n in (0, 1) and float(value) == n else None
    except Exception: return None

def resolve_path(value, root):
    if value is None or pd.isna(value) or not str(value).strip(): return None
    path = Path(str(value).strip())
    return path if path.is_absolute() else root / path

def probe_video(path):
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    if ok: ok, _ = cap.read()
    cap.release()
    return bool(ok)

def frame_crop(frame, detector):
    h, w = frame.shape[:2]
    if detector is not None:
        try:
            from PIL import Image
            boxes, _ = detector.detect(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if boxes is not None and len(boxes):
                x1,y1,x2,y2 = np.maximum(boxes[0].astype(int), 0)
                x2,y2 = min(x2,w),min(y2,h)
                if x2>x1 and y2>y1: return frame[y1:y2,x1:x2]
        except Exception: pass
    # Documented fallback: centered square crop; if MTCNN is unavailable or no face is found.
    side = min(h, w); y, x = (h-side)//2, (w-side)//2
    return frame[y:y+side, x:x+side]

def extract_audio(video_path, audio_path, target_rate, duration, start_seconds=0.0):
    source="standalone_audio"
    if audio_path and audio_path.is_file():
        try: samples, sr = sf.read(str(audio_path), dtype="float32", always_2d=True); samples=samples.mean(1)
        except Exception as e: samples, sr = None, None; source=f"video_fallback_after_audio_decode_error: {type(e).__name__}"
    else:
        samples, sr = None, None
        source="video_fallback_audio_path_missing" if audio_path else "video_extracted_no_standalone_path"
    if samples is None:
        cmd=["ffmpeg","-v","error"]
        if start_seconds > 0: cmd += ["-ss",str(start_seconds)]
        cmd += ["-i",str(video_path),"-vn","-ac","1","-ar",str(target_rate),"-f","f32le","pipe:1"]
        try:
            raw=subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120).stdout
            samples=np.frombuffer(raw,dtype="<f4").copy(); sr=target_rate
        except Exception as e: raise RuntimeError(f"Audio decode/extraction failed: {e}") from e
    if sr != target_rate:
        # Polyphase resampling avoids librosa/numba compatibility issues.
        from scipy.signal import resample_poly
        from math import gcd
        d=gcd(int(sr),int(target_rate)); samples=resample_poly(samples,target_rate//d,int(sr)//d).astype(np.float32)
    if not len(samples): raise RuntimeError("Audio track is empty")
    start=int(max(0.0,start_seconds)*target_rate)
    samples=samples[start:] if start < len(samples) else np.empty(0,dtype=np.float32)
    if not len(samples): raise RuntimeError("Audio window is outside the decoded track")
    n=int(target_rate*duration)
    if len(samples)<n: samples=np.pad(samples,(0,n-len(samples)))
    else: samples=samples[:n]
    peak=float(np.max(np.abs(samples)))
    if not np.isfinite(samples).all() or peak < 1e-7: raise RuntimeError("Audio is silent or non-finite")
    # Keep source amplitude; RawNet uses its own learned front end.
    return np.asarray(samples,dtype=np.float32), {"source_rate":int(sr),"peak":peak,"rms":float(np.sqrt(np.mean(samples**2))),"source":source}

def split_groups(frame, group_col, seed):
    groups=frame[group_col].astype(str).to_numpy(); all_idx=np.arange(len(frame))
    if len(np.unique(groups)) < 3: raise ValueError("At least three independent source groups are required for train/val/test.")
    labels=frame["category"].astype(str).str.lower().to_numpy()
    def score(idx, target):
        counts=pd.Series(labels[idx]).value_counts(normalize=True)
        return abs(len(idx)/len(frame)-target)+sum(abs(counts.get(k,0)-v) for k,v in pd.Series(labels).value_counts(normalize=True).items())*.2
    # Deterministic random group holdouts; select closest category/size balance.
    best=None
    for s in range(seed,seed+200):
        a, rem=next(GroupShuffleSplit(n_splits=1,test_size=.30,random_state=s).split(all_idx,groups=groups))
        tr_target=0.70
        if best is None or score(a,tr_target)<best[0]: best=(score(a,tr_target),a,rem)
    _,train,remain=best
    remgroups=groups[remain]
    best=None
    for s in range(seed+200,seed+400):
        va,te=next(GroupShuffleSplit(n_splits=1,test_size=.50,random_state=s).split(remain,groups=remgroups))
        val,test=remain[va],remain[te]
        cost=score(val,.15)+score(test,.15)
        if best is None or cost<best[0]: best=(cost,val,test)
    return train,*best[1:]

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root",required=True); p.add_argument("--manifest",default=None)
    p.add_argument("--output-root",default="prepared_dataset"); p.add_argument("--frames",type=int,default=3)
    p.add_argument("--audio-seconds",type=float,default=3.0); p.add_argument("--sample-rate",type=int,default=16000)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--limit",type=int,default=0,help="Smoke mode: process only N valid samples")
    p.add_argument("--no-mtcnn",action="store_true",help="Use centered-square crop fallback only")
    args=p.parse_args(); root=Path(args.dataset_root).expanduser().resolve()
    if args.frames < 1: p.error("--frames must be at least 1")
    if args.audio_seconds <= 0: p.error("--audio-seconds must be positive")
    if args.sample_rate != 16000: p.error("RawNet in this repository is configured for 16,000 Hz; --sample-rate must remain 16000")
    manifest=Path(args.manifest).expanduser() if args.manifest else root/"etmc_multimodal_manifest.csv"
    if not manifest.is_absolute(): manifest=root/manifest
    out=Path(args.output_root); cache=out/"cache"; cache.mkdir(parents=True,exist_ok=True)
    if not manifest.exists(): p.error(f"Manifest does not exist: {manifest}")
    raw=pd.read_csv(manifest); frame,columns=canonicalize(raw)
    errors=[]; normalized=[]; seen=set(); seen_video=set(); seen_audio=set(); dup_paths=[]; detector=None
    if not args.no_mtcnn:
        try:
            from facenet_pytorch import MTCNN
            import torch
            detector=MTCNN(keep_all=True,device="cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e: print(f"MTCNN unavailable ({e}); using centered-square crop fallback.")
    for i,row in frame.iterrows():
        issue=[]; video=resolve_path(row.video_path,root); audio=resolve_path(row.audio_path,root) if "audio_path" in frame else None
        category=str(row.category).strip().lower()
        raw_labels=[row.visual_label,row.audio_label,row.overall_label]
        labels=[binary(v) for v in raw_labels]
        for name,value,parsed in zip(("visual_label","audio_label","overall_label"),raw_labels,labels):
            if not pd.isna(value) and str(value).strip() and parsed is None: issue.append(f"{name} is not a valid binary label: {value!r}")
        expected=CATEGORY_MAP.get(category.replace(" ",""))
        if expected is None: issue.append(f"unknown/unsupported category: {row.category!r}")
        if expected and all(v is not None for v in labels) and tuple(labels)!=expected: issue.append(f"labels {labels} conflict with category {row.category!r} expected {expected}")
        if expected:
            labels=[expected[j] if labels[j] is None else labels[j] for j in range(3)]
        if any(v is None for v in labels): issue.append("missing/invalid binary labels or unknown category")
        if video is None or not video.is_file(): issue.append(f"video missing: {video}")
        elif not probe_video(video): issue.append(f"video corrupt/unreadable: {video}")
        # Missing/corrupt standalone audio falls back to extraction from the video.
        identity=str(row.actor_id).strip() if pd.notna(row.actor_id) else ""
        # Filename-derived fallback groups likely alternate video/audio renders by same source.
        if not identity and video:
            identity=video.stem.split("_")[0] if video.stem else ""
        record={"row_index":int(i),"video_path":str(video) if video else "","audio_path":str(audio) if audio else "","warnings":[],
                "visual_label":labels[0],"audio_label":labels[1],"overall_label":labels[2],"category":str(row.category),
                "actor_id":str(row.actor_id) if pd.notna(row.actor_id) else "","source_identity":identity}
        if video and video.is_file() and not any(x.startswith("video corrupt") for x in issue):
            digest=hashlib.sha1((str(video.resolve())+"|"+str(audio.resolve() if audio else "")).encode()).hexdigest()
            duplicates=[]
            if digest in seen: duplicates.append("duplicate audiovisual path pair")
            if str(video.resolve()).lower() in seen_video: duplicates.append("duplicate video path")
            if audio and audio.is_file() and str(audio.resolve()).lower() in seen_audio: duplicates.append("duplicate audio path")
            if duplicates: dup_paths.append(int(i)); issue.extend(duplicates)
            seen.add(digest)
            seen_video.add(str(video.resolve()).lower())
            if audio and audio.is_file(): seen_audio.add(str(audio.resolve()).lower())
            try:
                cap=cv2.VideoCapture(str(video)); n=max(1,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                fps=float(cap.get(cv2.CAP_PROP_FPS) or 0); video_seconds=(n/fps) if fps>0 else 0.0
                segment_start=max(0.0,(video_seconds-args.audio_seconds)/2.0)
                segment_end=min(video_seconds,segment_start+args.audio_seconds) if video_seconds else video_seconds
                if video_seconds and fps>0:
                    positions=np.clip((np.linspace(segment_start,segment_end,args.frames)*fps).round().astype(int),0,n-1)
                else: positions=np.linspace(0,max(0,n-1),args.frames).round().astype(int)
                frames=[]
                for pos in positions:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,int(pos)); ok,bgr=cap.read()
                    if not ok or bgr is None: continue
                    face=frame_crop(bgr,detector); rgb=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
                    frames.append(cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA))
                cap.release()
                if not frames: raise RuntimeError("No decodable video frames")
                while len(frames)<args.frames: frames.append(frames[-1].copy())
                wav,audio_meta=extract_audio(video,audio,args.sample_rate,args.audio_seconds,segment_start)
                if audio_meta["source"] != "standalone_audio": record["warnings"].append(f"audio source fallback used: {audio_meta['source']}")
                stem=hashlib.sha1(str(video.resolve()).encode()).hexdigest()
                fpath=cache/f"{stem}_frames.npy"; apath=cache/f"{stem}_audio.npy"
                np.save(fpath,np.stack(frames[:args.frames]).astype(np.uint8)); np.save(apath,wav)
                record.update(cache_frames=str(fpath.resolve()),cache_audio=str(apath.resolve()),audio_peak=audio_meta["peak"],audio_rms=audio_meta["rms"])
            except Exception as e: issue.append(f"preprocessing failed: {type(e).__name__}: {e}")
        record["issues"]=issue; normalized.append(record)
        if issue: errors.append({"row_index":int(i),"issues":issue})
        if args.limit and sum(not r.get("issues") for r in normalized)>=args.limit: break
        if (i+1)%100==0: print(f"Validated/prepared {i+1}/{len(frame)} rows")
    table=pd.DataFrame(normalized); valid=table[table.issues.map(lambda x: len(x)==0)].copy()
    if valid.empty: raise RuntimeError("No valid rows were prepared; see validation report.")
    group_col="actor_id" if valid.actor_id.astype(str).str.strip().ne("").all() else "source_identity"
    identity_source="manifest_identity_column" if "actor_id" in columns else "filename_derived_fallback"
    missing_manifest_identities=int(valid.actor_id.astype(str).str.strip().isin(["", "nan", "None"]).sum()) if "actor_id" in columns else len(valid)
    unresolved_rows=valid.loc[valid.actor_id.astype(str).str.strip().isin(["", "nan", "None"]),["row_index","video_path"]].to_dict("records") if "actor_id" in columns else valid[["row_index","video_path"]].to_dict("records")
    unresolved=valid[group_col].astype(str).str.strip().eq("")
    if unresolved.any():
        # Unique-row grouping is an explicit fallback and is not identity-independent evaluation.
        valid.loc[unresolved,group_col]=valid.loc[unresolved,"video_path"].map(lambda x:"unresolved:"+hashlib.sha1(x.encode()).hexdigest())
    tr,va,te=split_groups(valid,group_col,args.seed)
    split_idx={"train":tr,"validation":va,"test":te}; summaries={}
    def dist(series): return {str(k):int(v) for k,v in series.value_counts(dropna=False).items()}
    for name,idx in split_idx.items():
        part=valid.iloc[idx].copy(); part.drop(columns=["issues"],errors="ignore").to_csv(out/f"{name}.csv",index=False)
        summaries[name]={"samples":len(part),"unique_actors_or_sources":int(part[group_col].nunique()),
                         "category_distribution":dist(part.category),"visual_class_distribution":dist(part.visual_label),
                         "audio_class_distribution":dist(part.audio_label),"overall_class_distribution":dist(part.overall_label)}
    splits={k:set(valid.iloc[idx][group_col].astype(str)) for k,idx in split_idx.items()}
    overlap={f"{a}_{b}":sorted(splits[a]&splits[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))}
    summary={"seed":args.seed,"grouping_column":group_col,"identity_source":identity_source,"group_overlap":overlap,"splits":summaries,
             "unresolved_source_identities":missing_manifest_identities,"unresolved_source_identity_rows":unresolved_rows,"identity_independent_claim_supported":bool(identity_source == "manifest_identity_column" and missing_manifest_identities==0),
             "preprocessing":{"frames":args.frames,"frame_size":[256,256],"frame_color":"RGB","image_normalization":"ETMC applies timm tf_efficientnet_b4_ns config: mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)","audio_sample_rate":args.sample_rate,"audio_seconds":args.audio_seconds}}
    (out/"split_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    warnings=[{"row_index":int(r["row_index"]),"warnings":r["warnings"]} for r in normalized if r.get("warnings")]
    report={"manifest":str(manifest),"original_rows":len(raw),"processed_rows":len(table),"valid_rows":len(valid),"invalid_rows":errors,"rows_with_warnings":warnings,
            "duplicate_path_rows":dup_paths,"detected_columns":columns,"source_columns":list(raw.columns),"alias_mapping":{k:str(v) for k,v in columns.items()},
            "validation_policy":"Invalid rows are reported and excluded from prepared splits; they are not silently discarded.","mtcnn_used":detector is not None}
    (out/"data_validation_report.json").write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    print(f"Prepared {len(valid)} valid samples; {len(errors)} rows have recorded issues. Output: {out.resolve()}")
    print(f"Group key: {group_col}; overlap: { {k:len(v) for k,v in overlap.items()} }")
if __name__=="__main__": main()
