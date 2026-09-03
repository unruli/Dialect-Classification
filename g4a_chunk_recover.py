import os, sys, csv, subprocess, tempfile
sys.path.insert(0, os.path.join("/blue/ufdatastudios/c.okocha/Dialect-Classification","inference"))
R="/blue/ufdatastudios/c.okocha/Dialect-Classification"
os.environ["HF_HOME"]=f"{R}/runs/architecture_audit/G4-A/cache"; os.environ["HF_HUB_OFFLINE"]="1"
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import build_transcription_messages, generate_transcription, resolve_device
from common import rttm_tools

TARGETS=["EN2002c","herring01","herring07","herring17","sastre09","sastre10","zeledon06","zeledon08"]
CHUNK=480.0; OVERLAP=30.0; MAXTOK=65536
manifest={r["recording_id"]:r for r in csv.DictReader(open(f"{R}/data/inference_ready/manifest.csv"))}
seldur={r["recording_id"]:float(r["audio_duration_sec"]) for r in csv.DictReader(open(f"{R}/dataset_metadata/final_evaluation_manifest.csv"))}

device=resolve_device("auto"); dtype=torch.bfloat16 if device.type=="cuda" else torch.float32
print("device",device,"cuda",torch.cuda.is_available(),flush=True)
mid="OpenMOSS-Team/MOSS-Transcribe-Diarize"
model=AutoModelForCausalLM.from_pretrained(mid,trust_remote_code=True,dtype="auto",attn_implementation="sdpa").to(dtype=dtype).to(device).eval()
proc=AutoProcessor.from_pretrained(mid,trust_remote_code=True)

def infer(wav):
    msgs=build_transcription_messages(wav)
    g=generate_transcription(model,proc,msgs,max_new_tokens=MAXTOK,do_sample=False,device=device,dtype=dtype)
    return [(float(s.start),float(s.end),s.speaker) for s in parse_transcript(g["text"])]

def overlap(a0,a1,b0,b1): return max(0.0,min(a1,b1)-max(a0,b0))

def link(prev_segs, cur_segs, ov0, ov1):
    # map cur local speaker -> global label using temporal overlap in [ov0,ov1]
    import collections
    scores=collections.defaultdict(float)
    for c0,c1,cs in cur_segs:
        if c1<=ov0 or c0>=ov1: continue
        for p0,p1,ps in prev_segs:
            if p1<=ov0 or p0>=ov1: continue
            scores[(cs,ps)]+=overlap(c0,c1,p0,p1)
    mapping={}; used=set()
    for (cs,ps),_ in sorted(scores.items(),key=lambda x:-x[1]):
        if cs in mapping or ps in used: continue
        mapping[cs]=ps; used.add(ps)
    return mapping

for rid in TARGETS:
    row=manifest[rid]; ds=row["dataset"]; wav=row["audio_path"]; dur=seldur[rid]
    starts=[]; t=0.0
    while t<dur: starts.append(t); t+=CHUNK-OVERLAP
    glob=[]; gcount=[0]
    def newg():
        gcount[0]+=1; return f"G{gcount[0]:02d}"
    for i,cs in enumerate(starts):
        clen=min(CHUNK, dur-cs)
        with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as tf: cw=tf.name
        subprocess.run(["/blue/ufdatastudios/c.okocha/envs/g3a_sortformer/bin/ffmpeg","-nostdin","-y","-ss",f"{cs}","-i",wav,"-t",f"{clen}","-ar","16000","-ac","1",cw],
                       capture_output=True)
        local=[(s+cs,e+cs,spk) for s,e,spk in infer(cw)]
        os.remove(cw)
        if i==0:
            m={spk:newg() for spk in dict.fromkeys(s[2] for s in local)}
            glob=[(s,e,m[spk]) for s,e,spk in local]
        else:
            ov0,ov1=cs, starts[i-1]+CHUNK
            mp=link(glob,local,ov0,ov1)
            for spk in dict.fromkeys(s[2] for s in local):
                if spk not in mp: mp[spk]=newg()
            # keep prev in overlap; add cur segments starting at/after ov0 midpoint to avoid dup
            cut=(ov0+ov1)/2
            glob+=[(s,e,mp[spk]) for s,e,spk in local if s>=cut]
        print(f"  {rid} chunk{i} [{cs:.0f},{cs+clen:.0f}] segs={len(local)}",flush=True)
    glob.sort()
    raw=f"{R}/runs/architecture_audit/G4-A/raw/{ds}/{rid}.g4a_moss.chunked.raw.rttm"
    os.makedirs(os.path.dirname(raw),exist_ok=True)
    with open(raw,"w") as f:
        for s,e,spk in glob:
            if e-s>0: f.write(f"SPEAKER {rid} 1 {s:.3f} {e-s:.3f} <NA> <NA> {spk} <NA> <NA>\n")
    outr=f"{R}/runs/architecture_audit/G4-A/rttm/{ds}/{rid}.rttm"
    try:
        nseg,nspk=rttm_tools.normalize_rttm_file(raw,outr,rid,source_duration_sec=dur)
        print(f"RECOVERED {rid}: chunks={len(starts)} segs={nseg} spk={nspk} maxts={max((e for _,e,_ in glob),default=0):.0f}/{dur:.0f}",flush=True)
    except Exception as ex:
        print(f"NORMALIZE_FAIL {rid}: {ex}",flush=True)
print("CHUNK_RECOVER_DONE",flush=True)
