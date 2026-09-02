import os, sys, csv, tempfile, collections, numpy as np, soundfile as sf, librosa
sys.path.insert(0, os.path.join("/blue/ufdatastudios/c.okocha/Dialect-Classification","inference"))
os.environ.setdefault("NEMO_CACHE_DIR","/blue/ufdatastudios/c.okocha/Dialect-Classification/runs/architecture_audit/G4-A/cache")
os.environ.setdefault("HF_HOME", os.environ["NEMO_CACHE_DIR"])
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from sklearn.cluster import AgglomerativeClustering
from common import rttm_tools
R="/blue/ufdatastudios/c.okocha/Dialect-Classification"
TARGETS=["EN2002c","herring01","herring07","herring17","sastre09","sastre10","zeledon06","zeledon08"]
GRID=[0.35,0.40,0.45,0.50,0.55,0.60,0.65]; MAX_LABEL_SEC=45.0; MIN_SEG=0.40
man={r["recording_id"]:r for r in csv.DictReader(open(f"{R}/data/inference_ready/manifest.csv"))}
ref={r["recording_id"]:int(r["num_speakers"]) for r in csv.DictReader(open(f"{R}/dataset_metadata/final_evaluation_manifest.csv")) if r.get("num_speakers")}
dur={r["recording_id"]:float(r["audio_duration_sec"]) for r in csv.DictReader(open(f"{R}/dataset_metadata/final_evaluation_manifest.csv"))}
print("loading TitaNet...",flush=True)
spk=EncDecSpeakerLabelModel.from_pretrained("titanet_large", map_location="cpu"); spk.eval()
def embed(w):
    with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as tf: p=tf.name
    sf.write(p,w,16000)
    try: e=spk.get_embedding(p).squeeze().cpu().numpy()
    finally: os.remove(p)
    return e/(np.linalg.norm(e)+1e-9)
# embed once per (file,label)
FILES={}
for rid in TARGETS:
    ds=man[rid]["dataset"]; raw=f"{R}/runs/architecture_audit/G4-A/raw/{ds}/{rid}.g4a_moss.chunked.raw.rttm"
    segs=[(float(f[3]),float(f[4]),f[7]) for f in (ln.split() for ln in open(raw)) if len(f)>=8 and f[0]=="SPEAKER"]
    y,_=librosa.load(man[rid]["audio_path"],sr=16000,mono=True)
    by=collections.defaultdict(list)
    for s,d,l in segs: by[l].append((s,d))
    labs=sorted(by); C=[]; emblab=[]
    for l in labs:
        acc=[]; tot=0
        for s,d in sorted(by[l]):
            if d<MIN_SEG: continue
            a=y[int(s*16000):int((s+d)*16000)]
            if len(a): acc.append(a); tot+=d
            if tot>=MAX_LABEL_SEC: break
        if tot>=0.5: C.append(embed(np.concatenate(acc))); emblab.append(l)
    FILES[rid]=dict(ds=ds,segs=segs,labs=labs,C=np.vstack(C),emblab=emblab)
    print(f"embedded {rid}: {len(labs)} labels",flush=True)
def kcount(C,thr):
    if len(C)==1: return np.array([0])
    return AgglomerativeClustering(n_clusters=None,metric="cosine",linkage="average",distance_threshold=thr).fit_predict(C)
# sweep: pick global thr minimizing sum|K-ref|
best=None
for thr in GRID:
    tot=0; ks={}
    for rid in TARGETS:
        k=len(set(kcount(FILES[rid]["C"],thr))); ks[rid]=k; tot+=abs(k-ref.get(rid,k))
    print(f"thr={thr}: sum|dev|={tot}  "+" ".join(f"{r}={ks[r]}" for r in TARGETS),flush=True)
    if best is None or tot<best[0]: best=(tot,thr,ks)
tot,thr,ks=best; print(f"CHOSEN thr={thr} (sum|dev|={tot})",flush=True)
# write final RTTMs at chosen thr
for rid in TARGETS:
    F=FILES[rid]; clus=kcount(F["C"],thr); lab2c={l:int(clus[i]) for i,l in enumerate(F["emblab"])}
    for l in F["labs"]:
        if l not in lab2c: lab2c[l]=0
    reid=f"{R}/runs/architecture_audit/G4-A/raw/{F['ds']}/{rid}.g4a_moss.reid.raw.rttm"
    with open(reid,"w") as f:
        for s,d,l in F["segs"]:
            if d>0: f.write(f"SPEAKER {rid} 1 {s:.3f} {d:.3f} <NA> <NA> SPK{lab2c[l]:02d} <NA> <NA>\n")
    outr=f"{R}/runs/architecture_audit/G4-A/rttm/{F['ds']}/{rid}.rttm"
    nseg,nspk=rttm_tools.normalize_rttm_file(reid,outr,rid,source_duration_sec=dur[rid])
    print(f"FINAL {rid}: {nspk} spk (ref {ref.get(rid)})",flush=True)
print("G4A_REID2_DONE")
