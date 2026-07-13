
import os, gzip, time, json, math, random, bisect
import numpy as np, pandas as pd, torch
import torch.nn as nn
from collections import defaultdict
from pyfaidx import Fasta
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, average_precision_score
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)
SEED=1234; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
ROOT="/home/stark/Claude_Science_Hackathon/glm_bench"
GEN="/home/stark/Claude_Science_Hackathon/genome/hg38.fa"
PK="/home/stark/nucel_atac/peaks.bed.gz"
WIN=1000; HALF=500; N_POS=175000   # 50x of 3500
DATA=f"{ROOT}/unified_dataset_50x.tsv"

# ---------- BUILD ----------
if not os.path.exists(DATA):
    log("building 50x dataset, target pos", N_POS)
    fa=Fasta(GEN); sizes={c:len(fa[c]) for c in fa.keys()}
    MAIN=set("chr"+str(i) for i in range(1,23))|{"chrX"}
    TESTCHR={"chr1","chr2"}
    # train-eligible peaks: main chr, NOT chr1/chr2, in-bounds, summit-centered
    peaks=[]
    with gzip.open(PK,'rt') as f:
        for line in f:
            p=line.rstrip("\n").split("\t"); chrom=p[0]
            if chrom not in MAIN or chrom in TESTCHR: continue
            start=int(p[1]); end=int(p[2]); sig=float(p[6]); off=int(p[9])
            summit=start+off if off>=0 else (start+end)//2
            s=summit-HALF; e=summit+HALF
            if s<0 or e>sizes.get(chrom,0): continue
            peaks.append((chrom,s,e,sig))
    log("train-eligible peaks in-bounds:", len(peaks))
    random.shuffle(peaks); peaks=peaks[:N_POS]
    # peak intervals (all chroms) for negative-overlap exclusion
    peak_iv=defaultdict(list)
    with gzip.open(PK,'rt') as f:
        for line in f:
            p=line.rstrip("\n").split("\t"); chrom=p[0]
            if chrom not in MAIN: continue
            peak_iv[chrom].append((int(p[1])-HALF,int(p[2])+HALF))
    for ch in peak_iv: peak_iv[ch].sort()
    peak_starts={ch:[x[0] for x in peak_iv[ch]] for ch in peak_iv}  # precompute once
    def overlaps(chrom,s,e):
        ivs=peak_iv.get(chrom,[])
        if not ivs: return False
        starts=peak_starts[chrom]; i=bisect.bisect_right(starts,e)
        for j in range(max(0,i-1),min(len(ivs),i+1)):
            a,b=ivs[j]
            if s<b and a<e: return True
        return False
    def gc(seq):
        seq=seq.upper(); n=len(seq); return (seq.count("G")+seq.count("C"))/n if n else 0.0
    def getseq(chrom,s,e): return str(fa[chrom][s:e])
    pos=[]
    for chrom,s,e,sig in peaks:
        seq=getseq(chrom,s,e)
        if seq.upper().count("N")>0.05*WIN: continue
        pos.append((chrom,gc(seq),sig,seq))
    log("positives after N-filter:", len(pos))
    pos_gc=np.array([x[1] for x in pos]); bins=np.linspace(0,1,21)
    pos_hist,_=np.histogram(pos_gc,bins=bins)
    # GC-matched negatives from non-test chroms
    chroms=[c for c in MAIN if c in sizes and c not in TESTCHR]
    need=pos_hist.copy(); neg=[]; tries=0; maxtries=200000000
    last_rem=int(need.sum()); stall=0
    while need.sum()>0 and tries<maxtries:
        tries+=1
        chrom=random.choice(chroms); L=sizes[chrom]
        s=random.randint(0,L-WIN); e=s+WIN
        if overlaps(chrom,s,e): continue
        seq=getseq(chrom,s,e)
        if seq.upper().count("N")>0.05*WIN: continue
        g=gc(seq); b=max(0,min(np.digitize(g,bins)-1,19))
        if need[b]>0: neg.append((chrom,g,seq)); need[b]-=1
        if tries%5000000==0:
            rem=int(need.sum()); log("  neg sampling tries",tries,"remaining",rem)
            if rem==last_rem and rem<=10: stall+=1
            else: stall=0
            last_rem=rem
            if stall>=1: log("  stall on",rem,"rare-GC-bin negs; accepting shortfall"); break
    log("negatives sampled:", len(neg), "tries", tries, "remaining", int(need.sum()))
    # train rows
    rows=[]
    for i,(chrom,g,sig,seq) in enumerate(pos): rows.append((f"pos50_{i}",1,chrom,"train",round(g,4),round(sig,4),seq))
    for i,(chrom,g,seq) in enumerate(neg): rows.append((f"neg50_{i}",0,chrom,"train",round(g,4),0.0,seq))
    random.shuffle(rows)
    # append byte-identical test block from 10x file
    base=pd.read_csv(f"{ROOT}/unified_dataset_10x.tsv",sep="\t")
    test=base[base.split=="test"]
    with open(DATA,"w") as out:
        out.write("id\tlabel\tchrom\tsplit\tgc\tsignal\tseq\n")
        for r in rows: out.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t{r[5]}\t{r[6]}\n")
        for _,t in test.iterrows(): out.write(f"{t.id}\t{t.label}\t{t.chrom}\t{t.split}\t{t.gc}\t{t.signal}\t{t.seq}\n")
    log("wrote",DATA)
else:
    log("dataset exists, reuse")

df=pd.read_csv(DATA,sep="\t")
tr_full=df[df.split=="train"].reset_index(drop=True); te=df[df.split=="test"].reset_index(drop=True)
log(f"train {len(tr_full)} pos {int(tr_full.label.sum())} | test {len(te)} pos {int(te.label.sum())}")
import hashlib
tmd5=hashlib.md5(("".join(te.sort_values('id').id+te.sort_values('id').seq)).encode()).hexdigest()
log("test signature", tmd5)

# ---------- TRAIN (mean-pool, lr=1.5e-4, best config) ----------
LR=1.5e-4; DEV="cuda"; MAXLEN=1024; BATCH=16; EPOCHS=6; PATIENCE=4; WARMUP_FRAC=0.10
CKPT=f"{ROOT}/scaling/ckpt_50x"; os.makedirs(CKPT,exist_ok=True)
rng=np.random.RandomState(SEED); val_idx=[]
for lab in [0,1]:
    idx=tr_full.index[tr_full.label==lab].to_numpy().copy(); rng.shuffle(idx)
    val_idx.extend(idx[:max(1,int(0.1*len(idx)))].tolist())
vm=tr_full.index.isin(val_idx); va=tr_full[vm].reset_index(drop=True); tr=tr_full[~vm].reset_index(drop=True)
log(f"fit {len(tr)} val {len(va)} test {len(te)}")
tok=AutoTokenizer.from_pretrained("FreakingPotato/NucEL",trust_remote_code=True)
cfg=AutoConfig.from_pretrained("/home/stark/NucEL",trust_remote_code=True); cfg.reference_compile=False
bb=AutoModel.from_pretrained("/home/stark/NucEL",trust_remote_code=True,config=cfg)
class DS(torch.utils.data.Dataset):
    def __init__(s,fr): s.id=fr.id.tolist(); s.s=fr.seq.tolist(); s.y=fr.label.tolist()
    def __len__(s): return len(s.s)
    def __getitem__(s,i): return s.id[i],s.s[i],int(s.y[i])
def coll(b):
    enc=tok([x[1] for x in b],return_tensors="pt",padding=True,truncation=True,max_length=MAXLEN)
    return [x[0] for x in b],enc,torch.tensor([x[2] for x in b])
class Clf(nn.Module):
    def __init__(s,bb): super().__init__(); s.backbone=bb; s.drop=nn.Dropout(0.1); s.head=nn.Linear(512,2)
    def forward(s,i,a):
        h=s.backbone(input_ids=i,attention_mask=a).last_hidden_state
        m=a.unsqueeze(-1).to(h.dtype); return s.head(s.drop((h*m).sum(1)/m.sum(1).clamp(min=1.0)))
model=Clf(bb).to(DEV); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01); lf=nn.CrossEntropyLoss()
trdl=torch.utils.data.DataLoader(DS(tr),batch_size=BATCH,shuffle=True,collate_fn=coll,num_workers=4)
vadl=torch.utils.data.DataLoader(DS(va),batch_size=64,shuffle=False,collate_fn=coll,num_workers=4)
tedl=torch.utils.data.DataLoader(DS(te),batch_size=64,shuffle=False,collate_fn=coll,num_workers=4)
spe=len(trdl); tot=spe*EPOCHS; sched=get_linear_schedule_with_warmup(opt,int(WARMUP_FRAC*tot),tot); ev=math.ceil(spe/3)
log(f"steps/epoch {spe} total {tot} eval_every {ev}")
def ev_dl(dl):
    model.eval(); P=[]; Y=[]
    with torch.no_grad():
        for _,enc,y in dl:
            enc={k:v.to(DEV) for k,v in enc.items()}
            with torch.autocast("cuda",dtype=torch.bfloat16): lo=model(enc["input_ids"],enc["attention_mask"])
            P+=torch.softmax(lo.float(),1)[:,1].cpu().tolist(); Y+=y.tolist()
    return np.array(P),np.array(Y)
best=-1; bstep=-1; bad=0; g=0; stop=False; t0=time.time()
for ep in range(1,EPOCHS+1):
    model.train(); wl=0; wn=0
    for i,(_,enc,y) in enumerate(trdl):
        enc={k:v.to(DEV) for k,v in enc.items()}; y=y.to(DEV)
        with torch.autocast("cuda",dtype=torch.bfloat16): lo=model(enc["input_ids"],enc["attention_mask"]); loss=lf(lo,y)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
        g+=1; wl+=loss.item(); wn+=1
        if (i+1)%ev==0 or (i+1)==spe:
            vp,vy=ev_dl(vadl); va_auc=roc_auc_score(vy,vp)
            log(f"ep{ep} {i+1}/{spe} g{g} loss {wl/max(1,wn):.4f} val_auROC {va_auc:.4f}"); wl=0; wn=0; model.train()
            if va_auc>best: best=va_auc; bstep=g; bad=0; torch.save({"model":model.state_dict(),"gstep":g,"val_auroc":va_auc},f"{CKPT}/best.pt")
            else:
                bad+=1
                if bad>=PATIENCE: log("early stop"); stop=True; break
    if stop: break
tsec=time.time()-t0; log(f"train done {tsec:.1f}s best_step {bstep} val {best:.4f}")
ck=torch.load(f"{CKPT}/best.pt",map_location=DEV); model.load_state_dict(ck["model"])
tp,ty=ev_dl(tedl); auroc=roc_auc_score(ty,tp); auprc=average_precision_score(ty,tp)
# bootstrap CI
rs=np.random.RandomState(0); bs=[]
for _ in range(1000):
    idx=rs.randint(0,len(ty),len(ty))
    if len(np.unique(ty[idx]))<2: continue
    bs.append(roc_auc_score(ty[idx],tp[idx]))
ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
out=dict(tag="50x",n_train=int(len(tr_full)),n_train_fit=int(len(tr)),n_val=int(len(va)),n_test=int(len(te)),
         lr=LR,best_val_auROC=float(best),best_step=int(bstep),test_auROC=float(auroc),test_auPRC=float(auprc),
         ci_auROC=ci,train_seconds=float(tsec),test_md5=tmd5)
json.dump(out,open(f"{ROOT}/scaling/nucel_50x.json","w"),indent=1)
log("RESULT",json.dumps(out)); log("RUN_DONE")
