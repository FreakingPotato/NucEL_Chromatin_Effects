import os, sys, time, json
os.environ["TOKENIZERS_PARALLELISM"]="false"
import numpy as np, pandas as pd, torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

BASE="/home/stark/Claude_Science_Hackathon/glm_bench"
ROOT="/home/stark/Claude_Science_Hackathon"
DEV="cuda"
torch.cuda.set_device(0)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)

TARGETS=["rs6701713","rs10939105","rs59325138","rs148163066","rs4698412","rs4575098"]
HALF=50   # +-50 bp -> 100 positions (center-50 .. center+49)
BASES=list("ACGT"); B2I={b:i for i,b in enumerate(BASES)}

# ---------- load windows ----------
win={}
for f in ["AD_windows.tsv","PD_windows.tsv"]:
    d=pd.read_csv(f"{BASE}/{f}",sep="\t")
    for _,r in d.iterrows():
        if r.variant_id in TARGETS and r.variant_id not in win:
            win[r.variant_id]=dict(chrom=r.chrom,pos=int(r.pos),ref=r.ref,alt=r.alt,
                                   ref_seq=r.ref_seq,alt_seq=r.alt_seq)
for v in TARGETS:
    assert v in win, f"missing window {v}"
log("windows loaded for", list(win.keys()))

# find SNP index = single diff between ref_seq and alt_seq
for v in TARGETS:
    rs=win[v]["ref_seq"]; asq=win[v]["alt_seq"]
    assert len(rs)==len(asq), v
    diffs=[i for i in range(len(rs)) if rs[i]!=asq[i]]
    assert len(diffs)==1, f"{v} diffs {len(diffs)}"
    si=diffs[0]
    win[v]["snp_idx"]=si; win[v]["L"]=len(rs)
    assert rs[si]==win[v]["ref"] and asq[si]==win[v]["alt"], f"{v} allele mismatch {rs[si]} {asq[si]}"
log("snp idx per variant:", {v:win[v]["snp_idx"] for v in TARGETS})

# ---------- load NucEL classifier ----------
log("loading NucEL backbone + tokenizer")
tok=AutoTokenizer.from_pretrained("FreakingPotato/NucEL",trust_remote_code=True)
cfg=AutoConfig.from_pretrained("/home/stark/NucEL",trust_remote_code=True); cfg.reference_compile=False
bb=AutoModel.from_pretrained("/home/stark/NucEL",trust_remote_code=True,config=cfg)
class NucelClf(nn.Module):
    def __init__(self,bb,hidden=512,ndrop=0.1):
        super().__init__(); self.backbone=bb; self.drop=nn.Dropout(ndrop); self.head=nn.Linear(hidden,2)
    def forward(self,input_ids,attention_mask):
        out=self.backbone(input_ids=input_ids,attention_mask=attention_mask)
        h=out.last_hidden_state; m=attention_mask.unsqueeze(-1).to(h.dtype)
        pooled=(h*m).sum(1)/m.sum(1).clamp(min=1.0)
        return self.head(self.drop(pooled))
model=NucelClf(bb).to(DEV)
ck=torch.load(f"{BASE}/ckpt_nucel/best.pt",map_location=DEV); model.load_state_dict(ck["model"]); model.eval()
log("loaded ckpt val_auroc", ck.get("val_auroc"))

MAXLEN=1024
@torch.no_grad()
def ppeak(seqs, bs=32):
    out=[]
    for i in range(0,len(seqs),bs):
        chunk=seqs[i:i+bs]
        enc=tok(chunk,return_tensors="pt",padding=True,truncation=True,max_length=MAXLEN)
        enc={k:v.to(DEV) for k,v in enc.items()}
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logits=model(enc["input_ids"],enc["attention_mask"])
        p=torch.softmax(logits.float(),dim=1)[:,1].cpu().numpy()
        out.append(p)
    return np.concatenate(out)

# ---------- SATURATION ISM ----------
log("=== saturation ISM ===")
ism={}
for v in TARGETS:
    rs=list(win[v]["ref_seq"]); si=win[v]["snp_idx"]; Lseq=win[v]["L"]
    start=si-HALF; positions=list(range(start, start+2*HALF))  # 100 positions
    p_ref=float(ppeak(["".join(rs)])[0])
    # build all mutant seqs
    mut_seqs=[]; meta=[]  # (pos_local, base_idx)
    for pi,pos in enumerate(positions):
        refb=rs[pos]
        for b in BASES:
            if b==refb: continue
            s=rs.copy(); s[pos]=b
            mut_seqs.append("".join(s)); meta.append((pi, B2I[b]))
    pm=ppeak(mut_seqs, bs=32)
    delta=np.zeros((2*HALF,4),dtype=np.float64)  # positions x ACGT
    for (pi,bi),pv in zip(meta, pm):
        delta[pi,bi]=pv-p_ref
    # importance = mean over 3 alt bases of (P(mut)-P(ref))
    imp=delta.sum(axis=1)/3.0
    ism[v]=dict(chrom=win[v]["chrom"],pos=win[v]["pos"],ref=win[v]["ref"],alt=win[v]["alt"],
                snp_idx=si, positions=positions, ref_bases="".join(rs[p] for p in positions),
                p_ref=p_ref, delta_ACGT=delta.tolist(), importance=imp.tolist())
    log(f"  {v}: p_ref={p_ref:.4f} imp min/max {imp.min():.4f}/{imp.max():.4f} |imp|_center={abs(imp[HALF]):.4f}")
json.dump(ism, open(f"{BASE}/ism_nucel_saturation.json","w"))
log("wrote ism_nucel_saturation.json")

del model, bb; torch.cuda.empty_cache()

# ---------- MOTIF (JASPAR2024 CORE vertebrates) ----------
log("=== motif matching ===")
from pyjaspar import jaspardb
jdb=jaspardb(release="JASPAR2024")
motifs=jdb.fetch_motifs(collection="CORE", tax_group="vertebrates")
log("n JASPAR motifs", len(motifs))
# precompute log-odds (4,L) arrays, ACGT order, bg 0.25
BG=0.25
mo=[]
for m in motifs:
    pwm=m.counts  # dict A/C/G/T -> tuple
    L=m.length
    arr=np.zeros((4,L))
    tot=np.array([pwm['A'][j]+pwm['C'][j]+pwm['G'][j]+pwm['T'][j] for j in range(L)])
    pc=0.25  # pseudocount
    for bi,b in enumerate("ACGT"):
        arr[bi]=np.log2(((np.array(pwm[b])+pc)/(tot+4*pc))/BG)
    mo.append((m.matrix_id, m.name, L, arr))

def seq_idx(s):
    return np.array([B2I.get(c,0) for c in s], dtype=int)

def best_overlap_score(idx_arr, snp, lo):  # lo: (4,L) log-odds
    L=lo.shape[1]
    starts=range(max(0,snp-L+1), min(len(idx_arr)-L, snp)+1)
    best=-1e9
    for s in starts:
        sc=lo[idx_arr[s:s+L], np.arange(L)].sum()
        if sc>best: best=sc
    return best

motif_rows=[]
for v in TARGETS:
    rs=win[v]["ref_seq"]; asq=win[v]["alt_seq"]; si=win[v]["snp_idx"]
    ridx=seq_idx(rs); aidx=seq_idx(asq)
    best=None
    for mid,name,L,lo in mo:
        # also try reverse complement of motif
        lo_rc=lo[::-1,::-1]
        sr=max(best_overlap_score(ridx,si,lo), best_overlap_score(ridx,si,lo_rc))
        sa=max(best_overlap_score(aidx,si,lo), best_overlap_score(aidx,si,lo_rc))
        d=sr-sa
        if best is None or abs(d)>abs(best[3]):
            best=(mid,name,L,d,sr,sa)
    mid,name,L,d,sr,sa=best
    motif_rows.append(dict(variant=v,top_motif=name,matrix_id=mid,motif_len=L,
                           ref_logodds=round(float(sr),4),alt_logodds=round(float(sa),4),
                           motif_delta=round(float(d),4)))
    log(f"  {v}: {name} ({mid}) delta={d:.4f} ref={sr:.3f} alt={sa:.3f}")
mdf=pd.DataFrame(motif_rows)
mdf.to_csv(f"{BASE}/motif_nucel.tsv",sep="\t",index=False)
log("wrote motif_nucel.tsv")

# ---------- CONSISTENCY vs ChromBPNet DeepSHAP ----------
log("=== consistency vs ChromBPNet ===")
from scipy.stats import pearsonr
npz=np.load(f"{ROOT}/results/shap/brain_top_shap.npz")
proj=npz["proj"]  # (32,4,2114)  rows 0-15 = ref (allele0)
tgt=pd.read_csv(f"{ROOT}/results/shap/shap_targets.tsv",sep="\t",header=None,
                names=["chrom","pos","ref","alt","variant_id"])
row_of={vid:i for i,vid in enumerate(tgt.variant_id.tolist())}  # 0..15 = ref rows
W=proj.shape[2]; cbp_center=W//2  # 1057
cons={}
rs_list=[]
for v in TARGETS:
    ri=row_of[v]  # ref DeepSHAP row (0-15)
    cbp=proj[ri].sum(axis=0)  # per-base scalar contribution, length 2114
    cbp_win=cbp[cbp_center-HALF:cbp_center+HALF]  # central 100bp
    nuc_imp=np.array(ism[v]["importance"])        # 100
    r,_=pearsonr(np.abs(nuc_imp), np.abs(cbp_win))
    cons[v]=float(r); rs_list.append(r)
    log(f"  {v}: r(|NucEL ISM|,|ChromBPNet SHAP|) = {r:.4f}")
mean_r=float(np.mean(rs_list))
log("MEAN r =", round(mean_r,4))

# ---------- assemble per_variant + save consistency json ----------
per_variant=[]
for row in motif_rows:
    v=row["variant"]
    per_variant.append(dict(variant=v, top_motif=row["top_motif"], matrix_id=row["matrix_id"],
                            motif_delta=row["motif_delta"],
                            ref_logodds=row["ref_logodds"], alt_logodds=row["alt_logodds"],
                            r_vs_chrombpnet=round(cons[v],4)))
result=dict(per_variant=per_variant, mean_r_vs_chrombpnet=round(mean_r,4))
json.dump(result, open(f"{BASE}/interp_result.json","w"), indent=2)
log("wrote interp_result.json")
log("ALL_DONE")
