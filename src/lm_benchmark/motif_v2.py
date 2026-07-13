import json, numpy as np, pandas as pd
BASE="/home/stark/Claude_Science_Hackathon/glm_bench"
TARGETS=["rs6701713","rs10939105","rs59325138","rs148163066","rs4698412","rs4575098"]
B2I={b:i for i,b in enumerate("ACGT")}

# windows
win={}
for f in ["AD_windows.tsv","PD_windows.tsv"]:
    d=pd.read_csv(f"{BASE}/{f}",sep="\t")
    for _,r in d.iterrows():
        if r.variant_id in TARGETS and r.variant_id not in win:
            rs=r.ref_seq; asq=r.alt_seq
            si=[i for i in range(len(rs)) if rs[i]!=asq[i]][0]
            win[r.variant_id]=dict(ref_seq=rs,alt_seq=asq,snp_idx=si)

ism=json.load(open(f"{BASE}/ism_nucel_saturation.json"))

from pyjaspar import jaspardb
jdb=jaspardb(release="JASPAR2024")
motifs=jdb.fetch_motifs(collection="CORE", tax_group="vertebrates")
BG=0.25; pc=0.25
mo=[]
for m in motifs:
    pwm=m.counts; L=m.length
    tot=np.array([sum(pwm[b][j] for b in "ACGT") for j in range(L)])
    arr=np.zeros((4,L))
    for bi,b in enumerate("ACGT"):
        arr[bi]=np.log2(((np.array(pwm[b])+pc)/(tot+4*pc))/BG)
    mo.append((m.matrix_id, m.name, L, arr))

def seq_idx(s): return np.array([B2I.get(c,0) for c in s],dtype=int)

def best_over_snp(idx_arr, snp, lo):
    """best log-odds AND relative score over windows overlapping snp; return (score, relscore)"""
    L=lo.shape[1]
    mx=lo.max(0).sum(); mn=lo.min(0).sum(); rng=mx-mn if mx>mn else 1.0
    lo_all=[lo, lo[::-1,::-1]]  # fwd + revcomp
    best_sc=-1e9; best_rel=-1e9
    for l in lo_all:
        for s in range(max(0,snp-L+1), min(len(idx_arr)-L, snp)+1):
            sc=l[idx_arr[s:s+L], np.arange(L)].sum()
            if sc>best_sc:
                best_sc=sc; best_rel=(sc-mn)/rng
    return best_sc, best_rel

REL_THR=0.80   # motifbreakR-style "strong match" threshold
rows=[]
for v in TARGETS:
    rs=win[v]["ref_seq"]; asq=win[v]["alt_seq"]; si=win[v]["snp_idx"]
    ridx=seq_idx(rs); aidx=seq_idx(asq)
    cands=[]
    for mid,name,L,lo in mo:
        sr,rr=best_over_snp(ridx,si,lo)
        sa,ra=best_over_snp(aidx,si,lo)
        rel=max(rr,ra)               # best-allele relative match
        d=sr-sa                      # ref->alt log-odds change (bits)
        cands.append((mid,name,L,sr,sa,d,rel,rr,ra))
    # among motifs that are a strong match in at least one allele, pick largest |delta|
    strong=[c for c in cands if c[6]>=REL_THR]
    pool = strong if strong else cands
    best=max(pool, key=lambda c: abs(c[5]))
    mid,name,L,sr,sa,d,rel,rr,ra=best
    rows.append(dict(variant=v, top_motif=name, matrix_id=mid, motif_len=L,
                     ref_logodds=round(float(sr),3), alt_logodds=round(float(sa),3),
                     motif_delta=round(float(d),3),
                     rel_score=round(float(rel),3),
                     n_strong=len(strong),
                     r_vs_chrombpnet=None))
    print(f"{v}: {name} ({mid}) delta={d:.3f} ref={sr:.2f} alt={sa:.2f} rel={rel:.3f} strong={len(strong)}", flush=True)

mdf=pd.DataFrame(rows)
mdf.to_csv(f"{BASE}/motif_nucel.tsv",sep="\t",index=False)
# merge r into per_variant from prior interp_result.json
prev=json.load(open(f"{BASE}/interp_result.json"))
rmap={pv["variant"]:pv["r_vs_chrombpnet"] for pv in prev["per_variant"]}
per=[]
for row in rows:
    row2=dict(row); row2["r_vs_chrombpnet"]=rmap[row["variant"]]; per.append(row2)
result=dict(per_variant=per, mean_r_vs_chrombpnet=prev["mean_r_vs_chrombpnet"])
json.dump(result, open(f"{BASE}/interp_result.json","w"), indent=2)
print("DONE mean_r", prev["mean_r_vs_chrombpnet"], flush=True)
