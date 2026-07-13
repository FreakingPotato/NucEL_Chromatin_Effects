
import os, gzip, urllib.request, numpy as np, random
from pyfaidx import Fasta
random.seed(1234); np.random.seed(1234)
WORK=os.path.expanduser("~/nucel_atac"); os.makedirs(WORK, exist_ok=True)
GEN=os.path.expanduser("~/Claude_Science_Hackathon/genome/hg38.fa")
WIN=1000; HALF=WIN//2
MAIN=set("chr"+str(i) for i in range(1,23))|{"chrX"}
pk=os.path.join(WORK,"peaks.bed.gz")
if not os.path.exists(pk):
    urllib.request.urlretrieve("https://www.encodeproject.org/files/ENCFF748UZH/@@download/ENCFF748UZH.bed.gz", pk)
fa=Fasta(GEN)
sizes={c:len(fa[c]) for c in fa.keys()}
peaks=[]
with gzip.open(pk,'rt') as f:
    for line in f:
        p=line.rstrip("\n").split("\t"); chrom=p[0]
        if chrom not in MAIN: continue
        start=int(p[1]); end=int(p[2]); sig=float(p[6]); off=int(p[9])
        summit=start+off if off>=0 else (start+end)//2
        s=summit-HALF; e=summit+HALF
        if s<0 or e>sizes.get(chrom,0): continue
        peaks.append((chrom,s,e,summit,sig))
print("total peaks (main chr, in-bounds):", len(peaks))
random.shuffle(peaks); peaks=peaks[:3500]
from collections import defaultdict
import bisect
peak_iv=defaultdict(list)
with gzip.open(pk,'rt') as f:
    for line in f:
        p=line.rstrip("\n").split("\t"); chrom=p[0]
        if chrom not in MAIN: continue
        peak_iv[chrom].append((int(p[1])-HALF,int(p[2])+HALF))
for ch in peak_iv: peak_iv[ch].sort()
def overlaps_peak(chrom,s,e):
    ivs=peak_iv.get(chrom,[])
    if not ivs: return False
    starts=[x[0] for x in ivs]
    i=bisect.bisect_right(starts,e)
    for j in range(max(0,i-1),min(len(ivs),i+1)):
        a,b=ivs[j]
        if s<b and a<e: return True
    return False
def gc(seq):
    seq=seq.upper(); n=len(seq)
    return (seq.count("G")+seq.count("C"))/n if n else 0.0
def getseq(chrom,s,e): return str(fa[chrom][s:e])
pos=[]
for chrom,s,e,summit,sig in peaks:
    seq=getseq(chrom,s,e)
    if seq.upper().count("N")>0.05*WIN: continue
    pos.append((chrom,s,e,sig,gc(seq),seq))
print("positives after N-filter:", len(pos))
pos_gc=np.array([x[4] for x in pos]); bins=np.linspace(0,1,21)
pos_hist,_=np.histogram(pos_gc,bins=bins)
chroms=[c for c in MAIN if c in sizes]
neg_by_bin=defaultdict(list); need=pos_hist.copy(); tries=0; maxtries=6000000
while need.sum()>0 and tries<maxtries:
    tries+=1
    chrom=random.choice(chroms); L=sizes[chrom]
    s=random.randint(0,L-WIN); e=s+WIN
    if overlaps_peak(chrom,s,e): continue
    seq=getseq(chrom,s,e)
    if seq.upper().count("N")>0.05*WIN: continue
    g=gc(seq); b=min(np.digitize(g,bins)-1,19); b=max(b,0)
    if need[b]>0: neg_by_bin[b].append((chrom,s,e,g,seq)); need[b]-=1
neg=[x for b in neg_by_bin for x in neg_by_bin[b]]
print("negatives sampled:", len(neg), "tries:", tries, "remaining need:", int(need.sum()))
recs=[]
for i,(chrom,s,e,sig,g,seq) in enumerate(pos): recs.append((f"pos_{i}",1,chrom,seq,g,sig))
for i,(chrom,s,e,g,seq) in enumerate(neg): recs.append((f"neg_{i}",0,chrom,seq,g,0.0))
random.shuffle(recs)
with open(os.path.join(WORK,"dataset.tsv"),"w") as out:
    out.write("id\tlabel\tchrom\tgc\tsignal\tseq\n")
    for r in recs: out.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[4]:.4f}\t{r[5]:.4f}\t{r[3]}\n")
lab=np.array([r[1] for r in recs]); ch=np.array([r[2] for r in recs])
test_mask=np.isin(ch,["chr1","chr2"])
print("TOTAL",len(recs),"pos",int(lab.sum()),"neg",int((lab==0).sum()))
print("TEST(chr1,chr2)",int(test_mask.sum()),"pos",int(lab[test_mask].sum()),"TRAIN",int((~test_mask).sum()),"pos",int(lab[~test_mask].sum()))
print("pos gc mean",round(float(pos_gc.mean()),3),"neg gc mean",round(float(np.array([r[4] for r in recs if r[1]==0]).mean()),3))
print("DONE_BUILD")
