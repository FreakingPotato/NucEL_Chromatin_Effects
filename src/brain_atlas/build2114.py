
import os, gzip, time, random, bisect
import numpy as np
from collections import defaultdict
from pyfaidx import Fasta
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)
SEED=1234; random.seed(SEED); np.random.seed(SEED)
GEN="/home/stark/Claude_Science_Hackathon/genome/hg38.fa"
OUT="/home/stark/brain_atac"; WIN=1000; HALF=500; H2=1057; N_POS=15000
TESTCHR={"chr1","chr2"}; MAINCHR=set("chr"+str(i) for i in range(1,23))|{"chrX"}
fa=Fasta(GEN); sizes={c:len(fa[c]) for c in fa.keys()}
def gc(s): s=s.upper(); n=len(s); return (s.count("G")+s.count("C"))/n if n else 0
CELLS=["brain","DLPFC","astrocyte","glut_neuron"]
for cell in CELLS:
    pk=f"{OUT}/{cell}.bed.gz"
    allpk=[]
    with gzip.open(pk,'rt') as f:
        for line in f:
            p=line.rstrip("\n").split("\t"); ch=p[0]
            if ch not in MAINCHR: continue
            st=int(p[1]); en=int(p[2]); sig=float(p[6]); off=int(p[9])
            summit=st+off if off>=0 else (st+en)//2
            s=summit-HALF; e=summit+HALF
            if s<0 or e>sizes.get(ch,0): continue
            allpk.append((ch,s,e,sig))
    train_pk=[p for p in allpk if p[0] not in TESTCHR]
    test_pk =[p for p in allpk if p[0] in TESTCHR]
    random.shuffle(train_pk); random.shuffle(test_pk)
    train_pk=train_pk[:N_POS]
    n_test_pos=max(300, int(N_POS*len(test_pk)/max(1,len(allpk))))
    test_pk=test_pk[:n_test_pos]
    iv=defaultdict(list)
    with gzip.open(pk,'rt') as f:
        for line in f:
            p=line.rstrip("\n").split("\t"); ch=p[0]
            if ch not in MAINCHR: continue
            iv[ch].append((int(p[1])-HALF,int(p[2])+HALF))
    for ch in iv: iv[ch].sort()
    starts={ch:[x[0] for x in iv[ch]] for ch in iv}
    def ovl(ch,s,e):
        st=starts.get(ch)
        if not st: return False
        i=bisect.bisect_right(st,e)
        for j in range(max(0,i-1),min(len(iv[ch]),i+1)):
            a,b=iv[ch][j]
            if s<b and a<e: return True
        return False
    def seq2114(ch,center):
        s=center-H2; e=center+H2
        if s<0 or e>sizes.get(ch,0): return None
        return str(fa[ch][s:e])
    def build_side(peaks, allow_test):
        pos=[]
        for ch,s,e,sig in peaks:
            seq=str(fa[ch][s:e])
            if seq.upper().count("N")>0.05*WIN: continue
            pos.append((ch,s,gc(seq),sig,seq))    # keep start s
        pgc=np.array([x[2] for x in pos]); bins=np.linspace(0,1,21)
        hist,_=np.histogram(pgc,bins=bins); need=hist.copy()
        pool=[c for c in MAINCHR if c in sizes and ((c in TESTCHR)==allow_test)]
        neg=[]; tries=0; last=int(need.sum()); stall=0
        while need.sum()>0 and tries<50000000:
            tries+=1; ch=random.choice(pool); L=sizes[ch]
            s=random.randint(0,L-WIN); e=s+WIN
            if ovl(ch,s,e): continue
            seq=str(fa[ch][s:e])
            if seq.upper().count("N")>0.05*WIN: continue
            b=max(0,min(np.digitize(gc(seq),bins)-1,19))
            if need[b]>0: neg.append((ch,s,gc(seq),seq)); need[b]-=1   # keep start s
            if tries%3000000==0:
                rem=int(need.sum())
                if rem==last and rem<=8: stall+=1
                else: stall=0
                last=rem
                if stall>=1: break
        return pos,neg
    tr_pos,tr_neg=build_side(train_pk, False)
    te_pos,te_neg=build_side(test_pk, True)
    # verify 1kb identity against existing ds file (test rows)
    # emit test2114: id, label, seq2114  (center = start+HALF)
    out2114=[]
    def add2114(rows,lab,tag):
        for i,r in enumerate(rows):
            ch=r[0]; s=r[1]; center=s+HALF
            q=seq2114(ch,center)
            if q is None or q.upper().count("N")>0.05*len(q): continue
            out2114.append((f"{cell}_{tag}{i}",lab,q))
    add2114(te_pos,1,"tep"); add2114(te_neg,0,"ten")
    fn=f"{OUT}/test2114_{cell}.tsv"
    with open(fn,"w") as o:
        o.write("id\tlabel\tseq2114\n")
        for r in out2114: o.write("\t".join(map(str,r))+"\n")
    log(f"{cell}: test2114 rows {len(out2114)} (pos {sum(1 for x in out2114 if x[1]==1)}) -> {fn}")
log("DONE2114")
