
import sys, os, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.expanduser("~/Claude_Science_Hackathon"))
from gpu_chrombpnet import ChromBPNetTorch, load_weights
from sklearn.metrics import roc_auc_score, average_precision_score
H=os.path.expanduser("~/Claude_Science_Hackathon"); W=os.path.join(H,"glm_bench")
dev="cuda:0"
m=ChromBPNetTorch()
load_weights(m, os.path.join(H,"weights/GM12878_fold0.npz"))
m=m.to(dev).eval()
MAP={'A':0,'C':1,'G':2,'T':3}
def oh(s):
    a=np.zeros((len(s),4),np.float32)
    for i,ch in enumerate(s):
        j=MAP.get(ch,-1)
        if j>=0: a[i,j]=1
    return a
d=pd.read_csv(os.path.join(W,"cbp_2114.tsv"),sep="\t")
te=d[d.split=="test"].reset_index(drop=True)
scores=[]
B=128
with torch.no_grad():
    for i in range(0,len(te),B):
        chunk=te.seq2114.values[i:i+B]
        x=np.stack([oh(s) for s in chunk]).transpose(0,2,1)  # (N,4,2114)
        xt=torch.from_numpy(x).to(dev)
        prof,cnt=m(xt)
        # fwd+revcomp average
        xr=torch.flip(xt,dims=[1,2])
        _,cntr=m(xr)
        c2=0.5*(torch.exp(cnt.float())+torch.exp(cntr.float()))
        scores.extend(c2.squeeze(-1).cpu().numpy().tolist())
te["cbp_score"]=scores
y=te.label.values; s=np.array(scores)
auroc=roc_auc_score(y,s); auprc=average_precision_score(y,s)
te[["id","label","cbp_score"]].to_csv(os.path.join(W,"cbp_test_scores.tsv"),sep="\t",index=False)
print(f"ChromBPNet GM12878 on shared test set (n={len(te)}, pos={int(y.sum())}):")
print(f"  auROC={auroc:.4f}  auPRC={auprc:.4f}")
