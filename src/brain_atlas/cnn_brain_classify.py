import os, numpy as np, pandas as pd, torch, time
import sys; sys.path.insert(0, os.path.expanduser("~/Claude_Science_Hackathon"))
from gpu_chrombpnet import ChromBPNetTorch, load_weights
from sklearn.metrics import roc_auc_score, average_precision_score
H=os.path.expanduser("~/Claude_Science_Hackathon"); BR="/home/stark/brain_atac"; dev="cuda:0"
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)
MAP={'A':0,'C':1,'G':2,'T':3}
def oh(s):
    a=np.zeros((len(s),4),np.float32)
    for i,ch in enumerate(s):
        j=MAP.get(ch,-1)
        if j>=0: a[i,j]=1
    return a
res={}
for cell in ["brain","DLPFC","astrocyte","glut_neuron"]:
    m=ChromBPNetTorch(); load_weights(m, f"{H}/weights/{cell}_fold0.npz"); m=m.to(dev).eval()
    te=pd.read_csv(f"{BR}/test2114_{cell}.tsv",sep="\t")
    scores=[]; B=128
    with torch.no_grad():
        for i in range(0,len(te),B):
            chunk=te.seq2114.values[i:i+B]
            x=np.stack([oh(s) for s in chunk]).transpose(0,2,1)
            xt=torch.from_numpy(x).to(dev)
            prof,cnt=m(xt); xr=torch.flip(xt,dims=[1,2]); _,cntr=m(xr)
            c2=0.5*(torch.exp(cnt.float())+torch.exp(cntr.float()))
            scores.extend(c2.squeeze(-1).cpu().numpy().tolist())
    y=te.label.values; s=np.array(scores)
    auroc=roc_auc_score(y,s); auprc=average_precision_score(y,s)
    res[cell]={"cnn_auROC":float(auroc),"cnn_auPRC":float(auprc),"n":int(len(y))}
    log(f"{cell}: CNN zero-shot auROC {auroc:.4f} auPRC {auprc:.4f} n={len(y)}")
import json; json.dump(res, open(f"{BR}/cnn_brain_classify.json","w"), indent=1)
log("CNN_DONE")
