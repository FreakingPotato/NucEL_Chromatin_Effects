import os, sys, json, time
import numpy as np, pandas as pd, torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
CELLS=["brain","DLPFC","astrocyte","glut_neuron"]
GB="/home/stark/Claude_Science_Hackathon/glm_bench"; BR="/home/stark/brain_atac"
LOCAL_MODEL="/home/stark/NucEL"; HUB="FreakingPotato/NucEL"; DEV="cuda"; MAXLEN=1024
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)
COMP={'A':'T','T':'A','C':'G','G':'C','N':'N'}
def rc(s): return "".join(COMP.get(c,'N') for c in reversed(s))
tok=AutoTokenizer.from_pretrained(HUB, trust_remote_code=True)
cfg=AutoConfig.from_pretrained(LOCAL_MODEL, trust_remote_code=True); cfg.reference_compile=False
class Classifier(nn.Module):
    def __init__(self, backbone, hidden=512, ndrop=0.1):
        super().__init__(); self.backbone=backbone; self.drop=nn.Dropout(ndrop); self.head=nn.Linear(hidden,2)
    def forward(self, input_ids, attention_mask):
        out=self.backbone(input_ids=input_ids, attention_mask=attention_mask); h=out.last_hidden_state
        m=attention_mask.unsqueeze(-1).to(h.dtype); pooled=(h*m).sum(1)/m.sum(1).clamp(min=1.0)
        return self.head(self.drop(pooled))
def pprob(model, seqs, B=64):
    out=[]
    with torch.no_grad():
        for i in range(0,len(seqs),B):
            enc=tok(seqs[i:i+B], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN)
            enc={k:v.to(DEV) for k,v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits=model(enc["input_ids"], enc["attention_mask"])
            out+=torch.softmax(logits.float(),dim=1)[:,1].cpu().numpy().tolist()
    return np.array(out)
wins={}
for dis in ["AD","PD"]:
    wins[dis]=pd.read_csv(f"{GB}/{dis}_windows.tsv", sep="\t")
for cell in CELLS:
    backbone=AutoModel.from_pretrained(LOCAL_MODEL, trust_remote_code=True, config=cfg)
    model=Classifier(backbone).to(DEV).eval()
    ck=torch.load(f"{BR}/ckpt_{cell}/best.pt", map_location=DEV); model.load_state_dict(ck["model"])
    for dis in ["AD","PD"]:
        w=wins[dis]
        ref=w.ref_seq.tolist(); alt=w.alt_seq.tolist()
        pr=0.5*(pprob(model,ref)+pprob(model,[rc(s) for s in ref]))
        pa=0.5*(pprob(model,alt)+pprob(model,[rc(s) for s in alt]))
        eff=pa-pr
        o=pd.DataFrame({"variant_id":w.variant_id,"chrom":w.chrom,"pos":w.pos,
                        "p_ref":pr,"p_alt":pa,"nucel_effect":eff,"abs_effect":np.abs(eff)})
        fn=f"{BR}/atlas_{cell}_{dis}.tsv"; o.to_csv(fn,sep="\t",index=False)
        log(f"{cell} {dis}: n={len(o)} eff_std={eff.std():.4f} -> {fn}")
    del model, backbone; torch.cuda.empty_cache()
log("SCORING_DONE")
