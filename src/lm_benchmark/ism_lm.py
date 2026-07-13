
import os, sys, time, json
os.environ["TOKENIZERS_PARALLELISM"]="false"
import numpy as np, pandas as pd, torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.models.bert.configuration_bert import BertConfig

BASE="/home/stark/Claude_Science_Hackathon/glm_bench"
DEV="cuda"
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)

TARGETS=["rs6701713","rs10939105","rs59325138","rs148163066","rs4698412","rs4575098"]
CENTER=500; HALF=50            # positions 450..549 inclusive -> 100 positions
START=CENTER-HALF; END=CENTER+HALF   # [450,550)
BASES=list("ACGT")

# gather ref_seq for each target
ad=pd.read_csv(f"{BASE}/AD_windows.tsv",sep="\t")
pd_=pd.read_csv(f"{BASE}/PD_windows.tsv",sep="\t")
allw=pd.concat([ad,pd_],ignore_index=True)
refseqs={}
for t in TARGETS:
    row=allw[allw.variant_id==t]
    assert len(row)>=1, f"missing {t}"
    refseqs[t]=row.iloc[0].ref_seq
    assert len(refseqs[t])==1000, f"{t} len {len(refseqs[t])}"
log("loaded ref_seq for",len(refseqs),"variants")

def batched_probs(seqs, tok, maxlen, forward_prob, bs=64):
    probs=[]
    for i in range(0,len(seqs),bs):
        chunk=seqs[i:i+bs]
        enc=tok(chunk,return_tensors="pt",padding=True,truncation=True,max_length=maxlen)
        enc={kk:vv.to(DEV) for kk,vv in enc.items()}
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            p=forward_prob(enc)
        probs.append(p)
    return np.concatenate(probs)

def ism_for_model(name, tok, maxlen, forward_prob, bs=64):
    log("ISM model",name)
    out={}
    for t in TARGETS:
        ref=refseqs[t]
        seqs=[ref]  # index 0 = reference
        meta=[]     # (position_index_in_window, alt_base)
        for pos in range(START,END):
            for b in BASES:
                if b==ref[pos]: continue
                mut=ref[:pos]+b+ref[pos+1:]
                seqs.append(mut); meta.append((pos-START,b))
        probs=batched_probs(seqs,tok,maxlen,forward_prob,bs)
        p_ref=float(probs[0])
        imp=np.zeros(END-START,dtype=np.float64)
        cnt=np.zeros(END-START,dtype=np.int64)
        for j,(wi,b) in enumerate(meta):
            imp[wi]+=(float(probs[1+j])-p_ref)
            cnt[wi]+=1
        # mean over the 3 alts (positions with ref base contribute 3 alts each -> cnt=3)
        imp=imp/np.maximum(cnt,1)
        out[t]=imp.tolist()
        pk=int(np.argmax(np.abs(imp)))
        log(f"  {t} p_ref={p_ref:.4f} nseq={len(seqs)} peak|imp|@win{pk}(genome{pk+START}) val={imp[pk]:+.4f}")
    return out

# ---------- NucEL ----------
log("loading NucEL")
tok_n=AutoTokenizer.from_pretrained("FreakingPotato/NucEL",trust_remote_code=True)
cfg_n=AutoConfig.from_pretrained("/home/stark/NucEL",trust_remote_code=True); cfg_n.reference_compile=False
bb_n=AutoModel.from_pretrained("/home/stark/NucEL",trust_remote_code=True,config=cfg_n)
class NucelClf(nn.Module):
    def __init__(self,bb,hidden=512,ndrop=0.1):
        super().__init__(); self.backbone=bb; self.drop=nn.Dropout(ndrop); self.head=nn.Linear(hidden,2)
    def forward(self,input_ids,attention_mask):
        out=self.backbone(input_ids=input_ids,attention_mask=attention_mask)
        h=out.last_hidden_state; m=attention_mask.unsqueeze(-1).to(h.dtype)
        pooled=(h*m).sum(1)/m.sum(1).clamp(min=1.0)
        return self.head(self.drop(pooled))
m_n=NucelClf(bb_n).to(DEV)
ck=torch.load(f"{BASE}/ckpt_nucel/best.pt",map_location=DEV); m_n.load_state_dict(ck["model"]); m_n.eval()
def fwd_nucel(enc):
    logits=m_n(enc["input_ids"],enc["attention_mask"])
    return torch.softmax(logits.float(),dim=1)[:,1].cpu().numpy()
res_n=ism_for_model("nucel",tok_n,1024,fwd_nucel,bs=64)
json.dump(res_n,open(f"{BASE}/ism_nucel.json","w"))
log("WROTE ism_nucel.json")
del m_n,bb_n; torch.cuda.empty_cache()

# ---------- DNABERT2 ----------
log("loading DNABERT2")
tok_d=AutoTokenizer.from_pretrained(f"{BASE}/ckpt_dnabert2",trust_remote_code=True)
class D2Clf(nn.Module):
    def __init__(self):
        super().__init__()
        _cfg=BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        self.enc=AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M",trust_remote_code=True,config=_cfg)
        h=self.enc.config.hidden_size; self.drop=nn.Dropout(0.1); self.head=nn.Linear(h,1)
    def forward(self,enc):
        out=self.enc(**enc)[0]; mask=enc["attention_mask"].unsqueeze(-1).float()
        pooled=(out*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
        return self.head(self.drop(pooled)).squeeze(-1)
m_d=D2Clf().to(DEV)
sd=torch.load(f"{BASE}/ckpt_dnabert2/model.pt",map_location=DEV); m_d.load_state_dict(sd); m_d.eval()
def fwd_d2(enc):
    logit=m_d(enc)
    return torch.sigmoid(logit.float()).cpu().numpy()
res_d=ism_for_model("dnabert2",tok_d,320,fwd_d2,bs=64)
json.dump(res_d,open(f"{BASE}/ism_dnabert2.json","w"))
log("WROTE ism_dnabert2.json")
del m_d; torch.cuda.empty_cache()
log("ALL_DONE")
