import os, sys, time, json, math, random
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, average_precision_score

CELL=sys.argv[1]; LR=1.5e-4
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
ROOT="/home/stark/brain_atac"
DATA=f"{ROOT}/ds_{CELL}.tsv"
CKPT=f"{ROOT}/ckpt_{CELL}"; os.makedirs(CKPT, exist_ok=True)
OUTJSON=f"{ROOT}/nucel_{CELL}.json"
LOCAL_MODEL="/home/stark/NucEL"; HUB="FreakingPotato/NucEL"
DEV="cuda"; MAXLEN=1024; BATCH=16; EPOCHS=8; PATIENCE=4; GRAD_CLIP=1.0; WARMUP_FRAC=0.10
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]",*a,flush=True)
log(f"=== BRAIN FT {CELL} lr={LR:g} ===")
df=pd.read_csv(DATA, sep="\t")
tr_full=df[df.split=="train"].reset_index(drop=True); te=df[df.split=="test"].reset_index(drop=True)
rng=np.random.RandomState(SEED); val_idx=[]
for lab in [0,1]:
    idx=tr_full.index[tr_full.label==lab].to_numpy().copy(); rng.shuffle(idx)
    n=max(1,int(0.1*len(idx))); val_idx.extend(idx[:n].tolist())
val_mask=tr_full.index.isin(val_idx)
va=tr_full[val_mask].reset_index(drop=True); tr=tr_full[~val_mask].reset_index(drop=True)
log(f"train {len(tr)} (pos {tr.label.sum()}), val {len(va)} (pos {va.label.sum()}), test {len(te)} (pos {te.label.sum()})")
tok=AutoTokenizer.from_pretrained(HUB, trust_remote_code=True)
cfg=AutoConfig.from_pretrained(LOCAL_MODEL, trust_remote_code=True); cfg.reference_compile=False
backbone=AutoModel.from_pretrained(LOCAL_MODEL, trust_remote_code=True, config=cfg)
class SeqDS(Dataset):
    def __init__(self, fr): self.ids=fr.id.tolist(); self.seq=fr.seq.tolist(); self.y=fr.label.tolist()
    def __len__(self): return len(self.seq)
    def __getitem__(self,i): return self.ids[i], self.seq[i], int(self.y[i])
def collate(b):
    ids=[x[0] for x in b]; seqs=[x[1] for x in b]; ys=[x[2] for x in b]
    enc=tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN)
    return ids, enc, torch.tensor(ys, dtype=torch.long)
class Classifier(nn.Module):
    def __init__(self, backbone, hidden=512, ndrop=0.1):
        super().__init__(); self.backbone=backbone; self.drop=nn.Dropout(ndrop); self.head=nn.Linear(hidden,2)
    def forward(self, input_ids, attention_mask):
        out=self.backbone(input_ids=input_ids, attention_mask=attention_mask); h=out.last_hidden_state
        m=attention_mask.unsqueeze(-1).to(h.dtype); pooled=(h*m).sum(1)/m.sum(1).clamp(min=1.0)
        return self.head(self.drop(pooled))
model=Classifier(backbone).to(DEV)
opt=torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01); lossf=nn.CrossEntropyLoss()
tr_dl=DataLoader(SeqDS(tr), batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=2)
va_dl=DataLoader(SeqDS(va), batch_size=32, shuffle=False, collate_fn=collate, num_workers=2)
te_dl=DataLoader(SeqDS(te), batch_size=32, shuffle=False, collate_fn=collate, num_workers=2)
steps_per=len(tr_dl); total_steps=steps_per*EPOCHS; warmup_steps=max(1,int(WARMUP_FRAC*total_steps))
sched=get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
eval_every=math.ceil(steps_per/3)
log(f"steps/epoch {steps_per}, total {total_steps}, eval_every {eval_every}")
def evaluate(dl):
    model.eval(); probs=[]; ys=[]
    with torch.no_grad():
        for bids, enc, y in dl:
            enc={k:v.to(DEV) for k,v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits=model(enc["input_ids"], enc["attention_mask"])
            probs+=torch.softmax(logits.float(),dim=1)[:,1].cpu().numpy().tolist(); ys+=y.tolist()
    return np.array(probs), np.array(ys)
best_auc=-1; best_step=-1; bad=0; gstep=0; stop=False; t0=time.time()
for ep in range(1, EPOCHS+1):
    model.train(); win_loss=0.0; win_n=0
    for i,(bids, enc, y) in enumerate(tr_dl):
        enc={k:v.to(DEV) for k,v in enc.items()}; y=y.to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits=model(enc["input_ids"], enc["attention_mask"]); loss=lossf(logits, y)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step(); sched.step(); gstep+=1; win_loss+=loss.item(); win_n+=1
        if (i+1)%eval_every==0 or (i+1)==steps_per:
            tl=win_loss/max(1,win_n); vp,vy=evaluate(va_dl)
            vauc=roc_auc_score(vy,vp); vap=average_precision_score(vy,vp)
            log(f"ep{ep} {i+1}/{steps_per} g{gstep} loss {tl:.4f} val_auROC {vauc:.4f}")
            win_loss=0.0; win_n=0; model.train()
            if vauc>best_auc:
                best_auc=vauc; best_step=gstep; bad=0
                torch.save({"model":model.state_dict(),"gstep":gstep,"val_auroc":vauc}, f"{CKPT}/best.pt")
            else:
                bad+=1
                if bad>=PATIENCE: log("early stop"); stop=True; break
    if stop: break
train_seconds=time.time()-t0
log(f"training done {train_seconds:.1f}s best_step {best_step} val {best_auc:.4f}")
ck=torch.load(f"{CKPT}/best.pt", map_location=DEV); model.load_state_dict(ck["model"])
tp,ty=evaluate(te_dl); auroc=roc_auc_score(ty,tp); auprc=average_precision_score(ty,tp)
# bootstrap CI
rs=np.random.RandomState(0); boots=[]
for _ in range(1000):
    ix=rs.randint(0,len(ty),len(ty))
    if len(set(ty[ix].tolist()))<2: continue
    boots.append(roc_auc_score(ty[ix],tp[ix]))
ci=[float(np.percentile(boots,2.5)), float(np.percentile(boots,97.5))]
log(f"TEST auROC {auroc:.4f} auPRC {auprc:.4f} CI {ci}")
out={"cell":CELL,"lr":float(LR),"best_val_auROC":float(best_auc),"best_step":int(best_step),
     "test_auROC":float(auroc),"test_auPRC":float(auprc),"ci_auROC":ci,
     "n_train":int(len(tr)),"n_test":int(len(ty)),"train_seconds":float(train_seconds)}
json.dump(out, open(OUTJSON,"w"), indent=2); log("wrote",OUTJSON); log("RUN_DONE")
