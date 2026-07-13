import os, sys, time, json, math, random
os.environ["TOKENIZERS_PARALLELISM"]="false"
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, average_precision_score

# ---- LR from argv ----
LR = float(sys.argv[1]) if len(sys.argv) > 1 else 2e-5

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

ROOT="/home/stark/Claude_Science_Hackathon/glm_bench"
DATA=f"{ROOT}/unified_dataset.tsv"
SWEEP=f"{ROOT}/lrsweep"; os.makedirs(SWEEP, exist_ok=True)
CKPT=f"{SWEEP}/ckpt_dnabert2_lr{LR:g}"; os.makedirs(CKPT, exist_ok=True)
OUTJSON=f"{SWEEP}/dnabert2_lr{LR:g}.json"
HUB="zhihan1996/DNABERT-2-117M"
DEV="cuda"
MAXLEN=320
BATCH=24
EPOCHS=8            # max epochs
PATIENCE=4         # in units of fine evals
GRAD_CLIP=1.0
WARMUP_FRAC=0.10

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

log(f"=== DNABERT2 LR SWEEP RUN lr={LR:g} ===")
log("loading data")
df=pd.read_csv(DATA, sep="\t")
tr_full=df[df.split=="train"].reset_index(drop=True)
te=df[df.split=="test"].reset_index(drop=True)
# stratified 10% val slice from train (seed 42) -- identical scheme to train_nucel_lr.py
rng=np.random.RandomState(SEED)
val_idx=[]
for lab in [0,1]:
    idx=tr_full.index[tr_full.label==lab].to_numpy().copy()
    rng.shuffle(idx)
    n=max(1,int(0.1*len(idx)))
    val_idx.extend(idx[:n].tolist())
val_mask=tr_full.index.isin(val_idx)
va=tr_full[val_mask].reset_index(drop=True)
tr=tr_full[~val_mask].reset_index(drop=True)
log(f"train {len(tr)} (pos {tr.label.sum()}), val {len(va)} (pos {va.label.sum()}), test {len(te)} (pos {te.label.sum()})")

tok=AutoTokenizer.from_pretrained(HUB, trust_remote_code=True)

class DS(Dataset):
    def __init__(self,d): self.seq=d.seq.tolist(); self.y=d.label.tolist()
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.seq[i], self.y[i]
def collate(b):
    seqs=[x[0] for x in b]; ys=torch.tensor([x[1] for x in b],dtype=torch.float)
    enc=tok(seqs,return_tensors="pt",padding=True,truncation=True,max_length=MAXLEN)
    return enc, ys

class Clf(nn.Module):
    def __init__(self):
        super().__init__()
        _cfg=BertConfig.from_pretrained(HUB)
        self.enc=AutoModel.from_pretrained(HUB,trust_remote_code=True,config=_cfg)
        h=self.enc.config.hidden_size
        self.drop=nn.Dropout(0.1); self.head=nn.Linear(h,1)
    def forward(self,enc):
        out=self.enc(**enc)[0]          # last_hidden_state [B,T,H]
        mask=enc["attention_mask"].unsqueeze(-1).float()
        pooled=(out*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
        return self.head(self.drop(pooled)).squeeze(-1)

model=Clf().to(DEV)
log("model loaded, hidden",model.enc.config.hidden_size)

opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01)
lossf=nn.BCEWithLogitsLoss()

tr_dl=DataLoader(DS(tr),batch_size=BATCH,shuffle=True,collate_fn=collate,num_workers=4)
va_dl=DataLoader(DS(va),batch_size=64,shuffle=False,collate_fn=collate,num_workers=2)
te_dl=DataLoader(DS(te),batch_size=64,shuffle=False,collate_fn=collate,num_workers=2)

steps_per=len(tr_dl)
total_steps=steps_per*EPOCHS
warmup_steps=max(1,int(WARMUP_FRAC*total_steps))
sched=get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
eval_every=math.ceil(steps_per/3)
log(f"steps/epoch {steps_per}, total_steps {total_steps}, warmup {warmup_steps}, eval_every {eval_every}")

@torch.no_grad()
def evaluate(dl):
    model.eval(); probs=[]; ys=[]
    for enc,y in dl:
        enc={k:v.to(DEV) for k,v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit=model(enc)
        probs+=torch.sigmoid(logit.float()).cpu().tolist(); ys+=y.tolist()
    return np.array(ys),np.array(probs)

curve=[]
best_auc=-1; best_step=-1; bad=0; gstep=0; stop=False
t0=time.time()
for ep in range(1, EPOCHS+1):
    model.train()
    win_loss=0.0; win_n=0
    for i,(enc,y) in enumerate(tr_dl):
        enc={k:v.to(DEV) for k,v in enc.items()}; y=y.to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit=model(enc); loss=lossf(logit,y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step(); sched.step()
        gstep+=1; win_loss+=loss.item(); win_n+=1
        if (i+1)%eval_every==0 or (i+1)==steps_per:
            tl=win_loss/max(1,win_n)
            vy,vp=evaluate(va_dl)
            # guard against NaN probs (divergence)
            if np.any(np.isnan(vp)):
                vauc=float("nan"); vap=float("nan")
            else:
                vauc=roc_auc_score(vy,vp); vap=average_precision_score(vy,vp)
            cur_lr=sched.get_last_lr()[0]
            curve.append({"global_step":gstep,"epoch":ep,"train_loss_running":float(tl),
                          "val_auROC":float(vauc),"val_auPRC":float(vap),"lr":float(cur_lr)})
            log(f"ep{ep} step {i+1}/{steps_per} gstep {gstep} train_loss(win) {tl:.4f} val_auROC {vauc:.4f} val_auPRC {vap:.4f} lr {cur_lr:.2e}")
            win_loss=0.0; win_n=0
            model.train()
            improved = (not math.isnan(vauc)) and (vauc>best_auc)
            if improved:
                best_auc=vauc; best_step=gstep; bad=0
                torch.save({"model":model.state_dict(),"gstep":gstep,"val_auroc":vauc}, f"{CKPT}/best.pt")
                log(f"  saved best (val_auROC {vauc:.4f} @ gstep {gstep})")
            else:
                bad+=1; log(f"  no improve ({bad}/{PATIENCE})")
                if bad>=PATIENCE:
                    log("early stop"); stop=True; break
    if stop: break
train_seconds=time.time()-t0
log(f"training done in {train_seconds:.1f}s, best_step {best_step} val_auROC {best_auc:.4f}")

# load best and eval test; if never improved (total divergence), eval current model
if os.path.exists(f"{CKPT}/best.pt"):
    ck=torch.load(f"{CKPT}/best.pt", map_location=DEV)
    model.load_state_dict(ck["model"])
    ty,tp=evaluate(te_dl)
    if np.any(np.isnan(tp)):
        auroc=float("nan"); auprc=float("nan")
    else:
        auroc=roc_auc_score(ty,tp); auprc=average_precision_score(ty,tp)
else:
    log("no best checkpoint saved (diverged); reporting NaN test metrics")
    auroc=float("nan"); auprc=float("nan")
log(f"TEST auROC {auroc:.4f} auPRC {auprc:.4f}")

out={"lr":float(LR),"best_val_auROC":float(best_auc),"best_step":int(best_step),
     "test_auROC":float(auroc),"test_auPRC":float(auprc),
     "steps_per_epoch":int(steps_per),"eval_every":int(eval_every),
     "train_seconds":float(train_seconds),"n_test":int(len(te)),
     "curve":curve}
json.dump(out, open(OUTJSON,"w"), indent=2)
log("wrote", OUTJSON)
log("RUN_DONE")
