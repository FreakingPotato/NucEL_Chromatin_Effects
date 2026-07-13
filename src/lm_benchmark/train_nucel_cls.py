import os, sys, time, json, math, random
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, average_precision_score

# ---- LR from argv ----
LR = float(sys.argv[1]) if len(sys.argv) > 1 else 2e-5

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

ROOT="/home/stark/Claude_Science_Hackathon/glm_bench"
DATA=f"{ROOT}/unified_dataset.tsv"
SWEEP=f"{ROOT}/lrsweep_cls"; os.makedirs(SWEEP, exist_ok=True)
CKPT=f"{SWEEP}/ckpt_cls_lr{LR:g}"; os.makedirs(CKPT, exist_ok=True)
OUTJSON=f"{SWEEP}/nucel_cls_lr{LR:g}.json"
LOCAL_MODEL="/home/stark/NucEL"
HUB="FreakingPotato/NucEL"
DEV="cuda"
MAXLEN=1024
BATCH=16
EPOCHS=8            # max epochs
PATIENCE=4         # in units of fine evals
GRAD_CLIP=1.0
WARMUP_FRAC=0.10

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

log(f"=== CLS LR SWEEP RUN lr={LR:g} ===")
log("loading data")
df=pd.read_csv(DATA, sep="\t")
tr_full=df[df.split=="train"].reset_index(drop=True)
te=df[df.split=="test"].reset_index(drop=True)
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

log("loading tokenizer (hub) + model (local)")
tok=AutoTokenizer.from_pretrained(HUB, trust_remote_code=True)
cfg=AutoConfig.from_pretrained(LOCAL_MODEL, trust_remote_code=True); cfg.reference_compile=False
backbone=AutoModel.from_pretrained(LOCAL_MODEL, trust_remote_code=True, config=cfg)

class SeqDS(Dataset):
    def __init__(self, frame):
        self.ids=frame.id.tolist(); self.seq=frame.seq.tolist(); self.y=frame.label.tolist()
    def __len__(self): return len(self.seq)
    def __getitem__(self,i): return self.ids[i], self.seq[i], int(self.y[i])

def collate(batch):
    ids=[b[0] for b in batch]; seqs=[b[1] for b in batch]; ys=[b[2] for b in batch]
    enc=tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN)
    return ids, enc, torch.tensor(ys, dtype=torch.long)

class Classifier(nn.Module):
    def __init__(self, backbone, hidden=512, ndrop=0.1):
        super().__init__()
        self.backbone=backbone
        self.drop=nn.Dropout(ndrop)
        self.head=nn.Linear(hidden,2)
    def forward(self, input_ids, attention_mask):
        out=self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h=out.last_hidden_state
        pooled=h[:,0,:]  # CLS token = first token last_hidden_state
        return self.head(self.drop(pooled))

model=Classifier(backbone).to(DEV)
opt=torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
lossf=nn.CrossEntropyLoss()

tr_dl=DataLoader(SeqDS(tr), batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=2)
va_dl=DataLoader(SeqDS(va), batch_size=32, shuffle=False, collate_fn=collate, num_workers=2)
te_dl=DataLoader(SeqDS(te), batch_size=32, shuffle=False, collate_fn=collate, num_workers=2)

steps_per=len(tr_dl)
total_steps=steps_per*EPOCHS
warmup_steps=max(1,int(WARMUP_FRAC*total_steps))
sched=get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
eval_every=math.ceil(steps_per/3)
log(f"steps/epoch {steps_per}, total_steps {total_steps}, warmup {warmup_steps}, eval_every {eval_every}")

def evaluate(dl):
    model.eval(); ids=[]; probs=[]; ys=[]
    with torch.no_grad():
        for bids, enc, y in dl:
            enc={k:v.to(DEV) for k,v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits=model(enc["input_ids"], enc["attention_mask"])
            p=torch.softmax(logits.float(),dim=1)[:,1].cpu().numpy()
            ids+=bids; probs+=p.tolist(); ys+=y.tolist()
    return np.array(ids), np.array(probs), np.array(ys)

curve=[]          # list of dicts: global_step, train_loss_running, val_auROC, val_auPRC, lr
best_auc=-1; best_step=-1; bad=0; gstep=0
stop=False
t0=time.time()
for ep in range(1, EPOCHS+1):
    model.train()
    win_loss=0.0; win_n=0   # running train loss since last eval (window)
    for i,(bids, enc, y) in enumerate(tr_dl):
        enc={k:v.to(DEV) for k,v in enc.items()}; y=y.to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits=model(enc["input_ids"], enc["attention_mask"])
            loss=lossf(logits, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step(); sched.step()
        gstep+=1; win_loss+=loss.item(); win_n+=1
        # fine eval at 1/3, 2/3, 3/3 of each epoch
        if (i+1)%eval_every==0 or (i+1)==steps_per:
            tl=win_loss/max(1,win_n)
            _,vp,vy=evaluate(va_dl)
            vauc=roc_auc_score(vy,vp); vap=average_precision_score(vy,vp)
            cur_lr=sched.get_last_lr()[0]
            curve.append({"global_step":gstep,"epoch":ep,"train_loss_running":float(tl),
                          "val_auROC":float(vauc),"val_auPRC":float(vap),"lr":float(cur_lr)})
            log(f"ep{ep} step {i+1}/{steps_per} gstep {gstep} train_loss(win) {tl:.4f} val_auROC {vauc:.4f} val_auPRC {vap:.4f} lr {cur_lr:.2e}")
            win_loss=0.0; win_n=0
            model.train()
            if vauc>best_auc:
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

ck=torch.load(f"{CKPT}/best.pt", map_location=DEV)
model.load_state_dict(ck["model"])
tids,tp,ty=evaluate(te_dl)
auroc=roc_auc_score(ty,tp); auprc=average_precision_score(ty,tp)
log(f"TEST auROC {auroc:.4f} auPRC {auprc:.4f}")

out={"lr":float(LR),"best_val_auROC":float(best_auc),"best_step":int(best_step),
     "test_auROC":float(auroc),"test_auPRC":float(auprc),
     "steps_per_epoch":int(steps_per),"eval_every":int(eval_every),
     "train_seconds":float(train_seconds),"n_test":int(len(ty)),
     "curve":curve}
json.dump(out, open(OUTJSON,"w"), indent=2)
log("wrote", OUTJSON)
log("RUN_DONE")
