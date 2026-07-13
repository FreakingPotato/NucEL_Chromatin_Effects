"""PyTorch port of official ChromBPNet (chrombpnet_nobias .h5) for GPU inference.
Mirrors the Keras architecture layer-for-layer:
  Conv1d(4->512,k21,relu) -> 8 dilated residual blocks(512,k3,relu,dil=2^i, crop=dil, add)
  -> profile head Conv1d(512->1,k75,linear)->flatten(1000 logits)
  -> count head GAP(over last residual)->Linear(512->1) logcount
Weights loaded from an .npz dumped from the Keras model (see dump_weights.py).
Scoring math replicates kundajelab/variant-scorer exactly:
  counts = exp(logcount); logfc = log2(c2/c1); jsd = JS(softmax(prof2),softmax(prof1),base2)
  fwd+revcomp averaged; shuffled-null p-values via dinuc-shuffled flanks.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

class ChromBPNetTorch(nn.Module):
    def __init__(self, n_filters=512, n_dil=8, conv1_k=21, prof_k=75):
        super().__init__()
        self.iconv = nn.Conv1d(4, n_filters, conv1_k)           # valid
        self.dconvs = nn.ModuleList([
            nn.Conv1d(n_filters, n_filters, 3, dilation=2**i) for i in range(1, n_dil+1)])
        self.prof = nn.Conv1d(n_filters, 1, prof_k)             # linear
        self.count = nn.Linear(n_filters, 1)
        self.n_dil = n_dil
    def forward(self, x):                                        # x: (N,4,2114)
        x = F.relu(self.iconv(x))
        for i, conv in enumerate(self.dconvs):
            d = 2**(i+1)
            cx = F.relu(conv(x))
            x = cx + x[:, :, d:x.shape[2]-d]                     # crop input by dilation each side, add
        prof = self.prof(x).squeeze(1)                           # (N,1000) logits
        cnt = self.count(x.mean(dim=2)).squeeze(1)               # GAP over length -> Linear -> (N,)
        return prof, cnt

def load_weights(model, npz_path):
    w = np.load(npz_path)
    # Keras Conv1D kernel: (k, in, out) -> torch (out, in, k); Dense: (in,out)->(out,in)
    def conv(dst, name):
        dst.weight.data = torch.tensor(w[name+'_k'].transpose(2,1,0).copy())
        dst.bias.data = torch.tensor(w[name+'_b'].copy())
    conv(model.iconv, 'iconv')
    for i in range(model.n_dil):
        conv(model.dconvs[i], f'dconv{i+1}')
    conv(model.prof, 'prof')
    model.count.weight.data = torch.tensor(w['count_k'].T.copy())
    model.count.bias.data = torch.tensor(w['count_b'].copy())
    return model

def softmax_np(x, temp=1):
    nx = x - np.mean(x, axis=1, keepdims=True)
    return np.exp(temp*nx)/np.sum(np.exp(temp*nx), axis=1, keepdims=True)

def jsd_batch(p, q):  # p,q already softmaxed (N,1000); JS distance base2
    m = 0.5*(p+q)
    def kl(a,b):
        a=np.clip(a,1e-12,1); b=np.clip(b,1e-12,1)
        return np.sum(a*np.log2(a/b), axis=1)
    return np.sqrt(0.5*kl(p,m)+0.5*kl(q,m))

def get_pvals(obs, bg, tail='both'):
    sb = np.sort(bg)
    rr = len(sb) - np.searchsorted(sb, obs, side='left'); pr = (rr+1)/(len(sb)+1)
    rl = np.searchsorted(sb, obs, side='right'); pl = (rl+1)/(len(sb)+1)
    if tail=='right': return pr
    if tail=='left': return pl
    return np.minimum(pl,pr)*2
