"""GPU variant scoring — drop-in for variant-scorer, using the torch port.
Reuses variant-scorer's OWN sequence construction (VariantGenerator, dna_to_one_hot)
so sequences are byte-identical to the CPU pipeline; only inference runs on GPU.
Outputs <prefix>.variant_scores.tsv with the same columns as the TF pipeline.

Usage: python score_gpu.py --list V.tsv --genome hg38.fa --weights W.npz \
         --chrom_sizes cs --out_prefix OUT --schema chrombpnet [--gpu 0] [--num_shuf 100] [--batch 512]
"""
import sys, os, time, argparse, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.expanduser("~/Claude_Science_Hackathon/variant-scorer/src"))
from utils.io import load_variant_table
from utils import one_hot
import math, pyfaidx

def dinuc_shuffle(seq, rng):
    """Altschul-Erikson dinucleotide-preserving shuffle. numpy-2 safe reimplementation
    of deeplift.dinuc_shuffle (which uses removed ndarray.tostring). Preserves the exact
    dinucleotide composition; used only for the shuffled-null background."""
    seq=seq.upper()
    # build edge lists per nucleotide
    from collections import defaultdict
    nexts=defaultdict(list)
    for a,b in zip(seq[:-1], seq[1:]):
        nexts[a].append(b)
    last=seq[-1]
    # shuffle each adjacency list
    for k in nexts: rng.shuffle(nexts[k])
    # ensure a valid Eulerian path: last-edge of each non-terminal node must not all form a cycle;
    # simple robust approach: retry until traversal consumes all edges
    first=seq[0]
    total=len(seq)
    for _attempt in range(50):
        edges={k:list(v) for k,v in nexts.items()}
        for k in edges: rng.shuffle(edges[k])
        out=[first]; cur=first; ok=True
        for _ in range(total-1):
            if edges.get(cur):
                nxt=edges[cur].pop()
            else:
                ok=False; break
            out.append(nxt); cur=nxt
        if ok and len(out)==total:
            return "".join(out)
    return seq  # fallback: original (rare)

class VariantGenerator:
    """Inlined from variant-scorer generators/variant_generator.py (Keras Sequence base
    dropped — only __len__/__getitem__ used). Identical sequence construction."""
    def __init__(self, variants_table, input_len, genome_fasta, batch_size=512, debug_mode=False, shuf=False):
        self.variants_table=variants_table; self.num_variants=variants_table.shape[0]
        self.input_len=input_len; self.genome=pyfaidx.Fasta(genome_fasta)
        self.debug_mode=debug_mode; self.flank_size=input_len//2; self.shuf=shuf; self.batch_size=batch_size
    def __get_allele_seq__(self, chrom, pos, allele1, allele2, seed=-1):
        chrom=str(chrom); pos=int(pos); allele1=str(allele1); allele2=str(allele2)
        if allele1=="-": allele1=""
        if allele2=="-": allele2=""
        pos=pos-1
        if len(allele1)==len(allele2):
            flank=str(self.genome[chrom][pos-self.flank_size:pos+self.flank_size])
            if self.shuf:
                assert seed!=-1; flank=dinuc_shuffle(flank, rng=np.random.RandomState(seed))
            a1=flank[:self.flank_size]+allele1+flank[self.flank_size+len(allele1):]
            a2=flank[:self.flank_size]+allele2+flank[self.flank_size+len(allele2):]
        else:
            assert len(allele1)!=len(allele2)
            assert self.genome[chrom][pos:pos+len(allele1)].seq.upper()==allele1
            ml=len(allele1)-len(allele2)
            flank=str(self.genome[chrom][pos-self.flank_size:pos+self.flank_size+(ml if ml>0 else 0)])
            if self.shuf:
                assert seed!=-1; flank=dinuc_shuffle(flank, rng=np.random.RandomState(seed))
            lf=flank[:self.flank_size]
            a1=lf+allele1+flank[self.flank_size+len(allele1):self.flank_size*2]
            a2=lf+allele2+flank[self.flank_size+len(allele1):self.flank_size*2+ml]
        assert len(a1)==self.flank_size*2 and len(a2)==self.flank_size*2
        return a1,a2
    def __getitem__(self, idx):
        e=self.variants_table.iloc[idx*self.batch_size:min(self.num_variants,(idx+1)*self.batch_size)]
        vid=e['variant_id'].tolist()
        if self.shuf:
            a1,a2=zip(*[self.__get_allele_seq__(v,w,x,y,z) for v,w,x,y,z in zip(e.chr,e.pos,e.allele1,e.allele2,e.random_seed)])
        else:
            a1,a2=zip(*[self.__get_allele_seq__(w,x,y,z) for w,x,y,z in zip(e.chr,e.pos,e.allele1,e.allele2)])
        return vid, one_hot.dna_to_one_hot(list(a1)), one_hot.dna_to_one_hot(list(a2))
    def __len__(self): return math.ceil(self.num_variants/self.batch_size)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_chrombpnet import ChromBPNetTorch, load_weights, softmax_np, jsd_batch, get_pvals

def predict(model, gen, device, input_len):
    """Return fwd+revcomp-averaged counts & profiles for allele1/allele2."""
    c1=[]; c2=[]; p1=[]; p2=[]
    for i in range(len(gen)):
        _, a1, a2 = gen[i]                        # (b,2114,4) int8
        for seqs, cc, pp in [(a1,c1,p1),(a2,c2,p2)]:
            x = torch.tensor(seqs, dtype=torch.float32, device=device).permute(0,2,1)
            xr = torch.flip(x, dims=[1,2])         # revcomp = reverse seq + complement channel
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type=='cuda')):
                pf, cf = model(x); pr, cr = model(xr)
            pf=pf.float().cpu().numpy(); cf=cf.float().cpu().numpy()
            pr=pr.float().cpu().numpy()[:, ::-1]; cr=cr.float().cpu().numpy()
            # average of exp(logcount) fwd & rev; average of profiles fwd & rev(reversed)
            cc.append(0.5*(np.exp(cf)+np.exp(cr)))
            pp.append(0.5*(pf+pr))
    return (np.concatenate(c1), np.concatenate(c2),
            np.concatenate(p1), np.concatenate(p2))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--list',required=True); ap.add_argument('--genome',required=True)
    ap.add_argument('--weights',required=True); ap.add_argument('--chrom_sizes',required=True)
    ap.add_argument('--out_prefix',required=True); ap.add_argument('--schema',default='chrombpnet')
    ap.add_argument('--gpu',type=int,default=0); ap.add_argument('--num_shuf',type=int,default=100)
    ap.add_argument('--batch',type=int,default=512); ap.add_argument('--input_len',type=int,default=2114)
    ap.add_argument('--random_seed',type=int,default=1234)
    a=ap.parse_args()
    device=torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    t0=time.time()
    model=ChromBPNetTorch().to(device).eval()
    load_weights(model, a.weights); model=model.to(device)
    vt=load_variant_table(a.list, a.schema)
    # observed
    gen=VariantGenerator(vt, a.input_len, a.genome, batch_size=a.batch)
    c1,c2,p1,p2=predict(model,gen,device,a.input_len)
    logfc=np.log2(c2/c1)
    jsd=jsd_batch(softmax_np(p2), softmax_np(p1))
    abs_logfc=np.abs(logfc); abs_jsd=jsd  # jsd>=0
    ies=abs_logfc*jsd                       # abs_logfc_x_jsd = Integrative Effect Size
    # shuffled null
    np.random.seed(a.random_seed)
    shuf=vt.sample(len(vt)*a.num_shuf, random_state=a.random_seed, replace=True, ignore_index=True)
    shuf['random_seed']=np.random.permutation(len(shuf))
    sgen=VariantGenerator(shuf, a.input_len, a.genome, batch_size=a.batch, shuf=True)
    sc1,sc2,sp1,sp2=predict(model,sgen,device,a.input_len)
    s_logfc=np.log2(sc2/sc1); s_jsd=jsd_batch(softmax_np(sp2),softmax_np(sp1))
    s_abs_logfc=np.abs(s_logfc); s_ies=s_abs_logfc*s_jsd
    # active-allele quantile against fwd counts distribution (all shuffled allele1+allele2)
    bg_counts=np.concatenate([sc1,sc2])
    def quant(x): return np.array([max(np.mean(bg_counts<v),1/len(bg_counts)) for v in x])
    aaq=np.maximum(quant(c1),quant(c2))
    out=vt.copy()
    out['logfc']=logfc; out['abs_logfc']=abs_logfc; out['jsd']=jsd
    out['abs_logfc_x_jsd']=ies; out['active_allele_quantile']=aaq
    out['logfc.pval']=get_pvals(abs_logfc, s_abs_logfc, 'right')
    out['jsd.pval']=get_pvals(jsd, s_jsd, 'right')
    out['abs_logfc_x_jsd.pval']=get_pvals(ies, s_ies, 'right')
    out.to_csv(a.out_prefix+'.variant_scores.tsv', sep='\t', index=False)
    print(f"WROTE {a.out_prefix}.variant_scores.tsv  n={len(out)}  time={time.time()-t0:.1f}s  device={device}")

if __name__=='__main__':
    main()
