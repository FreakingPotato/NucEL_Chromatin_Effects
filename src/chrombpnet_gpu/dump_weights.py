"""Dump ChromBPNet Keras .h5 weights to .npz for the torch port. Runs in .venv (TF2.8)."""
import sys, numpy as np, tensorflow as tf
h5, out = sys.argv[1], sys.argv[2]
m = tf.keras.models.load_model(h5, compile=False)
g = {l.name: l for l in m.layers}
d = {}
def cw(layer, key):
    W = layer.get_weights(); d[key+'_k']=W[0]; d[key+'_b']=W[1]
cw(g['wo_bias_bpnet_1st_conv'], 'iconv')
for i in range(1,9):
    cw(g[f'wo_bias_bpnet_{i}conv'], f'dconv{i}')
cw(g['wo_bias_bpnet_prof_out_precrop'], 'prof')
cw(g['wo_bias_bpnet_logcount_predictions'], 'count')
np.savez(out, **d)
print("dumped", out, {k:v.shape for k,v in d.items()})
