"""Write the loop direction as a llama.cpp control vector.

direction.i (i>=1) is added to layer i's output and carried forward by the residual stream, so the file stores
per-layer increments of the mean loop-vs-normal gap: v_i = d_i - d_{i-1}, with d_i = mean(loop) - mean(normal)
at layer i. Applied to all layers at strength s, layer i ends up shifted by s * (d_i - d_0): s = -1 removes the
measured mean gap at every depth.
"""
import json, sys
import numpy as np
sys.path.insert(0, "A:/llama-qwen4exp/gguf-py")
import gguf

D = np.load("runs/swiftdiff/loop_direction/directions.npy").astype(np.float32)
S = json.load(open("runs/swiftdiff/loop_direction/layer_stats.json"))
d = D * np.array([s["diff_norm"] for s in S], dtype=np.float32)[:, None]
w = gguf.GGUFWriter("runs/steer/loop_gap.gguf", "controlvector")
w.add_string("controlvector.model_hint", "qwen35")
w.add_int32("controlvector.layer_count", d.shape[0] - 1)
for i in range(1, d.shape[0]):
    w.add_tensor(f"direction.{i}", (d[i] - d[i - 1]).astype(np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote runs/steer/loop_gap.gguf; per-layer gap norms (first/mid/last):", [round(float(np.linalg.norm(d[i])), 2) for i in (1, 32, 63)])
