"""Print random loop-entry examples from a label directory for manual precision checks."""
import json, random, sys
import numpy as np
from tokenizers import Tokenizer
d = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
tok = Tokenizer.from_file("A:/models/Qwen3.8-27B-base-bf16/tokenizer.json")
M = [json.loads(l) for l in open(f"{d}/windows.jsonl")]
tot = sum(m["n_waste"] + m["n_normal"] for m in M)
print(f"{len(M)} windows | median len {sorted(m['len'] for m in M)[len(M)//2]} | entries {sum(m['n_entry'] for m in M)} | waste share {sum(m['n_waste'] for m in M)/tot:.3f}")
shards = {}
random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 7)
for i, m in enumerate(random.sample(M, min(n, len(M)))):
    z = shards.setdefault(m["shard"], np.load(f"{d}/shard-{m['shard']:03d}.npz"))
    s, e = z["offsets"][m["index"]], z["offsets"][m["index"] + 1]
    L, I = z["labels"][s:e], z["input_ids"][s:e]
    ent = [k for k in np.flatnonzero(L == 3) if k == 0 or L[k - 1] != 3]
    k = random.choice(ent); w = k
    while w < len(L) and L[w] in (2, 3): w += 1
    print(f"\n[{i}] {m['source']} | iteration {w - k} tok\n...{tok.decode(I[max(0, k - 70):k].tolist())[-280:]}\n>>> {tok.decode(I[k:k + 45].tolist())}")
