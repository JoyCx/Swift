"""HF-level KL(ref || cand) + top-1 agreement on a held-out text, teacher-forced.

Loads ref (BF16) and cand (a complete compressed-tensors HF model, e.g. the GPTQ output)
SEQUENTIALLY (27B won't co-reside on 32 GB): forward ref -> save logits -> free -> forward
cand -> compare. Run with the swift 3.12 venv python.

  python kld_measure.py <ref_dir> <cand_dir> <heldout.txt> [max_tokens]
"""
from __future__ import annotations
import sys, gc, json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

ref_dir, cand_dir, text_path = sys.argv[1], sys.argv[2], sys.argv[3]
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 512
SCRATCH = Path("C:/Users/ilaya/AppData/Local/Temp/claude/A--ninfer-spark/79ffab74-e953-4647-8f64-a06ac226f6f9/scratchpad")
LOGITS = SCRATCH / "kld_ref_logits.pt"

tok = AutoTokenizer.from_pretrained(ref_dir)
text = Path(text_path).read_text(encoding="utf-8")
ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :MAXTOK]
print(f"held-out tokens: {ids.shape[1]}", flush=True)


def logits_for(model_dir: str) -> torch.Tensor:
    m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="auto")
    m.eval()
    with torch.no_grad():
        out = m.forward(input_ids=ids.to(m.device if hasattr(m, "device") else "cuda"))
        lg = out.logits[0].float().cpu()  # [T, V]
    del m, out
    gc.collect(); torch.cuda.empty_cache()
    return lg


print("== forwarding REF (BF16) ==", flush=True)
ref = logits_for(ref_dir)
torch.save(ref, LOGITS)
print(f"ref logits {tuple(ref.shape)} saved", flush=True)

print("== forwarding CAND ==", flush=True)
cand = logits_for(cand_dir)

# align: predict token t+1 from position t -> compare next-token distributions at each position
T = min(ref.shape[0], cand.shape[0])
rp = F.log_softmax(ref[:T], dim=-1)
cp = F.log_softmax(cand[:T], dim=-1)
# KL(ref || cand) = sum_x p_ref (log p_ref - log p_cand)
kl = (rp.exp() * (rp - cp)).sum(-1)  # [T]
top1 = (ref[:T].argmax(-1) == cand[:T].argmax(-1)).float()
res = {
    "ref": ref_dir, "cand": cand_dir, "tokens": int(T),
    "mean_kl": float(kl.mean()), "median_kl": float(kl.median()),
    "p99_kl": float(kl.quantile(0.99)), "max_kl": float(kl.max()),
    "top1_agreement": float(top1.mean()),
}
print("RESULT " + json.dumps(res), flush=True)
(SCRATCH / "kld_result.json").write_text(json.dumps(res, indent=2))
