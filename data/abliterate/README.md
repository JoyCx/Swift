# Abliteration prompt sets (user-supplied)

`swiftlab abliterate` needs two files:

* `harmful.txt` / `harmful.jsonl` — requests the model currently refuses (one per line, or `{"prompt": ...}`)
* `harmless.txt` / `harmless.jsonl` — ordinary requests of similar length and form

They are not shipped in this repo. The community tooling (heretic, the original abliteration
notebooks) uses public sets such as `mlabonne/harmful_behaviors` and `mlabonne/harmless_alpaca`
on Hugging Face; export ~100–200 of each with:

```python
from datasets import load_dataset
for name, out in [("mlabonne/harmful_behaviors", "harmful.txt"), ("mlabonne/harmless_alpaca", "harmless.txt")]:
    ds = load_dataset(name, split="train")
    open(out, "w").write("\n".join(r["text"] for r in ds.select(range(160))) + "\n")
```

Keep a held-out slice of each (the harness uses the last 30% for AUC) and never put these
prompts in the task bank.
