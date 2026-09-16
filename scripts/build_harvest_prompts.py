"""Build harvest prompt sets from locally cached datasets that are NOT in the UkisAI eval repo.

musique: MuSiQue-Ans dev, answerable questions, 20 paragraphs of context (supporting + distractors),
verified by exact/alias match on the 'Answer:' line.
"""
import argparse, json, random
from pathlib import Path

MUSIQUE = Path("A:/hf_cache/hub/datasets--dgslibisey--MuSiQue/snapshots/c8f4f8c9465fb69d31a8eae894c3fd509c4ca321/musique_ans_v1.0_dev.jsonl")

def musique(n, rng):
    rows = [json.loads(l) for l in MUSIQUE.open(encoding="utf-8")]
    rows = [r for r in rows if r.get("answerable", True)]
    rng.shuffle(rows)
    for r in rows[:n]:
        ctx = "\n\n".join(f"[{i+1}] {p['title']}\n{p['paragraph_text']}" for i, p in enumerate(r["paragraphs"]))
        prompt = (f"Answer the question using the paragraphs below. Some paragraphs are irrelevant.\n\n{ctx}\n\n"
                  f"Question: {r['question']}\n\nEnd your reply with a final line of the form 'Answer: <short answer>'.")
        hops = r["id"].split("__")[0]
        yield {"id": f"musique/{r['id']}", "domain": "multihop_qa", "subdomain": hops, "prompt": prompt,
               "verify": {"kind": "alias", "answers": [r["answer"], *r.get("answer_aliases", [])]}}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--musique", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/harvest/prompts.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = list(musique(a.musique, rng))
    rng.shuffle(rows)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(len(rows), "prompts ->", a.out)
