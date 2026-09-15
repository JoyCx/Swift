"""Knowledge tasks: the answer is grepped against gold strings / regexes.

`verify` spec:
  {"kind": "regex", "patterns": [...], "all": false, "forbid": [...]}
  {"kind": "exact", "gold": [...]}          # normalized exact match of the final answer line
Gold values come from real data the user owns (docs, tables, wiki dumps) so
the harness never depends on an LLM judge.
"""
from __future__ import annotations
import re, unicodedata
from .base import Task, Verdict

FINAL_RE = re.compile(r"(?:final answer|answer)\s*[:：]\s*(.+)", re.I)


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = re.sub(r"[\s\-_]+", " ", s)
    s = re.sub(r"[^\w\s\.]", "", s)
    return s.strip(" .")


def final_line(answer: str) -> str:
    m = list(FINAL_RE.finditer(answer))
    if m:
        return m[-1].group(1).strip()
    lines = [l for l in answer.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def verify_knowledge(task: Task, answer: str) -> Verdict:
    spec = task.verify
    if spec["kind"] == "exact":
        fl = normalize(final_line(answer))
        golds = [normalize(g) for g in spec["gold"]]
        ok = fl in golds or any(g and g == fl for g in golds)
        return Verdict(ok, f"final={fl!r} gold={golds}")
    pats = spec.get("patterns", [])
    hits = [bool(re.search(p, answer, re.I | re.S)) for p in pats]
    ok = all(hits) if spec.get("all") else any(hits)
    for fp in spec.get("forbid", []):
        if re.search(fp, answer, re.I | re.S):
            return Verdict(False, f"forbidden pattern matched: {fp}")
    return Verdict(ok, f"hits={hits}")


def knowledge_tasks_from_table(rows: list[dict], question_tpl: str, answer_field: str, id_prefix: str = "kb", seed: int = 0) -> list[Task]:
    """Build grep-verified tasks from a fact table (list of dict rows).

    question_tpl e.g. "What is the capital of {country}?"  answer_field e.g. "capital".
    """
    out = []
    for i, r in enumerate(rows):
        gold = str(r[answer_field])
        out.append(Task(id=f"{id_prefix}-{seed}-{i}", domain="knowledge", prompt=question_tpl.format(**r) + "\nEnd with a line 'Final answer: <answer>'.",
                        verify={"kind": "regex", "patterns": [re.escape(gold)]}, difficulty=0.3, source="synthetic", meta={"row": i}))
    return out


_FACTS = [
    {"entity": "the Nile", "field": "continent", "value": "Africa"},
    {"entity": "Python's list.sort", "field": "sort stability", "value": "stable"},
    {"entity": "the SHA-256 digest", "field": "length in bits", "value": "256"},
    {"entity": "IPv4 addresses", "field": "size in bits", "value": "32"},
    {"entity": "a leap year divisible by 100 but not 400", "field": "leap status", "value": "not a leap year"},
    {"entity": "the HTTP status code for 'Not Found'", "field": "code", "value": "404"},
    {"entity": "the chemical symbol for gold", "field": "symbol", "value": "Au"},
    {"entity": "the number of bytes in a kibibyte", "field": "bytes", "value": "1024"},
    {"entity": "the default port for HTTPS", "field": "port", "value": "443"},
    {"entity": "the time complexity of binary search", "field": "big-O", "value": r"O\(log ?n\)", "text": "O(log n)"},
    {"entity": "the planet closest to the Sun", "field": "name", "value": "Mercury"},
    {"entity": "the author of 'Pride and Prejudice'", "field": "name", "value": "Jane Austen"},
    {"entity": "the number of bits in a byte", "field": "bits", "value": "8"},
    {"entity": "the largest ocean on Earth", "field": "name", "value": "Pacific"},
    {"entity": "the process by which plants make sugar from light", "field": "name", "value": "photosynthesis"},
    {"entity": "the git command that creates a new commit from staged changes", "field": "command", "value": "git commit"},
    {"entity": "the Unix signal number for SIGKILL", "field": "number", "value": "9"},
    {"entity": "the base of the natural logarithm, to two decimals", "field": "value", "value": "2\\.72", "text": "2.72"},
    {"entity": "the HTTP method that is idempotent and replaces a resource", "field": "method", "value": "PUT"},
    {"entity": "the year the first moon landing happened", "field": "year", "value": "1969"},
]
_PHRASINGS = ["State {field} of {entity}.", "What is {field} of {entity}?", "Give {field} for {entity}.",
              "In one line, tell me {field} of {entity}.", "Quick fact check: {field} of {entity}?"]


def synthetic_knowledge_tasks(n: int, seed: int = 0) -> list[Task]:
    out = []
    for i in range(n):
        f = _FACTS[i % len(_FACTS)]
        q = _PHRASINGS[(i // len(_FACTS)) % len(_PHRASINGS)].format(**f) + " Be concise and end with a line 'Final answer: <answer>'."
        out.append(Task(id=f"syn-know-{seed}-{i}", domain="knowledge", prompt=q,
                        verify={"kind": "regex", "patterns": [f["value"]]}, difficulty=0.2, source="synthetic",
                        meta={"fact": i % len(_FACTS), "gold_text": f.get("text", f["value"])}))
    return out
