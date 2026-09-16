"""Second-round overthinking marker mining (post-decision waste).

Stages:
  load     -> read paired base/swift traces (MMLU-Pro subsample, GPQA, C-Eval), score, detect
              decision points, tokenize, cache to runs/markers/residual/cache_*.npz
  detect   -> print random detector examples (validation)
  analyze  -> POST-waste fractions, Monroe log-odds POST vs PRE, flip-flop contrast,
              sanity rates, write candidates.json + REPORT.md

Usage: python mine_residual_markers.py [load|detect|analyze|all] [--mmlu-seeds 0,1]
"""
import json, os, re, sys, glob, random, pickle, argparse, math
from collections import defaultdict
import numpy as np

ROOT = "A:/swift"
DATA = f"{ROOT}/data/external/ukisai-evals"
OUT = f"{ROOT}/runs/markers/residual"
TOK = "A:/models/Qwen3.8-27B/tokenizer.json"
POOL = f"{ROOT}/runs/markers/ukisai_pool_recovered.json"
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------------------
# correctness
MMLU_RE = re.compile(r"answer is \(?([A-J])\)?")
CEVAL_RE1 = re.compile(r"答案[：:]\s*([A-D])")
CEVAL_RE2 = re.compile(r"[Aa]nswer[：:]?\s*\(?([A-D])")


def last(rx, s):
    m = rx.findall(s or "")
    return m[-1] if m else None


# --------------------------------------------------------------------------------------
# decision detector.  Every pattern captures the committed letter in group "L".
LET = r"(?<![A-Za-z0-9])\(?(?P<L>[A-J])\)?(?![A-Za-z0-9'’])"
# letter followed by something that ends the claim (avoid "So I think", "So A study")
END = r"(?=\s*(?:[.,;!)\]\n]|$|\s(?:is|as|because|since|seems|looks|matches|fits|option|choice|--|—|-)\b|\s*->|\s*→))"
PATS = [
    # answer (is|would be|=|:) X     / final answer X / expected answer X
    r"\b[Aa]nswer\s*(?:is|would be|should be|must be|will be|seems to be|appears to be|likely|=|:|->|→)?\s*(?:likely|probably|clearly|indeed|then|therefore)?\s*(?:option|choice)?\s*" + LET,
    r"\bANSWER\s*[:=]?\s*" + LET,
    # option X is/seems/looks correct|right|best|answer
    r"\b(?:[Oo]ption|[Cc]hoice)\s*" + LET + r"\s*(?:is|seems|looks|appears|would be|should be)\s*(?:the\s+)?(?:most\s+)?(?:correct|right|best|answer|intended|true|valid|accurate|consistent)",
    # X is correct / the answer
    LET + r"\s+is\s+(?:the\s+)?(?:most\s+)?(?:correct|right|best|answer|intended|accurate)\b",
    # so / thus / therefore / hence (answer|option)? X
    r"\b(?:[Ss]o|[Tt]hus|[Tt]herefore|[Hh]ence|[Cc]onsequently)\s*,?\s*(?:the\s+)?(?:final\s+)?(?:answer\s*(?:is\s*)?|option\s*|choice\s*|pick\s*|choose\s*)?" + LET + END,
    # choose / pick / select X
    r"\b(?:[Cc]hoose|[Pp]ick|[Ss]elect|[Gg]o with|[Gg]oing with)\s*(?:option\s*|choice\s*)?" + LET + END,
    # arrows
    r"(?:->|=>|→)\s*(?:option\s*|answer\s*)?" + LET + END,
    # final X / final: X
    r"\b[Ff]inal(?:ly)?\s*[:=]?\s*(?:option\s*|choice\s*)?" + LET + END,
    # standalone "Option X." / "matches option X" / "corresponds to X"
    r"(?:^|(?<=[.;:]\s)|(?<=\n))(?:[Oo]ption|[Cc]hoice)\s+" + r"(?<![A-Za-z0-9])\(?(?P<L>[A-J])\)?" + r"(?=\s*[.!]\s|\s*[.!]?$)",
    r"\b(?:matches|match|corresponds to|consistent with|equals|gives|giving|yields)\s+(?:option|choice)\s*" + LET,
    # Chinese
    r"答案\s*(?:是|为|应为|应该是|选|:|：)?\s*(?:选项)?\s*(?P<L>[A-D])(?![A-Za-z])",
    r"(?:故|所以|因此|应|应该|即|则|我)选(?!项)\s*(?P<L>[A-D])(?![A-Za-z])",
    r"(?<![A-Za-z])(?P<L>[A-D])\s*(?:项)?\s*(?:正确|是正确的|对|符合)",
]
# compile into one alternation with renamed groups
_parts = []
for i, p in enumerate(PATS):
    _parts.append("(?:" + p.replace("?P<L>", f"?P<L{i}>") + ")")
COMMIT_RE = re.compile("|".join(_parts), re.M)

HEDGE_BEFORE = re.compile(r"\b(?:not|n't|maybe|perhaps|possibly|might|could|whether|if|or|unless|check|verify|is it|consider|considering|between|vs|versus|instead of|rather than|than)\b|不是|不|或|是否|如果", re.I)
SENT_SPLIT = re.compile(r"[.!?。！？\n]")
PRE_ANALYSIS = re.compile(r"[\s.:,]{0,4}(?:[Bb]ut\s+)?(?:[Ll]et'?s|[Nn]eed|[Ww]e need to|[Ww]e'll)\s+(?:analy[sz]e|solve|think|reason|work|parse|compute|figure|determine|evaluate each|go through)")


def find_commitments(text, letters="ABCDEFGHIJ"):
    """Return list of (start, end, letter) for non-tentative commitments, in order."""
    out = []
    if not text:
        return out
    for m in COMMIT_RE.finditer(text):
        L = None
        lstart = m.start()
        for k, v in m.groupdict().items():
            if v is not None:
                L = v
                lstart = m.start(k)
                break
        if L is None or L not in letters:
            continue
        # clause-level tentativeness check
        s0 = max(0, m.start() - 60)
        before = text[s0:m.start()]
        cut = max(before.rfind(c) for c in ".!?。！？\n,;，；")
        clause_before = before[cut + 1:] if cut >= 0 else before
        after = text[m.end():m.end() + 40]
        nxt = SENT_SPLIT.search(after)
        tail = after[:nxt.start() + 1] if nxt else after
        if "?" in tail[:25] or "？" in tail[:25]:
            continue
        if HEDGE_BEFORE.search(clause_before):
            continue
        # premature guess immediately followed by starting the actual analysis
        if PRE_ANALYSIS.match(text, m.end()):
            continue
        # "answer X" inside format instructions:  'Answer: X' literal placeholder already excluded by letter set
        out.append((lstart, m.end(), L))
    return out


# --------------------------------------------------------------------------------------
def load_rows(mmlu_seeds, ceval=True):
    rows = []  # dicts: src, arm, key, cat, gold, pred, correct, trunc, reasoning
    for arm in ("base", "swift"):
        for s in mmlu_seeds:
            d = f"{DATA}/02-mmlu-pro/{arm}/seed{s}"
            for fn in os.listdir(d):
                if not fn.endswith(".json"):
                    continue
                r = json.load(open(f"{d}/{fn}", encoding="utf-8"))
                pred = last(MMLU_RE, r.get("content") or "")
                rows.append(dict(src="mmlu", arm=arm, key=(s, r["question_id"]), cat=r.get("category", "?"),
                                 gold=r["answer"], pred=pred, correct=pred is not None and pred == r["answer"],
                                 trunc=r.get("finish_reason") != "stop", reasoning=r.get("reasoning") or ""))
        print("loaded mmlu", arm, len(rows), flush=True)
    for arm, sub in (("base", "base_reference"), ("swift", "swift")):
        for s in range(5):
            for line in open(f"{DATA}/01-gpqa-diamond/adapter_only_dp4x2/{sub}/seed{s}.raw.jsonl", encoding="utf-8"):
                r = json.loads(line)
                rows.append(dict(src="gpqa", arm=arm, key=(s, r["task_id"]), cat=r.get("domain", "?"),
                                 gold=r["gold"], pred=r.get("pred"), correct=bool(r.get("correct")),
                                 trunc=(r.get("finish_reason") != "stop") or bool(r.get("truncated")),
                                 reasoning=r.get("reasoning") or ""))
    print("loaded gpqa", len(rows), flush=True)
    if ceval:
        for arm in ("base", "swift"):
            for d in sorted(glob.glob(f"{DATA}/03-c-eval/{arm}/seed*")):
                s = int(d.rsplit("seed", 1)[1])
                for fn in os.listdir(d):
                    if not fn.endswith(".json"):
                        continue
                    r = json.load(open(f"{d}/{fn}", encoding="utf-8"))
                    c = r.get("content") or ""
                    pred = last(CEVAL_RE1, c) or last(CEVAL_RE2, c)
                    rows.append(dict(src="ceval", arm=arm, key=(s, r["question_id"]), cat=r.get("subject", "?"),
                                     gold=r["answer"], pred=pred, correct=pred is not None and pred == r["answer"],
                                     trunc=r.get("finish_reason") != "stop", reasoning=r.get("reasoning") or ""))
        print("loaded ceval", len(rows), flush=True)
    return rows


def stage_load(args):
    from tokenizers import Tokenizer
    seeds = [int(x) for x in args.mmlu_seeds.split(",")]
    rows = load_rows(seeds)
    tok = Tokenizer.from_file(TOK)
    ids_all, meta = [], []
    B = 1000
    off = 0
    for b in range(0, len(rows), B):
        chunk = rows[b:b + B]
        encs = tok.encode_batch([r["reasoning"] for r in chunk], add_special_tokens=False)
        for r, e in zip(chunk, encs):
            ids = np.asarray(e.ids, dtype=np.uint32)
            n = len(ids)
            letters = "ABCD" if r["src"] == "ceval" else "ABCDEFGHIJ"
            com = find_commitments(r["reasoning"], letters)
            if n:
                ends = np.fromiter((o[1] for o in e.offsets), dtype=np.int64, count=n)
            else:
                ends = np.zeros(0, dtype=np.int64)

            def c2t(c):  # number of tokens whose end <= c  -> first POST token index
                return int(np.searchsorted(ends, c, side="right"))
            dec_c = next((c for c in com if c[2] == r["pred"]), None)
            last_c = next((c for c in reversed(com) if c[2] == r["pred"]), None)
            # switch points: consecutive commitments with different letters
            sw = []
            for p, q in zip(com, com[1:]):
                if p[2] != q[2]:
                    sw.append(c2t(q[0]))
            meta.append(dict(src=r["src"], arm=r["arm"], key=r["key"], cat=r["cat"], gold=r["gold"], pred=r["pred"],
                             correct=r["correct"], trunc=r["trunc"], off=off, n=n,
                             dec_tok=c2t(dec_c[1]) if dec_c else -1, dec_char=dec_c[1] if dec_c else -1,
                             last_tok=c2t(last_c[1]) if last_c else -1,
                             n_commit=len(com), letters="".join(c[2] for c in com), switches=sw, ctoks=[c2t(c[0]) for c in com],
                             nchar=len(r["reasoning"])))
            ids_all.append(ids)
            off += n
        print(f"tokenized {min(b + B, len(rows))}/{len(rows)} tokens={off}", flush=True)
    np.save(f"{OUT}/cache_ids.npy", np.concatenate(ids_all))
    pickle.dump(meta, open(f"{OUT}/cache_meta.pkl", "wb"))
    # keep reasoning text for examples (compressed)
    import zstandard
    blob = zstandard.ZstdCompressor(level=3).compress(pickle.dumps([r["reasoning"] for r in rows]))
    open(f"{OUT}/cache_text.pkl.zst", "wb").write(blob)
    print("saved", off, "tokens", len(meta), "traces")


def load_cache():
    import zstandard
    ids = np.load(f"{OUT}/cache_ids.npy")
    meta = pickle.load(open(f"{OUT}/cache_meta.pkl", "rb"))
    texts = pickle.loads(zstandard.ZstdDecompressor().decompress(open(f"{OUT}/cache_text.pkl.zst", "rb").read(),
                                                                   max_output_size=1 << 34))
    return ids, meta, texts


def stage_detect(args):
    """Detector validation printout (works directly on raw data; no cache needed)."""
    random.seed(args.seed)
    rows = load_rows([int(args.mmlu_seeds.split(",")[0])], ceval=True)
    good = [r for r in rows if r["correct"] and not r["trunc"]]
    by = defaultdict(list)
    for r in good:
        by[(r["src"], r["arm"])].append(r)
    for k, rs in sorted(by.items()):
        cov = 0
        for r in rs:
            letters = "ABCD" if r["src"] == "ceval" else "ABCDEFGHIJ"
            if any(c[2] == r["pred"] for c in find_commitments(r["reasoning"], letters)):
                cov += 1
        print(f"coverage {k}: {cov}/{len(rs)} = {cov / max(1, len(rs)):.3f}")
    sample = random.sample(good, args.n)
    for r in sample:
        letters = "ABCD" if r["src"] == "ceval" else "ABCDEFGHIJ"
        com = find_commitments(r["reasoning"], letters)
        d = next((c for c in com if c[2] == r["pred"]), None)
        t = r["reasoning"]
        print("=" * 100)
        print(r["src"], r["arm"], r["key"], "pred", r["pred"], "nchar", len(t), "commits", "".join(c[2] for c in com))
        if d is None:
            print("  NO DECISION FOUND; tail:", repr(t[-300:]))
        else:
            print(f"  pos {d[1]}/{len(t)} ({d[1] / len(t):.2f})")
            print("  ..." + t[max(0, d[0] - 150):d[0]].replace("\n", "⏎") + "【" + t[d[0]:d[1]] + "】" + t[d[1]:d[1] + 80].replace("\n", "⏎"))


# --------------------------------------------------------------------------------------
def md(t):
    return "`" + repr(t).replace("|", "\\|").replace("`", "'") + "`"


def monroe(y1, n1, y2, n2, prior, a0):
    """log-odds ratio with informative Dirichlet prior; returns delta, z (arrays)."""
    a = prior * a0
    y1 = y1.astype(np.float64); y2 = y2.astype(np.float64)
    d = np.log((y1 + a) / (n1 + a0 - y1 - a)) - np.log((y2 + a) / (n2 + a0 - y2 - a))
    v = 1.0 / (y1 + a) + 1.0 / (y2 + a)
    return d, d / np.sqrt(v)


def stage_analyze(args):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK)
    ids, meta, texts = load_cache()
    V = int(max(tok.get_vocab_size(), ids.max() + 1))
    pool = json.load(open(POOL, encoding="utf-8"))
    pool_ids = {p["id"] for p in pool}
    N = len(meta)
    src = np.array([m["src"] for m in meta]); arm = np.array([m["arm"] for m in meta])
    cat = np.array([m["cat"] for m in meta])
    off = np.array([m["off"] for m in meta]); n = np.array([m["n"] for m in meta])
    dec = np.array([m["dec_tok"] for m in meta]); lastt = np.array([m["last_tok"] for m in meta])
    corr = np.array([m["correct"] for m in meta]); trunc = np.array([m["trunc"] for m in meta])
    usable = corr & ~trunc & (n > 0)
    det = usable & (dec >= 0)
    rep = []  # report lines

    def P(s=""):
        print(s); rep.append(s)

    # ---------------- per-token trace index + region labels
    tr_of = np.repeat(np.arange(N, dtype=np.int32), n)
    pos = (np.arange(len(ids), dtype=np.int64) - np.repeat(off, n)).astype(np.int32)
    region = np.full(len(ids), 0, np.int8)  # 0 = not in analysis, 1 = PRE, 2 = POST
    in_det = det[tr_of]
    region[in_det & (pos < dec[tr_of])] = 1
    region[in_det & (pos >= dec[tr_of])] = 2
    # late-PRE control: the <=512 PRE tokens right before the decision, skipping the question restatement
    late_start = np.maximum(dec - 512, np.minimum(300, dec // 2))
    lateR = (region == 1) & (pos >= late_start[tr_of])

    # ---------------- 1/2: coverage + waste fractions
    P("# Residual overthinking markers: post-decision waste mining\n")
    P("## 1. Decision-point detector coverage\n")
    P("| source | arm | traces | correct&non-trunc | decision found | coverage | mean #commitments |")
    P("|---|---|---|---|---|---|---|")
    for s in ("mmlu", "gpqa", "ceval"):
        for a in ("base", "swift"):
            m = (src == s) & (arm == a)
            u = m & usable
            nc = np.array([meta[i]["n_commit"] for i in np.where(u)[0]]) if u.any() else np.zeros(1)
            P(f"| {s} | {a} | {m.sum()} | {u.sum()} | {(m & det).sum()} | {(m & det).sum() / max(1, u.sum()):.3f} | {nc.mean():.2f} |")

    P("\n## 2. POST-decision token fraction (correct, non-truncated, decision found)\n")
    P("POST = reasoning tokens after the first commitment to the final (correct) letter. "
      "`last` = tokens after the LAST commitment to that letter.\n")
    P("| source | arm | traces | mean tokens | mean POST frac | median POST frac | mean POST tok | total POST share | mean post-LAST frac |")
    P("|---|---|---|---|---|---|---|---|---|")
    waste = {}
    for s in ("mmlu", "gpqa", "ceval"):
        for a in ("base", "swift"):
            m = (src == s) & (arm == a) & det
            post = (n - dec)[m]; frac = post / n[m]
            postl = ((n - lastt) / n)[m]
            waste[(s, a)] = dict(post=post.sum(), tot=n[m].sum(), cnt=m.sum(), mean_post=post.mean())
            P(f"| {s} | {a} | {m.sum()} | {n[m].mean():.0f} | {frac.mean():.3f} | {np.median(frac):.3f} | {post.mean():.0f} | {post.sum() / n[m].sum():.3f} | {postl.mean():.3f} |")
    P("\n### Swift POST vs base POST (paired on (seed, question) where both arms correct & decision found)\n")
    P("| source | pairs | base mean tok | swift mean tok | base mean POST | swift mean POST | swift/base POST | swift/base total | base PRE | swift PRE | swift/base PRE |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    idx = {(meta[i]["src"], meta[i]["arm"], meta[i]["key"]): i for i in range(N)}
    for s in ("mmlu", "gpqa", "ceval"):
        pb, ps = [], []
        for i in np.where((src == s) & (arm == "base") & det)[0]:
            j = idx.get((s, "swift", meta[i]["key"]))
            if j is not None and det[j]:
                pb.append(i); ps.append(j)
        pb = np.array(pb); ps = np.array(ps)
        if len(pb) == 0:
            continue
        bp, sp = (n - dec)[pb], (n - dec)[ps]
        P(f"| {s} | {len(pb)} | {n[pb].mean():.0f} | {n[ps].mean():.0f} | {bp.mean():.0f} | {sp.mean():.0f} | {sp.sum() / bp.sum():.3f} | {n[ps].sum() / n[pb].sum():.3f} | {dec[pb].mean():.0f} | {dec[ps].mean():.0f} | {dec[ps].sum() / dec[pb].sum():.3f} |")
    P("\n### MMLU-Pro POST fraction by category\n")
    P("| category | base frac (mean) | swift frac (mean) | base POST tok | swift POST tok | swift/base POST tok |")
    P("|---|---|---|---|---|---|")
    for c in sorted(set(cat[src == "mmlu"])):
        mb = (src == "mmlu") & (arm == "base") & det & (cat == c)
        ms = (src == "mmlu") & (arm == "swift") & det & (cat == c)
        fb = ((n - dec) / np.maximum(n, 1))[mb].mean(); fs = ((n - dec) / np.maximum(n, 1))[ms].mean()
        P(f"| {c} | {fb:.3f} | {fs:.3f} | {(n - dec)[mb].mean():.0f} | {(n - dec)[ms].mean():.0f} | {(n - dec)[ms].mean() / (n - dec)[mb].mean():.3f} |")

    # ---------------- 3: log-odds POST vs PRE
    SRC_M = {v: (src == v)[tr_of] for v in ('mmlu', 'gpqa', 'ceval')}; ARM_M = {v: (arm == v)[tr_of] for v in ('base', 'swift')}
    bg = np.bincount(ids[region > 0], minlength=V).astype(np.float64)
    bg_p = (bg + 0.01) / (bg + 0.01).sum()
    A0 = args.alpha0

    def cnts(mask):
        return np.bincount(ids[mask], minlength=V)

    res = {}
    tok_post = tok_pre = None
    groups = {}
    for s in ("mmlu", "gpqa", "ceval", "all"):
        for a in ("base", "swift", "pooled"):
            ms = np.ones(len(ids), bool) if s == "all" else SRC_M[s]
            ma = np.ones(len(ids), bool) if a == "pooled" else ARM_M[a]
            yP = cnts(ms & ma & (region == 2)); yR = cnts(ms & ma & (region == 1))
            d, z = monroe(yP, yP.sum(), yR, yR.sum(), bg_p, A0)
            res[(s, a)] = dict(yP=yP, yR=yR, nP=yP.sum(), nR=yR.sum(), d=d, z=z)
        print("log-odds done", s, flush=True)
    resL = {}
    for s in ("mmlu", "gpqa", "ceval", "all"):
        for a in ("base", "swift"):
            ms = np.ones(len(ids), bool) if s == "all" else SRC_M[s]
            yP = res[(s, a)]["yP"]; yL = cnts(ms & ARM_M[a] & lateR)
            d, z = monroe(yP, yP.sum(), yL, yL.sum(), bg_p, A0)
            resL[(s, a)] = dict(yL=yL, nL=yL.sum(), d=d, z=z)
    # per MMLU category (swift)
    cats = sorted(set(cat[src == "mmlu"]))
    cat_z = []
    for c in cats:
        mm = SRC_M["mmlu"] & ARM_M["swift"] & (cat == c)[tr_of]
        yP = cnts(mm & (region == 2)); yR = cnts(mm & (region == 1))
        _, z = monroe(yP, yP.sum(), yR, yR.sum(), bg_p, A0)
        cat_z.append(z)
    cat_z = np.stack(cat_z)
    cat_frac = (cat_z >= 3).mean(0)

    # ---------------- overall rates (all non-truncated traces) base vs swift, per source
    nontr = (~trunc)[tr_of]
    rate = {}
    for s in ("mmlu", "gpqa", "ceval", "all"):
        ms = np.ones(len(ids), bool) if s == "all" else SRC_M[s]
        for a in ("base", "swift"):
            mm = ms & nontr & ARM_M[a]
            c = cnts(mm)
            rate[(s, a)] = (c, mm.sum())
    # trace-level presence in POST (swift) to guard against few-trace bursts
    # (fraction of detected swift traces whose POST contains the token)
    sw_det = np.where(det & (arm == "swift"))[0]

    # ---------------- 4: flip-flop windows
    W = 64
    ff_mask = np.zeros(len(ids), bool)
    ff_stats = {}
    for a in ("base", "swift"):
        for s in ("mmlu", "gpqa", "ceval"):
            m = (src == s) & (arm == a) & ~trunc & (n > 0)
            k = np.array([len(meta[i]["switches"]) > 0 for i in np.where(m)[0]])
            kc = np.array([len(meta[i]["switches"]) > 0 for i in np.where(m & corr)[0]])
            multi = np.array([len(set(meta[i]["letters"])) > 1 for i in np.where(m)[0]])
            ff_stats[(s, a)] = (m.sum(), k.mean() if len(k) else 0, kc.mean() if len(kc) else 0,
                                np.mean([len(meta[i]["switches"]) for i in np.where(m)[0]]))
    ctl_mask = np.zeros(len(ids), bool)
    for i in range(N):
        if trunc[i]:
            continue
        swset = set(meta[i]["switches"])
        for t in meta[i]["switches"]:
            lo = off[i] + max(0, t - W); hi = off[i] + min(n[i], t + W)
            ff_mask[lo:hi] = True
        # control: windows around commitments that do NOT switch letter
        for t in meta[i]["ctoks"]:
            if t in swset:
                continue
            lo = off[i] + max(0, t - W); hi = off[i] + min(n[i], t + W)
            ctl_mask[lo:hi] = True
    ctl_mask &= ~ff_mask
    ff = {}
    for a in ("base", "swift", "pooled"):
        ma = np.ones(len(ids), bool) if a == "pooled" else ARM_M[a]
        yW = cnts(ma & nontr & ff_mask); yO = cnts(ma & nontr & ctl_mask)
        bgw = (yW + yO + 0.01) / (yW + yO + 0.01).sum()
        d, z = monroe(yW, yW.sum(), yO, yO.sum(), bgw, A0)
        ff[a] = dict(yW=yW, yO=yO, d=d, z=z)

    # ---------------- candidate selection
    def text_of(i):
        return tok.decode([int(i)])

    def norm(t):
        return t.strip().lower().lstrip("ġ▁")

    PUNCT = re.compile(r"^[\W_]+$", re.U)
    ART_WORDS = {"answer", "answers", "option", "options", "choice", "choices", "correct", "answer:", "答案", "选", "选项"}
    FUNC_WORDS = set("the a an is are was were be been being of to in on at by for with from as and that this these those it its it's "
                     "they them their he she we you i 's 'd 'll can will would should do does did not no if then so there here what which who "
                     "how when where all any some more most also only just very than too up out about into over such each".split())
    FMT_WORDS = {"need", "exact", "mention", "extra", "craft", "requested", "prompt", "instructions", "instruction", "explain",
                 "summary", "succinct", "minimal", "clear", "write", "writing", "final", "format", "line", "exactly", "output", "ensure", "concise", "explanation", "step", "steps",
                 "brief", "briefly", "short", "last", "include", "provide", "produce", "produced", "respond", "response",
                 "user", "asks", "ask", "wants", "want", "reasoning", "think", "thinking"}

    def flags_for(t):
        f = []
        s = t.strip()
        sl = s.lower().strip("()（）")
        if re.fullmatch(r"\(?[A-Ja-j]\)?[.:)]?", s):
            f.append("letter")
        if sl in ART_WORDS:
            f.append("answer_word")
        if s and re.fullmatch(r"[\d.,]+", s):
            f.append("digit")
        if s == "" or PUNCT.match(s):
            f.append("punct_space")
        if sl in FMT_WORDS:
            f.append("format_planning")
        if sl in FUNC_WORDS:
            f.append("function_word")
        return f

    rS = res[("mmlu", "swift")]
    consist_src = np.zeros(V, int)
    pos_src_list = [("mmlu", "swift"), ("gpqa", "swift"), ("ceval", "swift")]
    for key in pos_src_list:
        consist_src += (res[key]["z"] >= 3).astype(int)
    # pooled arms per source as alternative consistency
    consist_pool = np.zeros(V, int)
    for s in ("mmlu", "gpqa", "ceval"):
        consist_pool += (res[(s, "pooled")]["z"] >= 3).astype(int)
    allS = res[("all", "swift")]
    minc = args.min_post
    allL = resL[("all", "swift")]
    ok = (allS["yP"] >= minc) & (allS["z"] >= 3) & ((consist_src >= 2) | (cat_frac >= 0.7)) \
        & (allL["z"] >= 3) & (allL["d"] >= math.log(1.25))
    score = np.minimum(allS["z"], allL["z"])
    cand_ids = [i for i in np.where(ok)[0] if i not in pool_ids]
    cand_ids.sort(key=lambda i: -score[i])
    print("candidates", len(cand_ids), flush=True)

    # examples: search swift POST regions for occurrences
    rng = random.Random(0)
    det_sw = [i for i in sw_det]
    rng.shuffle(det_sw)
    need = set(cand_ids[:args.top_examples])
    ex = defaultdict(list)
    for i in det_sw:
        if not need:
            break
        seg = ids[off[i] + dec[i]: off[i] + n[i]]
        hit = need.intersection(np.unique(seg).tolist())
        for h in hit:
            p = int(np.where(seg == h)[0][0]) + dec[i]
            ctx = tok.decode(ids[off[i] + max(0, p - 20): off[i] + p].tolist()) + "【" + text_of(h) + "】" + \
                  tok.decode(ids[off[i] + p + 1: off[i] + min(n[i], p + 16)].tolist())
            ex[h].append(dict(src=meta[i]["src"], key=str(meta[i]["key"]), ctx=ctx.replace("\n", "⏎")))
            if len(ex[h]) >= 3:
                need.discard(h)

    pres_sw = None  # computed only for candidates below
    cands = []
    tot_sw_tok = rate[("all", "swift")][1]; tot_b_tok = rate[("all", "base")][1]
    for rank, i in enumerate(cand_ids):
        t = text_of(i)
        rb = rate[("all", "base")][0][i] / tot_b_tok * 1e4
        rs_ = rate[("all", "swift")][0][i] / tot_sw_tok * 1e4
        per_src_ratio = {}
        for s in ("mmlu", "gpqa", "ceval"):
            cb, nb = rate[(s, "base")]; cs, ns = rate[(s, "swift")]
            per_src_ratio[s] = round(float((cs[i] / ns) / (cb[i] / nb)), 3) if cb[i] > 0 else None
        e = dict(rank=rank + 1, id=int(i), text=t, group=norm(t), flags=flags_for(t),
                 z={f"{s}_{a}": round(float(res[(s, a)]["z"][i]), 2) for s in ("mmlu", "gpqa", "ceval", "all") for a in ("swift", "base", "pooled")},
                 delta_all_swift=round(float(allS["d"][i]), 3), score=round(float(score[i]), 2),
                 z_vs_latePRE={f"{s}_{a}": round(float(resL[(s, a)]["z"][i]), 2) for s in ("mmlu", "gpqa", "ceval", "all") for a in ("swift", "base")},
                 odds_ratio_vs_latePRE_swift=round(float(math.exp(allL["d"][i])), 3),
                 odds_ratio_all_swift=round(float(math.exp(allS["d"][i])), 3),
                 swift_post_count=int(allS["yP"][i]), swift_pre_count=int(allS["yR"][i]),
                 swift_post_per10k=round(float(allS["yP"][i] / allS["nP"] * 1e4), 3),
                 swift_pre_per10k=round(float(allS["yR"][i] / allS["nR"] * 1e4), 3),
                 base_post_per10k=round(float(res[("all", "base")]["yP"][i] / res[("all", "base")]["nP"] * 1e4), 3),
                 base_pre_per10k=round(float(res[("all", "base")]["yR"][i] / res[("all", "base")]["nR"] * 1e4), 3),
                 mmlu_cat_frac_z3=round(float(cat_frac[i]), 3), n_sources_z3_swift=int(consist_src[i]),
                 overall_base_per10k=round(float(rb), 3), overall_swift_per10k=round(float(rs_), 3),
                 swift_base_ratio=round(float(rs_ / rb), 3) if rb > 0 else None,
                 swift_base_ratio_by_src=per_src_ratio,
                 base_post_z_all=round(float(res[("all", "base")]["z"][i]), 2),
                 flipflop_z_swift=round(float(ff["swift"]["z"][i]), 2), flipflop_z_pooled=round(float(ff["pooled"]["z"][i]), 2),
                 examples=ex.get(i, []))
        cands.append(e)
    # variant groups
    grp = defaultdict(list)
    for e in cands:
        grp[e["group"]].append(e["id"])
    for e in cands:
        e["group_ids"] = grp[e["group"]]
    json.dump(cands, open(f"{OUT}/candidates.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # full per-token stat arrays for ad-hoc queries
    np.savez_compressed(f"{OUT}/token_stats.npz",
                        **{f"z_{s}_{a}": res[(s, a)]["z"] for (s, a) in res},
                        **{f"yP_{s}_{a}": res[(s, a)]["yP"] for (s, a) in res},
                        **{f"yR_{s}_{a}": res[(s, a)]["yR"] for (s, a) in res},
                        **{f"d_{s}_{a}": res[(s, a)]["d"] for (s, a) in res},
                        **{f"zlate_{s}_{a}": resL[(s, a)]["z"] for (s, a) in resL},
                        **{f"dlate_{s}_{a}": resL[(s, a)]["d"] for (s, a) in resL},
                        **{f"rate_{s}_{a}": rate[(s, a)][0] for (s, a) in rate},
                        **{f"ntok_{s}_{a}": np.array(rate[(s, a)][1]) for (s, a) in rate},
                        **{f"ffz_{a}": ff[a]["z"] for a in ff}, **{f"ffW_{a}": ff[a]["yW"] for a in ff},
                        **{f"ffd_{a}": ff[a]["d"] for a in ff}, cat_frac=cat_frac)

    # ---------------- report
    P(f"\n## 3. POST vs PRE log-odds candidates\n")
    P(f"Monroe et al. informative-Dirichlet log-odds, alpha0={A0}, background = pooled PRE+POST counts. "
      f"Filter: swift POST count (all sources) >= {minc}, z_all_swift >= 3, and (z>=3 in >=2 of 3 sources [swift arm] "
      f"or z>=3 in >=70% of {len(cats)} MMLU-Pro categories [swift]). Pool ids excluded ({len(pool_ids)}). "
      f"{len(cands)} candidates pass. Additionally requires z>=3 and OR>=1.25 vs the late-PRE control (<=512 PRE tokens right before the decision, skipping question restatement). Ranked by min(z vs PRE, z vs late-PRE). Overall rates are per 10k reasoning tokens over all non-truncated traces. ff z = flip-flop window z (switch windows vs non-switch commitment windows).\n")
    P("Token totals (swift): POST=%d PRE=%d; (base): POST=%d PRE=%d\n" % (allS["nP"], allS["nR"], res[("all", "base")]["nP"], res[("all", "base")]["nR"]))

    def table(rows_, title):
        P(f"\n### {title}\n")
        P("| # | id | text | group | z mmlu/gpqa/ceval (swift) | z base all | z/OR vs latePRE swift | OR post/pre swift | swift POST/10k | PRE/10k | overall base/10k | swift/10k | swift/base | ff z | cat%z3 | flags |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k, e in enumerate(rows_):
            z = e["z"]
            P(f"| {k + 1} | {e['id']} | {md(e['text'])} | {e['group']} | {z['mmlu_swift']}/{z['gpqa_swift']}/{z['ceval_swift']} | {z['all_base']} | {e['z_vs_latePRE']['all_swift']}/{e['odds_ratio_vs_latePRE_swift']} | {e['odds_ratio_all_swift']} | {e['swift_post_per10k']} | {e['swift_pre_per10k']} | {e['overall_base_per10k']} | {e['overall_swift_per10k']} | {e['swift_base_ratio']} | {e['flipflop_z_swift']} | {e['mmlu_cat_frac_z3']} | {','.join(e['flags'])} |")

    clean = [e for e in cands if not e["flags"]]
    table(clean[:80], "Top clean candidates (no artifact/format/function-word flags)")
    table([e for e in cands if "format_planning" in e["flags"]][:40], "Format-planning flagged candidates")
    table([e for e in cands if e["flags"] == ["function_word"]][:30], "Function-word flagged candidates")
    table([e for e in cands if e["flags"] and "format_planning" not in e["flags"] and e["flags"] != ["function_word"]][:30], "Artifact-flagged (letters/answer/digits/punct)")
    # variant-group aggregates
    P("\n### Variant groups (all case/space variants in vocab summed, pool ids removed)\n")
    P("| group | ids | swift POST cnt | z vs PRE (swift) | z vs latePRE (swift) | swift/base overall | flags |")
    P("|---|---|---|---|---|---|---|")
    gmembers = defaultdict(list)
    for i_ in range(tok.get_vocab_size()):
        gmembers[norm(tok.decode([i_]))].append(i_)
    seen = set()
    for e in cands:
        g = e["group"]
        if g in seen or not g:
            continue
        seen.add(g)
        mem = [i_ for i_ in gmembers.get(g, [e["id"]]) if i_ not in pool_ids and i_ < V]
        yP = sum(int(allS["yP"][i_]) for i_ in mem); yR = sum(int(allS["yR"][i_]) for i_ in mem); yL = sum(int(allL["yL"][i_]) for i_ in mem)
        a_ = float(sum(bg_p[i_] for i_ in mem)) * A0

        def zz(y1, n1, y2, n2):
            d_ = math.log((y1 + a_) / (n1 + A0 - y1 - a_)) - math.log((y2 + a_) / (n2 + A0 - y2 - a_))
            return d_ / math.sqrt(1 / (y1 + a_) + 1 / (y2 + a_))
        cb = sum(int(rate[("all", "base")][0][i_]) for i_ in mem) / tot_b_tok
        cs = sum(int(rate[("all", "swift")][0][i_]) for i_ in mem) / tot_sw_tok
        P(f"| {g!r} | {mem} | {yP} | {zz(yP, allS['nP'], yR, allS['nR']):.1f} | {zz(yP, allS['nP'], yL, allL['nL']):.1f} | {cs / cb if cb else float('nan'):.2f} | {','.join(e['flags'])} |")
        if len(seen) >= 120:
            break
    P("\n### Examples (top 30 clean)\n")
    for e in clean[:30]:
        P(f"- **{e['text']!r}** (id {e['id']}):")
        for x in e["examples"]:
            P(f"    - [{x['src']} {x['key']}] {x['ctx']}")

    P("\n## 4. Flip-flop (commitments to different letters)\n")
    P("| source | arm | non-trunc traces | frac with switch | frac with switch (correct) | mean switches |")
    P("|---|---|---|---|---|---|")
    for (s, a), v in sorted(ff_stats.items()):
        P(f"| {s} | {a} | {v[0]} | {v[1]:.3f} | {v[2]:.3f} | {v[3]:.3f} |")
    P(f"\nWindow tokens (±{W}) swift={ff['swift']['yW'].sum()} base={ff['base']['yW'].sum()}\n")
    for a in ("swift", "pooled"):
        z = ff[a]["z"].copy()
        z[(ff[a]["yW"] < args.min_ff)] = -np.inf
        top = [i for i in np.argsort(-z) if i not in pool_ids][:40]
        P(f"\n### Top flip-flop window tokens ({a}), pool excluded, window count >= {args.min_ff}\n")
        P("| id | text | z | OR win/else | win count | swift/base overall | flags | in POST cand list |")
        P("|---|---|---|---|---|---|---|---|")
        cset = {e["id"] for e in cands}
        for i in top:
            cb = rate[("all", "base")][0][i] / tot_b_tok; cs = rate[("all", "swift")][0][i] / tot_sw_tok
            P(f"| {i} | {md(text_of(i))} | {ff[a]['z'][i]:.1f} | {math.exp(ff[a]['d'][i]):.2f} | {ff[a]['yW'][i]} | {cs / cb if cb else float('nan'):.2f} | {','.join(flags_for(text_of(i)))} | {'Y' if i in cset else ''} |")
        if a == "swift":
            json.dump([dict(id=int(i), text=text_of(i), z=round(float(ff[a]["z"][i]), 2), or_=round(math.exp(ff[a]["d"][i]), 3),
                            win_count=int(ff[a]["yW"][i]), flags=flags_for(text_of(i))) for i in top],
                      open(f"{OUT}/flipflop_top.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # pool sanity: how do the known 47 behave in this contrast?
    P("\n## 5. Sanity: recovered pool tokens in the same contrast\n")
    P("| id | text | z all swift POST/PRE | z all base | z swift vs latePRE | swift/base overall | ff z swift |")
    P("|---|---|---|---|---|---|---|")
    for p in pool:
        i = p["id"]
        cb = rate[("all", "base")][0][i] / tot_b_tok; cs = rate[("all", "swift")][0][i] / tot_sw_tok
        P(f"| {i} | {md(p['text'])} | {allS['z'][i]:.1f} | {res[('all', 'base')]['z'][i]:.1f} | {allL['z'][i]:.1f} | {cs / cb if cb else float('nan'):.2f} | {ff['swift']['z'][i]:.1f} |")
    open(f"{OUT}/REPORT_auto.md", "w", encoding="utf-8").write("\n".join(rep))
    print("wrote", OUT)


def stage_variants(args):
    """Case/space siblings of recovered pool tokens that are NOT in the pool (uses token_stats.npz)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK)
    S = np.load(f"{OUT}/token_stats.npz")
    pool = json.load(open(POOL, encoding="utf-8"))
    pool_ids = {p["id"] for p in pool}
    V = len(S["z_all_swift"])
    groups = defaultdict(list)
    for i in range(min(V, tok.get_vocab_size())):
        t = tok.decode([i])
        k = t.strip().lower()
        if k:
            groups[k].append(i)
    nb, ns = float(S["ntok_all_base"]), float(S["ntok_all_swift"])
    out = []
    for p in pool:
        k = p["text"].strip().lower()
        for i in groups.get(k, []):
            if i in pool_ids:
                continue
            cb, cs = int(S["rate_all_base"][i]), int(S["rate_all_swift"][i])
            if cb + cs < 50:
                continue
            out.append(dict(id=int(i), text=tok.decode([i]), sibling_of=p["text"], sibling_id=p["id"],
                            base_count=cb, swift_count=cs, base_per10k=round(cb / nb * 1e4, 3), swift_per10k=round(cs / ns * 1e4, 3),
                            swift_base_ratio=round((cs / ns) / (cb / nb), 3) if cb else None,
                            sibling_swift_base_ratio=round((S["rate_all_swift"][p["id"]] / ns) / (S["rate_all_base"][p["id"]] / nb), 3) if S["rate_all_base"][p["id"]] else None,
                            z_post_vs_pre_swift=round(float(S["z_all_swift"][i]), 2), z_post_vs_pre_base=round(float(S["z_all_base"][i]), 2),
                            z_post_vs_latepre_swift=round(float(S["zlate_all_swift"][i]), 2),
                            flipflop_z_swift=round(float(S["ffz_swift"][i]), 2)))
    out.sort(key=lambda e: -e["swift_count"])
    json.dump(out, open(f"{OUT}/pool_variant_gaps.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    lines = ["| id | text | sibling in pool | swift cnt | swift/10k | swift/base | sibling swift/base | z POST/PRE swift | z POST/PRE base | ff z swift |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for e in out:
        lines.append(f"| {e['id']} | {md(e['text'])} | {md(e['sibling_of'])} | {e['swift_count']} | {e['swift_per10k']} | {e['swift_base_ratio']} | {e['sibling_swift_base_ratio']} | {e['z_post_vs_pre_swift']} | {e['z_post_vs_pre_base']} | {e['flipflop_z_swift']} |")
    open(f"{OUT}/pool_variant_gaps.md", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


# hand-curated tiers (chosen by reading REPORT_auto.md tables + example contexts); stats pulled from token_stats.npz
CURATED = {
    "A_pool_variant_gaps": [16018, 12525, 10380, 20734, 32645, 3850, 10560, 15896, 18200, 35542, 12908, 13634, 10883],
    "B_post_decision_verification": [7144, 15464, 92306, 7920, 4125, 16097, 13665, 22149, 2814, 3840, 69330, 83081, 82170, 2617, 10589],
    "C_self_affirmation_closure": [7441, 7179, 29169, 16415, 10092, 6820, 6699],
    "D_testmaker_speculation": [10031, 1328, 2624, 4779, 7658, 7656, 10287, 3889, 23535, 1228, 27502, 4674, 2450, 1380],
    "E_format_replanning": [28320, 61446, 22916, 1534, 12650, 3443, 9522, 28763, 38256, 3300, 2830, 15673, 6681, 4581, 6088, 4799,
                            10453, 85027, 63008, 11346, 10897, 20500, 13909, 28253, 809, 1500, 1483, 9764],
}


def stage_curate(args):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK)
    S = np.load(f"{OUT}/token_stats.npz")
    cands = {e["id"]: e for e in json.load(open(f"{OUT}/candidates.json", encoding="utf-8"))}
    nb, ns = float(S["ntok_all_base"]), float(S["ntok_all_swift"])
    out = {}
    lines = []
    for tier, idl in CURATED.items():
        rows = []
        for i in idl:
            cb, cs = int(S["rate_all_base"][i]), int(S["rate_all_swift"][i])
            r = dict(id=i, text=tok.decode([i]), swift_count=cs, swift_per10k=round(cs / ns * 1e4, 3),
                     swift_base_ratio=round((cs / ns) / (cb / nb), 3) if cb else None,
                     or_post_vs_pre_swift=round(math.exp(float(S["d_all_swift"][i])), 2),
                     or_post_vs_latepre_swift=round(math.exp(float(S["dlate_all_swift"][i])), 2),
                     z_post_vs_pre_swift={s: round(float(S[f"z_{s}_swift"][i]), 1) for s in ("mmlu", "gpqa", "ceval", "all")},
                     z_post_vs_pre_base_all=round(float(S["z_all_base"][i]), 1),
                     flipflop_z_swift=round(float(S["ffz_swift"][i]), 1),
                     in_candidates=i in cands, candidate_rank=cands[i]["rank"] if i in cands else None,
                     examples=cands[i]["examples"] if i in cands else [])
            rows.append(r)
        out[tier] = rows
        lines.append(f"\n### {tier}\n")
        lines.append("| id | text | swift cnt | swift/10k | swift/base | OR POST/PRE | OR POST/latePRE | z m/g/c (swift) | z base | ff z |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            z = r["z_post_vs_pre_swift"]
            lines.append(f"| {r['id']} | {md(r['text'])} | {r['swift_count']} | {r['swift_per10k']} | {r['swift_base_ratio']} | {r['or_post_vs_pre_swift']} | {r['or_post_vs_latepre_swift']} | {z['mmlu']}/{z['gpqa']}/{z['ceval']} | {r['z_post_vs_pre_base_all']} | {r['flipflop_z_swift']} |")
    json.dump(out, open(f"{OUT}/curated_tiers.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(f"{OUT}/curated_tiers.md", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["load", "detect", "analyze", "variants", "curate", "all"])
    ap.add_argument("--mmlu-seeds", default="0,1")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha0", type=float, default=10000.0)
    ap.add_argument("--min-post", type=int, default=200)
    ap.add_argument("--min-ff", type=int, default=50)
    ap.add_argument("--top-examples", type=int, default=150)
    args = ap.parse_args()
    if args.stage == "detect":
        stage_detect(args)
    if args.stage in ("load", "all"):
        stage_load(args)
    if args.stage in ("analyze", "all"):
        stage_analyze(args)
    if args.stage in ("variants", "all"):
        stage_variants(args)
    if args.stage in ("curate", "all"):
        stage_curate(args)
