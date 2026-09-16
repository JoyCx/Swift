"""Loop detector over long reasoning traces.
Units = paragraphs (blank-line split), long paragraphs split on newlines then sentences.
Unit repr = normalised word 4-gram shingles (lowercase, numbers->#, punctuation dropped).
Near-dup match: exact Jaccard over shingle sets >= THETA vs any earlier unit (inverted index).
Also topical match: content-word set Jaccard >= THETA_BAG (secondary).
Exact token repetition: 32-gram BPE id repeats.
"""
import re, pickle, json, sys, os, bisect, math
from collections import defaultdict, Counter
import numpy as np
from multiprocessing import Pool

THETA = float(os.environ.get('THETA', 0.5))
THETA_BAG = 0.6
MIN_SH = 8          # min shingles for unit to be matchable
MIN_BAG = 10
K = 4
STOP = set("""a an the of to in on for and or but is are be was were it this that these those we i you need so then if else with as at by from not no
can could would should will may might do does did have has had just also maybe let lets us our its than there their them they which what when
where how why all any each some more most other such only same into out up about over after before because while now use using get got one two s""".split())
WRE = re.compile(r"[a-z_][a-z_0-9]*|\d+(?:\.\d+)?|[=<>+\-*/%^!&|]+")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_units(text):
    units = []
    pos = 0
    for m in re.finditer(r"\n\s*\n", text):
        units.append((pos, m.start())); pos = m.end()
    units.append((pos, len(text)))
    out = []
    for s, e in units:
        if e - s <= 1200:
            if text[s:e].strip(): out.append((s, e))
            continue
        sub = []; p = s
        for m in re.finditer(r"\n", text[s:e]):
            sub.append((p, s + m.start())); p = s + m.end()
        sub.append((p, e))
        buf = None
        for a, b in sub:
            if b - a > 1200:
                if buf: out.append(buf); buf = None
                q = a
                for m in SENT.finditer(text[a:b]):
                    cut = a + m.start()
                    if cut - q >= 400: out.append((q, cut)); q = a + m.end()
                out.append((q, b))
            else:
                buf = (a, b) if buf is None else (buf[0], b)
                if buf[1] - buf[0] >= 400: out.append(buf); buf = None
        if buf: out.append(buf)
    return [(a, b) for a, b in out if text[a:b].strip()]


def norm_words(s):
    w = WRE.findall(s.lower())
    return ['#' if c[0].isdigit() else c for c in w]


def shingles(words):
    if len(words) < K: return set()
    return {hash(' '.join(words[i:i+K])) for i in range(len(words) - K + 1)}


CODE_RE = re.compile(r"^\s*(def |class |for |while |if |return |import |from |#include|int |[a-zA-Z_][\w\[\]\.]*\s*=[^=]|\}|\{|print\()", re.M)


def code_frac(s):
    lines = [l for l in s.split('\n') if l.strip()]
    if not lines: return 0.0
    return sum(1 for l in lines if CODE_RE.match(l) or l.startswith('    ')) / len(lines)


def exact_repeat_mask(ids, n=32):
    """mask of tokens inside a repeated 32-gram; degenerate = back-to-back short-period repetition."""
    L = len(ids); mask = np.zeros(L, dtype=np.uint8)
    if L < n: return mask, 0, 0
    last = {}
    dist = np.full(L, 10**9, dtype=np.int64)
    t = tuple(ids)
    for i in range(L - n + 1):
        v = hash(t[i:i+n])
        j = last.get(v)
        if j is not None:
            mask[i:i+n] = 1; dist[i] = i - j
        last[v] = i
    small = dist <= 512
    degen = 0; maxrun = 0; i = 0
    while i < L:
        if not small[i]:
            i += 1; continue
        j = i
        while j < L and small[j]: j += 1
        blen = j - i + n - 1
        md = int(np.median(dist[i:j]))
        if blen >= max(2 * md, 128):
            degen += blen
            maxrun = max(maxrun, blen)
        i = j
    return mask, degen, maxrun


def analyse(t):
    text = t['text']; ids = t['ids'].tolist(); starts = t['starts']
    units = split_units(text)
    U = []
    for a, b in units:
        s = text[a:b]; w = norm_words(s)
        sh = shingles(w); wr = WRE.findall(s.lower()); shr = shingles(wr)
        cw = [len(x) >= 3 and x[0].isalpha() and x not in STOP for x in w]
        pos_sh = [(hash(' '.join(w[q:q+K])), hash(' '.join(wr[q:q+K])), any(cw[q:q+K])) for q in range(max(0, len(w) - K + 1))]
        bag = {x for x in w if x not in STOP and x != '#' and len(x) > 1 and x[0].isalpha()}
        ta = int(np.searchsorted(starts, a)); tb_ = int(np.searchsorted(starts, b))
        U.append(dict(a=a, b=b, ta=ta, tb=max(tb_, ta), sh=sh, shr=shr, bag=bag, code=code_frac(s), pos_sh=pos_sh))
    n = len(U)
    inv = defaultdict(list); invb = defaultdict(list)
    df = Counter()
    for u in U:
        for x in u['sh']: df[x] += 1
    match = [-1] * n; mj = [0.0] * n; bmatch = [-1] * n; bj = [0.0] * n
    for i, u in enumerate(U):
        if len(u['sh']) >= MIN_SH:
            cnt = Counter()
            for x in u['sh']:
                if df[x] > 60: continue
                for j in inv[x]: cnt[j] += 1
            best = -1; bv = 0.0
            for j, c in cnt.items():
                jac = c / (len(u['sh']) + len(U[j]['sh']) - c)
                if jac > bv: bv, best = jac, j
            match[i] = best; mj[i] = bv
            for x in u['sh']: inv[x].append(i)
        if len(u['bag']) >= MIN_BAG:
            cnt = Counter()
            for x in u['bag']:
                for j in invb[x]: cnt[j] += 1
            best = -1; bv = 0.0
            for j, c in cnt.items():
                jac = c / (len(u['bag']) + len(U[j]['bag']) - c)
                if jac > bv: bv, best = jac, j
            bmatch[i] = best; bj[i] = bv
            for x in u['bag']: invb[x].append(i)
    # global redundancy (containment in everything written earlier)
    seen_n = {}; seen_r = {}
    red = [0.0] * n; redr = [0.0] * n; rsrc = [-1] * n
    for i, u in enumerate(U):
        ps = u['pos_sh']
        if len(ps) >= 12:
            hit = 0; hitr = 0; srcs = Counter()
            for kn, kr, c in ps:
                j = seen_r.get(kr)
                if j is not None: hitr += 1
                if j is None and c: j = seen_n.get(kn)
                if j is not None:
                    hit += 1; srcs[j] += 1
            red[i] = hit / len(ps); redr[i] = hitr / len(ps)
            if srcs: rsrc[i] = srcs.most_common(1)[0][0]
        for kn, kr, c in ps:
            seen_n.setdefault(kn, i); seen_r.setdefault(kr, i)
    isred = [red[i] >= 0.5 and not (rsrc[i] == i - 1 and redr[i] < 0.3) for i in range(n)]
    red_tok = sum(U[i]['tb'] - U[i]['ta'] for i in range(n) if isred[i])
    red_tok_code = sum(U[i]['tb'] - U[i]['ta'] for i in range(n) if isred[i] and U[i]['code'] >= 0.5)
    rawj = [0.0] * n
    for i in range(n):
        if match[i] >= 0 and mj[i] >= THETA:
            A_, B_ = U[i]['shr'], U[match[i]]['shr']
            rawj[i] = len(A_ & B_) / max(1, len(A_ | B_))
    prog = [mj[i] >= THETA and (i - match[i]) == 1 and rawj[i] < 0.35 for i in range(n)]
    isnd = [mj[i] >= THETA and not prog[i] for i in range(n)]
    isloop = [isnd[i] or isred[i] for i in range(n)]
    for i in range(n):
        if isred[i] and not isnd[i]:
            match[i] = rsrc[i]; mj[i] = max(mj[i], 0.0)
    openers = Counter(); tmpl = [False] * n
    for i, u in enumerate(U):
        w4 = ' '.join(norm_words(text[u['a']:u['a'] + 80])[:4])
        if len(w4.split()) == 4:
            if openers[w4] >= 2: tmpl[i] = True
            openers[w4] += 1
    tmpl_tok = sum(U[i]['tb'] - U[i]['ta'] for i in range(n) if tmpl[i])
    top_openers = [(o, c) for o, c in openers.most_common(3) if c >= 3]
    iscode = [U[i]['code'] >= 0.5 for i in range(n)]
    istop = [bj[i] >= THETA_BAG for i in range(n)]
    par = list(range(n))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for i in range(n):
        if isloop[i]: par[f(i)] = f(match[i])
    clus = defaultdict(list)
    for i in range(n): clus[f(i)].append(i)
    visits = [len(v) for v in clus.values() if len(v) >= 2]
    cseq = [f(i) for i in range(n) if len(clus[f(i)]) >= 2]
    comp = [c for k, c in enumerate(cseq) if k == 0 or c != cseq[k-1]]
    aba = sum(1 for k in range(2, len(comp)) if comp[k] == comp[k-2] and comp[k-1] != comp[k])
    emask, degen_tok, degen_maxrun = exact_repeat_mask(ids)
    ntok = len(ids)
    loop_tok = sum(U[i]['tb'] - U[i]['ta'] for i in range(n) if isloop[i])
    nd_tok = sum(U[i]['tb'] - U[i]['ta'] for i in range(n) if isnd[i])
    loop_tok_code = sum(U[i]['tb'] - U[i]['ta'] for i in range(n) if isloop[i] and iscode[i])
    prog_tok = sum(U[i]['tb'] - U[i]['ta'] for i in range(n) if prog[i])
    code_tok = sum(U[i]['tb'] - U[i]['ta'] for i in range(n) if iscode[i])
    top_tok = sum(U[i]['tb'] - U[i]['ta'] for i in range(n) if istop[i] or isloop[i])
    iters = []
    i = 0
    while i < n:
        if not isloop[i]:
            i += 1; continue
        j = i
        while True:
            if j + 1 < n and isloop[j+1]:
                j += 1; continue
            if j + 2 < n and isloop[j+2] and (U[j+1]['tb'] - U[j+1]['ta']) < 40:
                j += 2; continue
            break
        span_units = list(range(i, j + 1))
        tok = U[j]['tb'] - U[i]['ta']
        iters.append(dict(u0=i, u1=j, ta=U[i]['ta'], tb=U[j]['tb'], tok=tok, src=match[i], src_ta=U[match[i]]['ta'],
                          src_tb=U[match[i]]['tb'], dist_units=i - match[i], dist_tok=U[i]['ta'] - U[match[i]]['ta'], jac=mj[i],
                          exact_frac=float(emask[U[i]['ta']:U[j]['tb']].mean()) if tok > 0 else 0.0,
                          code=float(np.mean([U[k]['code'] for k in span_units])),
                          visits=len(clus[f(i)]), loop_tok=sum(U[k]['tb'] - U[k]['ta'] for k in span_units if isloop[k]), rawj=rawj[i], nd=isnd[i], red=red[i]))
        i = j + 1
    nonloop_starts = [U[k]['ta'] for k in range(n) if not isloop[k] and not istop[k] and (k == 0 or not isloop[k-1])]
    res = dict(bench=t['bench'], arm=t['arm'], tid=t['tid'], group=t['group'], pair=t.get('pair'), ntok=ntok,
               correct=t['correct'], truncated=t['truncated'], exc=t.get('exc'), difficulty=t.get('difficulty'),
               n_units=n, loop_units=sum(isloop), loop_tok=loop_tok, loop_share=loop_tok / max(ntok, 1),
               topical_tok=top_tok, topical_share=top_tok / max(ntok, 1),
               exact_rep_share=float(emask.mean()) if ntok else 0.0,
               loop_tok_code=loop_tok_code, loop_tok_prose=loop_tok - loop_tok_code, prog_tok=prog_tok, code_tok=code_tok,
               loop_share_prose=(loop_tok - loop_tok_code) / max(ntok, 1),
               nd_tok=nd_tok, nd_share=nd_tok / max(ntok, 1), degen_tok=degen_tok, degen_maxrun=degen_maxrun,
               tmpl_tok=tmpl_tok, tmpl_share=tmpl_tok / max(ntok, 1), top_openers=top_openers,
               red_tok=red_tok, red_share=red_tok / max(ntok, 1), red_share_prose=(red_tok - red_tok_code) / max(ntok, 1),
               n_iters=len(iters), n_iters_prose=sum(1 for it in iters if it['code'] < 0.5), max_visits=max(visits) if visits else 1, n_clusters_ge3=sum(1 for v in visits if v >= 3),
               aba=aba, max_cycle_tok=max((it['dist_tok'] for it in iters), default=0),
               max_iter_tok=max((it['tok'] for it in iters), default=0))
    # topic windows: tf-idf cosine between ~300-token windows (non-adjacent)
    wins = []; cur = []; cur_tok = 0; w0 = 0
    for i, u in enumerate(U):
        cur.extend(x for x in norm_words(text[u['a']:u['b']]) if x[0].isalpha() and len(x) >= 3 and x not in STOP)
        cur_tok += u['tb'] - u['ta']
        if cur_tok >= 300 or i == n - 1:
            wins.append((w0, i, Counter(cur))); cur = []; cur_tok = 0; w0 = i + 1
    topic_rev = []
    if len(wins) >= 4:
        vocab = {}
        for _, _, c in wins:
            for x in c: vocab.setdefault(x, len(vocab))
        M = np.zeros((len(wins), len(vocab)), dtype=np.float32)
        for r_, (_, _, c) in enumerate(wins):
            for x, v in c.items(): M[r_, vocab[x]] = 1 + math.log(v)
        dfw = (M > 0).sum(0)
        idf = np.log(len(wins) / dfw)
        idf[dfw > 0.5 * len(wins)] = 0.0
        M *= idf
        M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        S = M @ M.T
        for r_ in range(2, len(wins)):
            j = int(np.argmax(S[r_, :r_ - 1])); v = float(S[r_, j])
            topic_rev.append((U[wins[r_][0]]['ta'], U[wins[r_][1]]['tb'], round(v, 3), U[wins[j][0]]['ta'], U[wins[j][1]]['tb']))
    for th in (0.4, 0.5, 0.6):
        res[f'topic_rev_share_{int(th*10)}'] = sum(b_ - a_ for a_, b_, v, _, _ in topic_rev if v >= th) / max(ntok, 1)
    commits = commitments(text, starts, t['bench'])
    cseq2 = [c for _, c in commits]
    colf = [c for k, c in enumerate(cseq2) if k == 0 or c != cseq2[k-1]]
    seen_ = set(); returns = 0
    for k, c in enumerate(colf):
        if c in seen_: returns += 1
        seen_.add(c)
    res.update(n_commits=len(commits), n_distinct_commits=len(set(cseq2)), commit_switches=max(0, len(colf) - 1), flip_returns=returns)
    unit_rows = [(U[k]['ta'], U[k]['tb'], isloop[k], istop[k], round(mj[k], 3), match[k], round(bj[k], 3), bmatch[k], prog[k], iscode[k], round(rawj[k], 3), round(red[k], 3), rsrc[k]) for k in range(n)]
    return res, iters, nonloop_starts, unit_rows, commits, topic_rev


MCQ_PATS = [
    re.compile(r"(?:^|[.\n]\s*)(?:So|Thus|Hence|Therefore)\s*,?\s*(?:the\s+)?(?:answer\s*(?:is\s*)?)?(?:option\s*)?\(?([A-D])\)?\s*(?=[.\n]|$)"),
    re.compile(r"\b(?:I'?ll|I will|we'?ll|I would|we) (?:choose|pick|select|go with|answer|lean(?: toward| towards)?)\s*(?:option\s*)?\(?([A-D])\)?(?![/\w])"),
    re.compile(r"\b(?:final answer|correct answer|answer)\s*(?:is|:|=)\s*(?:option\s*)?\(?([A-D])\)?(?![/\w])"),
    re.compile(r"\blean(?:ing)?\s+(?:toward[s]?\s+)?(?:option\s*)?\(?([A-D])\)?(?![/\w])"),
]
NUM_PATS = [
    re.compile(r"\banswer\s*(?:is\s*|likely\s*|would be\s*|:\s*|=\s*)?\$?(?:\boxed\{)?\s*(-?\d+(?:/\d+)?)(?![\d.,/])", re.I),
    re.compile(r"boxed\s*\{?\s*(-?\d+(?:/\d+)?)(?![\d.,/])"),
]
HYPO = re.compile(r"\b(if|whether|possibility|maybe|perhaps|could|might|suppose|unless|check|not)\b[^.\n]{0,40}$", re.I)


def commitments(text, starts, bench):
    out = []
    if bench == 'gpqa':
        pats = MCQ_PATS
    elif bench in ('aime', 'hmmt'):
        pats = NUM_PATS
    else:
        return out
    seenpos = set()
    for P in pats:
        for m in P.finditer(text):
            if m.start(1) in seenpos: continue
            if HYPO.search(text[max(0, m.start() - 50):m.start(1)]): continue
            seenpos.add(m.start(1))
            out.append((int(np.searchsorted(starts, m.start(1))), m.group(1)))
    out.sort()
    return out


def work(t):
    try:
        return analyse(t)
    except Exception:
        import traceback; traceback.print_exc(); return None


if __name__ == '__main__':
    T = pickle.load(open('A:/swift/runs/loops/traces_light.pkl', 'rb'))
    lim = int(os.environ.get('LIMIT', 0))
    if lim:
        import random; random.seed(0); T = random.sample(T, lim)
    with Pool(8) as p:
        out = p.map(work, T, chunksize=2)
    stats = []; details = {}
    for t, o in zip(T, out):
        if o is None: continue
        res, iters, nls, units, commits, topic_rev = o
        stats.append(res)
        details[(t['bench'], t['arm'], t['tid'])] = dict(iters=iters, nonloop_starts=nls, units=units, commits=commits, topic_rev=topic_rev)
    tag = os.environ.get('TAG', '')
    with open(f'A:/swift/runs/loops/trace_stats{tag}.jsonl', 'w', encoding='utf-8') as fo:
        for r in stats: fo.write(json.dumps(r, ensure_ascii=False) + '\n')
    pickle.dump(details, open(f'A:/swift/runs/loops/details{tag}.pkl', 'wb'), protocol=5)
    print('done', len(stats))
