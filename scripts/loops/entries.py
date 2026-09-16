"""Loop entry / sustain / trigger token analysis + iteration categorisation.

Entry sets (per trace, prose only):
  unit  : first unit of each loop iteration (near-dup or redundant paragraph run), code<0.5
  topic : start of topic-revisit windows (tf-idf cosine >= 0.5 to a non-adjacent earlier window) not already in 'unit'
  code  : 24 tokens before a code-redraft iteration (the introducer)
Control: starts of prose units that are not loop units, not progression, not right after a loop unit, not topic revisits.
Stats: informative-Dirichlet-prior log-odds (Monroe et al. 2008), presence per window, grouped by stripped/lowercased token text.
"""
import pickle, json, re, math, os
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from tokenizers import Tokenizer

OUT = 'A:/swift/runs/loops'
tok = Tokenizer.from_file('A:/models/Qwen3.8-27B/tokenizer.json')
T = pickle.load(open(f'{OUT}/traces_light.pkl', 'rb'))
D = pickle.load(open(f'{OUT}/details.pkl', 'rb'))
pool = json.load(open('A:/swift/runs/markers/ukisai_pool_recovered.json', encoding='utf-8'))
pool_ids = {x['id'] for x in pool}
cand_ids = set()
ct_ = json.load(open('A:/swift/runs/markers/residual/curated_tiers.json', encoding='utf-8'))
cand_tier = {}
for tier, v in ct_.items():
    for x in v:
        cand_ids.add(x['id']); cand_tier.setdefault(x['id'], tier)
for x in json.load(open('A:/swift/runs/markers/residual/pool_variant_gaps.json', encoding='utf-8')):
    cand_ids.add(x['id']); cand_tier.setdefault(x['id'], 'pool_variant_gaps')

vocab_size = tok.get_vocab_size()
dec_cache = {}


def dec(i):
    s = dec_cache.get(i)
    if s is None:
        s = tok.decode([int(i)]); dec_cache[i] = s
    return s


def gkey(i):
    return dec(i).strip().lower()


NE = 12; PRE = 32
CODEISH = re.compile(r"[`{}();<>=\[\]#/\|$]|^\s*[\"'*\-\d]|https?:|(def|func|const|class|import|return|var|export|public|static|struct|fn|self|elif|else:|for \w+ in)")


def prose_ok(w):
    return not CODEISH.search(tok.decode([int(x) for x in w]))
CAT_RULES = [
    ('format_replan', re.compile(r"\b(final (answer|response|code|output|message|json)|format|json|delimiter|code block|concise|boxed|ensure (the )?(final|output|format|response)|provide (the )?(final|answer|explanation|code)|explanation|keystrokes|task_complete|respond|response)\b", re.I)),
    ('verification', re.compile(r"\b(verif\w*|check\w*|double|re-?check|confirm|test\w*|sanity|simulat\w*|example|recomput\w*|re-?comput\w*|trace|validat\w*|make sure|let'?s see if|edge case|potential (issue|bug|pitfall|problem)|hidden|careful)\b", re.I)),
    ('alternative_approach', re.compile(r"\b(alternativ\w*|another (approach|way|method|idea)|instead|different (approach|way|method)|other approach|what if|could we|could use|we could|maybe we can|consider using|or maybe|option|micro-?optim\w*|optimi[sz]\w*|faster|speed)\b", re.I)),
    ('hedge_reconsider', re.compile(r"\b(wait|hmm|actually|but|however|hold on|reconsider|maybe|perhaps|unless|not sure|hmm)\b", re.I)),
    ('restate_problem', re.compile(r"\b(problem|statement|task|question|requirement|says|constraint|given|asks|user)\b", re.I)),
]


def categorise(text_head, code, exact_frac, dist_units):
    if exact_frac >= 0.9 and dist_units <= 2:
        return 'degenerate_repeat'
    if code >= 0.5:
        return 'code_redraft'
    for name, rx in CAT_RULES:
        if rx.search(text_head):
            return name
    return 'rederivation_other'


sets = defaultdict(list)   # name -> list of (bench, arm, window ids)
iter_rows = []
sustain_loop = Counter(); sustain_ctl = Counter(); sustain_n = [0, 0]
rng = np.random.default_rng(0)
arm_tok = Counter()
for t in T:
    k = (t['bench'], t['arm'], t['tid'])
    d = D.get(k)
    if d is None: continue
    ids = t['ids']; st = t['starts']; text = t['text']; L = len(ids)
    arm_tok[(t['bench'], t['arm'])] += L
    units = d['units']
    loop_starts = set()
    for it in d['iters']:
        ta = it['ta']
        a = int(st[ta]) if ta < L else len(text)
        head = text[a:a + 300]
        cat = categorise(' '.join(head.split()[:40]), it['code'], it['exact_frac'], it['dist_units'])
        iter_rows.append(dict(bench=t['bench'], arm=t['arm'], tid=t['tid'], ta=ta, tb=it['tb'], tok=it['tok'], loop_tok=it['loop_tok'],
                              dist_tok=it['dist_tok'], dist_units=it['dist_units'], jac=it['jac'], red=it.get('red', 0), nd=it.get('nd', False),
                              code=it['code'], visits=it['visits'], exact_frac=it['exact_frac'], cat=cat, src_ta=it['src_ta'], src_tb=it['src_tb'],
                              head=' '.join(head.split()[:25]), correct=t['correct'], truncated=t['truncated'], ntok=L))
        if it['loop_tok'] < 30: continue
        if it['code'] >= 0.5:
            sets['code'].append((t['bench'], t['arm'], ids[max(0, ta - 24):ta], None, cat, k))
        elif prose_ok(ids[ta:ta + NE]):
            sets['unit'].append((t['bench'], t['arm'], ids[ta:ta + NE], ids[max(0, ta - PRE):ta], cat, k))
            loop_starts.add(ta)
    loopunit = {u[0] for u in units if u[2]}
    for a_, b_, v, sa, sb in d.get('topic_rev', []):
        if v >= 0.5 and a_ not in loop_starts and a_ not in loopunit and prose_ok(ids[a_:a_ + NE]):
            sets['topic'].append((t['bench'], t['arm'], ids[a_:a_ + NE], ids[max(0, a_ - PRE):a_], None, k))
            loop_starts.add(a_)
    topic_starts = {a_ for a_, b_, v, sa, sb in d.get('topic_rev', []) if v >= 0.5}
    prev_loop = False
    n_entries_here = len(loop_starts)
    ctl_here = []
    for u in units:
        ta, tb, isl, ist, mj, m, bj, bm, prog, isc, rawj, red, rsrc = u
        if not isl and not prog and not isc and not prev_loop and ta not in topic_starts and tb - ta >= 8 and prose_ok(ids[ta:ta + NE]):
            ctl_here.append((t['bench'], t['arm'], ids[ta:ta + NE], ids[max(0, ta - PRE):ta], None, k))
        prev_loop = isl
        # sustain: tokens inside prose loop units vs prose non-loop units, traces with loops only (subsample 30%)
        if d['iters'] and not isc and tb > ta and rng.random() < 0.3:
            seg = ids[ta:tb]
            if isl:
                sustain_loop.update(seg.tolist()); sustain_n[0] += len(seg)
            elif not prog:
                sustain_ctl.update(seg.tolist()); sustain_n[1] += len(seg)
    if n_entries_here and ctl_here:
        pick = rng.choice(len(ctl_here), size=min(len(ctl_here), 3 * n_entries_here), replace=False)
        sets['control'].extend(ctl_here[j] for j in pick)
        sets['control_all_n'].append(len(ctl_here))

# TB action-loop steps: reasoning start of looping steps vs other steps
sd = pd.read_json(f'{OUT}/tb_step_loops.jsonl', lines=True)
trials = {tr['trial']: tr for tr in pickle.load(open(f'{OUT}/tb_trials.pkl', 'rb'))}
LOOPK = {'retry_same_error', 'reread_same_output', 'json_format_fail', 'wait_empty', 'poll'}
tb_act = defaultdict(list)
steps_text = []
for r in sd.itertuples():
    st = trials[r.trial]['steps'][r.step]
    steps_text.append((r, st['reasoning'][:200]))
enc = tok.encode_batch([x[1] for x in steps_text], add_special_tokens=False)
for (r, _), e in zip(steps_text, enc):
    w = np.asarray(e.ids[:NE], dtype=np.int32)
    if len(w) == 0: continue
    name = 'tb_action_loop' if r.kind in LOOPK else 'tb_action_ctl'
    sets[name].append(('tb', r.arm, w, None, r.kind, r.trial))

print({k: len(v) for k, v in sets.items()})
print('entries per bench', Counter((x[0], x[1]) for x in sets['unit'] + sets['topic']))


def presence_counts(items, which=2):
    c = Counter(); n = 0
    for it in items:
        w = it[which]
        if w is None or len(w) == 0: continue
        c.update(set(int(x) for x in w)); n += 1
    return c, n


def group(c):
    g = Counter()
    for i, v in c.items(): g[gkey(i)] += v
    return g


def logodds(ca, cb, prior, a0=1000.0):
    na = sum(ca.values()); nb = sum(cb.values()); npr = sum(prior.values())
    out = {}
    for w in set(ca) | set(cb):
        al = a0 * prior.get(w, 0) / npr + 0.01
        ya = ca.get(w, 0); yb = cb.get(w, 0)
        d = math.log((ya + al) / (na + a0 - ya - al)) - math.log((yb + al) / (nb + a0 - yb - al))
        s = math.sqrt(1 / (ya + al) + 1 / (yb + al))
        out[w] = (d / s, d, ya, yb)
    return out


def ids_of_group():
    m = defaultdict(set)
    return m


# group -> member ids seen
members = defaultdict(Counter)
for name in ('unit', 'topic', 'control', 'tb_action_loop', 'tb_action_ctl', 'code'):
    for it in sets[name]:
        for which in (2, 3):
            w = it[which]
            if w is None: continue
            for x in w: members[gkey(int(x))][int(x)] += 1

loop_items = sets['unit'] + sets['topic']
ctl_items = sets['control']
ce, ne_ = presence_counts(loop_items); cc, nc = presence_counts(ctl_items)
ge, gc = group(ce), group(cc)
prior = ge + gc
lo = logodds(ge, gc, prior)
# first-token only
f_e = Counter(gkey(int(it[2][0])) for it in loop_items if len(it[2]))
f_c = Counter(gkey(int(it[2][0])) for it in ctl_items if len(it[2]))
lo_first = logodds(f_e, f_c, f_e + f_c)
# pre-window triggers
pe, _ = presence_counts(loop_items, 3); pc, _ = presence_counts(ctl_items, 3)
gpe, gpc = group(pe), group(pc)
lo_pre = logodds(gpe, gpc, gpe + gpc)
# sustain
gsl, gsc = group(sustain_loop), group(sustain_ctl)
lo_sus = logodds(gsl, gsc, gsl + gsc, a0=10000.0)
# TB action steps
ta_e, _ = presence_counts(sets['tb_action_loop']); ta_c, _ = presence_counts(sets['tb_action_ctl'])
lo_tba = logodds(group(ta_e), group(ta_c), group(ta_e) + group(ta_c))
# code redraft introducers vs control pre-windows
cd_e, _ = presence_counts(sets['code']); lo_code = logodds(group(cd_e), gpc, group(cd_e) + gpc)

# per-benchmark / per-arm
benches = ['tb', 'lcb', 'gpqa', 'aime', 'hmmt']
per_b = {}
for b in benches:
    e_, _ = presence_counts([x for x in loop_items if x[0] == b]); c_, _ = presence_counts([x for x in ctl_items if x[0] == b])
    ge_b, gc_b = group(e_), group(c_)
    per_b[b] = (logodds(ge_b, gc_b, prior), sum(1 for x in loop_items if x[0] == b), sum(1 for x in ctl_items if x[0] == b), ge_b, gc_b)
per_arm_entry = {}
for b in benches:
    for arm in ('base', 'swift'):
        e_, n_ = presence_counts([x for x in loop_items if x[0] == b and x[1] == arm])
        per_arm_entry[(b, arm)] = (group(e_), n_)

PUNCT = re.compile(r"^[\W_]*$")
# discourse filter: group must be alphabetic and occur (anywhere in entry/control windows) in >=2% of traces and in >=3 benchmarks
tdf = defaultdict(set); bdf = defaultdict(set)
for name in ('unit', 'topic', 'control'):
    for it in sets[name]:
        for which in (2, 3):
            w = it[which]
            if w is None: continue
            for x in set(int(y) for y in w):
                g = gkey(x); tdf[g].add(it[5]); bdf[g].add(it[0])
ntr = len({it[5] for it in sets['control']})
ALPHA = re.compile(r"^-?[a-z][a-z']+$")


def discourse(g):
    return bool(ALPHA.match(g)) and len(tdf[g]) >= 0.05 * ntr and len(bdf[g]) >= 4


def flags(g):
    ids = [i for i, _ in members[g].most_common()]
    return dict(ids=ids[:8], texts=[dec(i) for i in ids[:8]],
                in_pool=any(i in pool_ids for i in ids), in_candidates=any(i in cand_ids for i in ids),
                pool_ids=[i for i in ids if i in pool_ids], candidate_ids=[i for i in ids if i in cand_ids],
                candidate_tiers=sorted({cand_tier[i] for i in ids if i in cand_tier}),
                new=not any(i in pool_ids or i in cand_ids for i in ids))


def rank(lo_, top=80, min_count=30, skip_punct=True):
    rows = []
    for w, (z, d, ya, yb) in sorted(lo_.items(), key=lambda x: -x[1][0]):
        if ya < min_count: continue
        if skip_punct and (PUNCT.match(w) or w == ''): continue
        if skip_punct == 'discourse' and not discourse(w): continue
        rows.append((w, z, d, ya, yb))
        if len(rows) >= top: break
    return rows


entry_rank = []
for w, z, d, ya, yb in rank(lo, top=120, skip_punct='discourse'):
    r = dict(group=w, z=round(z, 2), log_odds=round(d, 3), n_loop_entries=ya, n_control=yb,
             rate_loop=round(ya / ne_, 4), rate_control=round(yb / nc, 4),
             first_token_z=round(lo_first[w][0], 2) if w in lo_first else None,
             first_token_n=f_e.get(w, 0))
    r.update(flags(w))
    pb = {}
    for b in benches:
        lb, nel, ncl, geb, gcb = per_b[b]
        if w in lb:
            pb[b] = dict(z=round(lb[w][0], 2), n_loop=lb[w][2], n_ctl=lb[w][3], rate_loop=round(lb[w][2] / max(nel, 1), 4), rate_ctl=round(lb[w][3] / max(ncl, 1), 4))
        arms = {}
        for arm in ('base', 'swift'):
            ga, na_ = per_arm_entry[(b, arm)]
            arms[arm] = dict(n=ga.get(w, 0), per_entry=round(ga.get(w, 0) / max(na_, 1), 4),
                             per10k_tokens=round(1e4 * ga.get(w, 0) / max(arm_tok[(b, arm)], 1), 3))
        pb.setdefault(b, {})['arms'] = arms
    r['per_benchmark'] = pb
    entry_rank.append(r)


def simple(lo_, top=60, min_count=30, filt='discourse'):
    out = []
    for w, z, d, ya, yb in rank(lo_, top=top, min_count=min_count, skip_punct=filt):
        r = dict(group=w, z=round(z, 2), n_loop=ya, n_ctl=yb); f = flags(w)
        r.update(ids=f['ids'][:5], in_pool=f['in_pool'], in_candidates=f['in_candidates'], new=f['new'], candidate_tiers=f['candidate_tiers'])
        out.append(r)
    return out


first_rank = simple(lo_first, top=60, min_count=15)
# word phrases at entry (first 1-4 words)
WR = re.compile(r"[A-Za-z']+|[^\sA-Za-z']")


def head_words(w):
    s = tok.decode([int(x) for x in w])
    return [x.lower() for x in WR.findall(s)][:6]


ph_e = Counter(); ph_c = Counter()
for items, C in ((loop_items, ph_e), (ctl_items, ph_c)):
    for it in items:
        hw = head_words(it[2])
        seen = set()
        for n in (1, 2, 3, 4):
            if len(hw) >= n: seen.add(' '.join(hw[:n]))
        C.update(seen)
lo_ph = logodds(ph_e, ph_c, ph_e + ph_c)
phr = [dict(phrase=w, z=round(z, 2), n_loop=ya, n_ctl=yb, rate_loop=round(ya / ne_, 4), rate_ctl=round(yb / nc, 4))
       for w, z, d, ya, yb in rank(lo_ph, top=200, min_count=25, skip_punct=False) if len(w.split()) >= 2 and all(discourse(x) or x in ('-', ',', ':', "'s", 'let', 's', 'i', 'a') for x in w.split())][:80]
# phrases counted per arm at entries (per 10k tokens)
ph_arm = defaultdict(Counter)
for it in loop_items:
    hw = head_words(it[2])
    for n in (2, 3, 4):
        if len(hw) >= n: ph_arm[(it[0], it[1])][' '.join(hw[:n])] += 1
for p in phr:
    p['per10k_tokens'] = {f'{b}/{a}': round(1e4 * ph_arm[(b, a)].get(p['phrase'], 0) / max(arm_tok[(b, a)], 1), 3) for b in benches for a in ('base', 'swift')}

known = {}
for gname in sorted({gkey(i) for i in pool_ids | cand_ids}):
    row = dict(in_pool=any(gkey(i) == gname for i in pool_ids), in_candidates=any(gkey(i) == gname for i in cand_ids))
    for nm, L_ in (('entry_window', lo), ('first_token', lo_first), ('pre_window', lo_pre), ('sustain', lo_sus), ('tb_action_step_opener', lo_tba)):
        if gname in L_:
            row[nm] = dict(z=round(L_[gname][0], 2), n_loop=L_[gname][2], n_ctl=L_[gname][3])
    arms = {}
    for b in benches:
        for arm in ('base', 'swift'):
            ga, na_ = per_arm_entry[(b, arm)]
            arms[f'{b}/{arm}'] = round(1e4 * ga.get(gname, 0) / max(arm_tok[(b, arm)], 1), 3)
    row['entry_windows_per10k_tokens'] = arms
    known[gname] = row
res = dict(
    known_pool_and_candidate_groups=known,
    description=__doc__,
    n_windows={k: len(v) for k, v in sets.items()},
    tokens_by_bench_arm={f'{b}/{a}': v for (b, a), v in arm_tok.items()},
    entry_markers=entry_rank,
    entry_first_token=first_rank,
    entry_phrases=phr,
    trigger_pre_window=simple(lo_pre, top=60, min_count=30),
    sustain_inside_loops=simple(lo_sus, top=60, min_count=200),
    tb_action_loop_step_openers=simple(lo_tba, top=40, min_count=15, filt=True),
    code_redraft_introducers=simple(lo_code, top=40, min_count=20, filt=True),
)
json.dump(res, open(f'{OUT}/entry_markers.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
pd.DataFrame(iter_rows).to_json(f'{OUT}/iterations.jsonl', orient='records', lines=True, force_ascii=False)
print('entry top:')
for r in entry_rank[:45]:
    print(f"{r['group']!r:18} z={r['z']:6.2f} loop={r['rate_loop']:.3f} ctl={r['rate_control']:.3f} pool={r['in_pool']} cand={r['in_candidates']} ids={r['ids'][:4]}")
print('first token:'); print([(r['group'], r['z'], r['n_loop'], r['n_ctl']) for r in first_rank[:30]])
print('phrases:'); print([(r['phrase'], r['z'], r['n_loop']) for r in phr[:40]])
print('pre:'); print([(r['group'], r['z']) for r in res['trigger_pre_window'][:30]])
print('sustain:'); print([(r['group'], r['z']) for r in res['sustain_inside_loops'][:40]])
print('tb action:'); print([(r['group'], r['z'], r['n_loop']) for r in res['tb_action_loop_step_openers'][:30]])
print('code intro:'); print([(r['group'], r['z']) for r in res['code_redraft_introducers'][:25]])
