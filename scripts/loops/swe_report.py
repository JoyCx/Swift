"""Open-SWE-Traces: outcome association, categories and loop-entry markers (incl. markers predicting non-resolution)."""
import json, math, pickle, re, os
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from tokenizers import Tokenizer

OUT = 'A:/swift/runs/loops/swe'
tok = Tokenizer.from_file('A:/models/Qwen3.8-27B/tokenizer.json')
ts = pd.read_json(f'{OUT}/trace_stats.jsonl', lines=True)
D = pickle.load(open(f'{OUT}/details.pkl', 'rb'))
T = {t['tid']: t for t in pickle.load(open(f'{OUT}/traces_light.pkl', 'rb'))}
pool_ids = {x['id'] for x in json.load(open('A:/swift/runs/markers/ukisai_pool_recovered.json', encoding='utf-8'))}
cand_ids = set(); cand_tier = {}
for tier, v in json.load(open('A:/swift/runs/markers/residual/curated_tiers.json', encoding='utf-8')).items():
    for x in v: cand_ids.add(x['id']); cand_tier.setdefault(x['id'], tier)
for x in json.load(open('A:/swift/runs/markers/residual/pool_variant_gaps.json', encoding='utf-8')):
    cand_ids.add(x['id']); cand_tier.setdefault(x['id'], 'pool_variant_gaps')

CAT_RULES = [
    ('format_replan', re.compile(r"\b(final (answer|response|code|output|message|json)|format|json|delimiter|code block|concise|boxed|ensure (the )?(final|output|format|response)|provide (the )?(final|answer|explanation|code)|explanation|keystrokes|task_complete|respond|response|submit\w*|COMPLETE_TASK)\b", re.I)),
    ('verification', re.compile(r"\b(verif\w*|check\w*|double|re-?check|confirm|test\w*|sanity|simulat\w*|example|recomput\w*|re-?comput\w*|trace|validat\w*|make sure|let'?s see if|edge case|potential (issue|bug|pitfall|problem)|hidden|careful)\b", re.I)),
    ('alternative_approach', re.compile(r"\b(alternativ\w*|another (approach|way|method|idea)|instead|different (approach|way|method)|other approach|what if|could we|could use|we could|maybe we can|consider using|or maybe|option|micro-?optim\w*|optimi[sz]\w*|faster|speed)\b", re.I)),
    ('hedge_reconsider', re.compile(r"\b(wait|hmm|actually|but|however|hold on|reconsider|maybe|perhaps|unless|not sure)\b", re.I)),
    ('restate_problem', re.compile(r"\b(problem|statement|task|question|requirement|says|constraint|given|asks|user|pr description|issue)\b", re.I)),
]


def categorise(head, code, exact_frac, dist_units):
    if exact_frac >= 0.9 and dist_units <= 2: return 'degenerate_repeat'
    if code >= 0.5: return 'code_redraft'
    for n, rx in CAT_RULES:
        if rx.search(head): return n
    return 'rederivation_other'


# ---------- iterations ----------
rows = []
for r in ts.itertuples():
    t = T[r.tid]; d = D[r.tid]; st = t['starts']; L = len(st)
    for it in d['iters']:
        a = int(st[it['ta']]) if it['ta'] < L else len(t['text'])
        head = ' '.join(t['text'][a:a + 300].split()[:40])
        rows.append(dict(tid=r.tid, bench='swe', arm='qwen38', ta=it['ta'], tb=it['tb'], tok=it['tok'], loop_tok=it['loop_tok'], src_ta=it['src_ta'], src_tb=it['src_tb'],
                         dist_units=it['dist_units'], jac=it['jac'], red=it.get('red', 0), code=it['code'], visits=it['visits'], exact_frac=it['exact_frac'],
                         step=it['step'], src_step=it['src_step'], cross_step=it['cross_step'], step_gap=it['step_gap'],
                         cat=categorise(head, it['code'], it['exact_frac'], it['dist_units']), head=' '.join(head.split()[:25]), correct=r.correct, ntok=r.ntok))
it = pd.DataFrame(rows)
it.to_json(f'{OUT}/iterations.jsonl', orient='records', lines=True, force_ascii=False)

ts['any_loop_tok'] = ts.loop_tok + ts.topic_extra_tok
ts['any_loop_share'] = ts.any_loop_tok / ts.ntok
ts['loopy15'] = ts.any_loop_share >= 0.15
ts['resolved'] = ts.correct.astype(int)
ts['len_decile'] = pd.qcut(ts.ntok, 10, labels=False, duplicates='drop')
ts['log_ntok'] = np.log(ts.ntok)
ts['category'] = ts.difficulty.fillna('unknown')
ts['cross_share'] = ts.cross_step_loop_tok / ts.ntok
ts['within_share'] = ts.within_step_loop_tok / ts.ntok
summ = {}
summ['n_trajectories'] = len(ts)
summ['sampling'] = 'stratified proportional subsample (~7000 per dataset, same fraction within resolved=0/1), seed 0; trajectories with <500 thinking tokens dropped'
summ['resolved_rate'] = round(ts.resolved.mean(), 4)
summ['thinking_tokens'] = dict(median=float(ts.ntok.median()), p90=float(ts.ntok.quantile(.9)), frac_ge8k=round((ts.ntok >= 8000).mean(), 4))
g = ts.groupby('resolved')
summ['by_resolved'] = {str(k): dict(n=len(v), ntok_median=float(v.ntok.median()), n_steps_median=float(v.n_steps.median()),
                                    any_loop_share=round(v.any_loop_share.mean(), 4), loop_share=round(v.loop_share.mean(), 4),
                                    near_dup_share=round(v.nd_share.mean(), 4), cross_step_share=round(v.cross_share.mean(), 4), within_step_share=round(v.within_share.mean(), 4),
                                    loopy_ge15pct=round(v.loopy15.mean(), 4), iters_per_10k=round(1e4 * v.n_iters.sum() / v.ntok.sum(), 3),
                                    action_loop=round(v.action_loop.mean(), 4), futile_repeats=round(v.n_futile.mean(), 3),
                                    same_fail_run_ge3=round((v.max_same_fail_run >= 3).mean(), 4), edit_test_fail=round(v.n_edit_test_fail.mean(), 3),
                                    format_errors=round(v.n_format_error.mean(), 3), polls=round(v.n_poll.mean(), 3), pages=round(v.n_page.mean(), 3),
                                    degenerate_traces=int((v.degen_tok > 0).sum()), template_opener_share=round(v.tmpl_share.mean(), 4))
                     for k, v in g}
dec = ts.groupby('len_decile').agg(ntok_lo=('ntok', 'min'), ntok_hi=('ntok', 'max'), n=('ntok', 'size'), resolved=('resolved', 'mean'),
                                     any_loop_share=('any_loop_share', 'mean'), action_loop=('action_loop', 'mean'))
dec2 = ts.groupby(['len_decile', 'resolved']).agg(any_loop_share=('any_loop_share', 'mean'), action_loop=('action_loop', 'mean'), same_fail3=('max_same_fail_run', lambda x: (x >= 3).mean())).unstack()
summ['by_length_decile'] = {int(i): dict(ntok_range=[int(r.ntok_lo), int(r.ntok_hi)], n=int(r.n), resolved=round(r.resolved, 4), any_loop_share=round(r.any_loop_share, 4), action_loop=round(r.action_loop, 4),
                                         any_loop_share_resolved=round(dec2.loc[i, ('any_loop_share', 1)], 4), any_loop_share_unresolved=round(dec2.loc[i, ('any_loop_share', 0)], 4),
                                         action_loop_resolved=round(dec2.loc[i, ('action_loop', 1)], 4), action_loop_unresolved=round(dec2.loc[i, ('action_loop', 0)], 4))
                            for i, r in dec.iterrows()}
# loop-share quintile vs resolution within decile (stratified)
ts['loop_q'] = ts.groupby('len_decile').any_loop_share.transform(lambda x: pd.qcut(x.rank(method='first'), 5, labels=False))
summ['resolved_by_within_decile_loop_quintile'] = ts.groupby('loop_q').resolved.mean().round(4).to_dict()
models = {}
for name, f in {
    'loop_share': 'resolved ~ any_loop_share + log_ntok + np.log(n_steps) + C(dataset) + C(category)',
    'loop_components': 'resolved ~ within_share + cross_share + I(topic_extra_tok/ntok) + log_ntok + np.log(n_steps) + C(dataset) + C(category)',
    'action_loops': 'resolved ~ I(max_same_fail_run>=3) + I(n_futile>=3) + I(n_format_error>0) + any_loop_share + log_ntok + np.log(n_steps) + C(dataset) + C(category)',
}.items():
    try:
        m = smf.logit(f, data=ts).fit(disp=0)
        models[name] = {k: dict(coef=round(m.params[k], 4), p=float(m.pvalues[k]), odds_ratio=round(math.exp(m.params[k]), 4)) for k in m.params.index if not k.startswith('C(') and k != 'Intercept'}
    except Exception as e:
        models[name] = str(e)
summ['logit_resolved'] = models
summ['logit_note'] = 'any_loop_share coefficient is per unit share (0->1); OR for +0.10 share = exp(coef*0.1)'
# categories
cat = it.groupby(['correct', 'cat']).loop_tok.sum().unstack(0).fillna(0)
ntok_by = ts.groupby('correct').ntok.sum()
summ['categories_loop_tok_per_1k_thinking_tokens'] = {c: {('resolved' if k else 'unresolved'): round(1e3 * cat.loc[c, k] / ntok_by[k], 3) for k in cat.columns} for c in cat.index}
summ['cross_step_fraction_of_loop_tokens'] = round(it[it.cross_step].loop_tok.sum() / max(it.loop_tok.sum(), 1), 4)
summ['step_gap_of_cross_step_iterations'] = it[it.cross_step].step_gap.describe().round(2).to_dict()

# ---------- entry markers ----------
dec_cache = {}


def dec(i):
    s = dec_cache.get(i)
    if s is None: s = tok.decode([int(i)]); dec_cache[i] = s
    return s


def gkey(i): return dec(i).strip().lower()


rng = np.random.default_rng(0)
NE, PRE = 12, 32
CODEISH = re.compile(r"[`{}();<>=\[\]#/\|$]|^\s*[\"'*\-\d]|https?:|(def|func|const|class|import|return|var|let \w+ =|export|public|static|struct|fn|self)")


def prose_ok(w):
    s_ = tok.decode([int(x) for x in w])
    return not CODEISH.search(s_)
loop_w = []; ctl_w = []
for r in ts.itertuples():
    t = T[r.tid]; d = D[r.tid]; ids = t['ids']
    starts = [x['ta'] for x in d['iters'] if x['loop_tok'] >= 30 and x['code'] < 0.5]
    tstarts = {a for a, b, v, sa, sb in d['topic_rev'] if v >= 0.5}
    loopunit = {u[0] for u in d['units'] if u[2]}
    extra = [a for a in tstarts if a not in loopunit and a not in starts]
    ents = starts + extra
    for a in ents:
        if prose_ok(ids[a:a + NE]): loop_w.append((r.tid, r.resolved, ids[a:a + NE], ids[max(0, a - PRE):a]))
    ctl = []; prev = False
    for u in d['units']:
        ta, tb_, isl, ist, mj, m, bj, bm, prog, isc, rawj, red, rsrc = u
        if not isl and not prog and not isc and not prev and ta not in tstarts and tb_ - ta >= 8: ctl.append(ta)
        prev = isl
    if ents and ctl:
        for j in rng.choice(len(ctl), size=min(len(ctl), 3 * len(ents)), replace=False):
            a = ctl[j]
            if prose_ok(ids[a:a + NE]): ctl_w.append((r.tid, r.resolved, ids[a:a + NE], ids[max(0, a - PRE):a]))


def pres(ws, which=2):
    c = Counter()
    for w in ws: c.update({gkey(int(x)) for x in w[which]})
    return c


def logodds(ca, cb, prior, a0=1000.0):
    na = sum(ca.values()); nb = sum(cb.values()); npr = sum(prior.values()); out = {}
    for w in set(ca) | set(cb):
        al = a0 * prior.get(w, 0) / npr + 0.01; ya = ca.get(w, 0); yb = cb.get(w, 0)
        dd = math.log((ya + al) / (na + a0 - ya - al)) - math.log((yb + al) / (nb + a0 - yb - al))
        out[w] = (dd / math.sqrt(1 / (ya + al) + 1 / (yb + al)), ya, yb)
    return out


members = defaultdict(Counter)
tdf = defaultdict(set)
for w in loop_w + ctl_w:
    for x in w[2]:
        members[gkey(int(x))][int(x)] += 1; tdf[gkey(int(x))].add(w[0])
ntr = len({w[0] for w in ctl_w})
ALPHA = re.compile(r"^-?[a-z][a-z']+$")


def disc(g): return bool(ALPHA.match(g)) and len(tdf[g]) >= 0.05 * ntr


def flags(g):
    ids = [i for i, _ in members[g].most_common(8)]
    return dict(ids=ids, texts=[dec(i) for i in ids], in_pool=any(i in pool_ids for i in ids), in_candidates=any(i in cand_ids for i in ids),
                candidate_tiers=sorted({cand_tier[i] for i in ids if i in cand_tier}), new=not any(i in pool_ids or i in cand_ids for i in ids))


ce, cc = pres(loop_w), pres(ctl_w)
lo = logodds(ce, cc, ce + cc)
ent = []
for w, (z, ya, yb) in sorted(lo.items(), key=lambda x: -x[1][0]):
    if ya < 40 or not disc(w): continue
    r = dict(group=w, z=round(z, 2), n_loop=ya, n_ctl=yb, rate_loop=round(ya / len(loop_w), 4), rate_ctl=round(yb / len(ctl_w), 4)); r.update(flags(w)); ent.append(r)
    if len(ent) >= 80: break
# first token
fe = Counter(gkey(int(w[2][0])) for w in loop_w if len(w[2])); fc = Counter(gkey(int(w[2][0])) for w in ctl_w if len(w[2]))
lof = logodds(fe, fc, fe + fc)
first = []
for w, (z, ya, yb) in sorted(lof.items(), key=lambda x: -x[1][0]):
    if ya < 20 or not disc(w): continue
    r = dict(group=w, z=round(z, 2), n_loop=ya, n_ctl=yb); r.update(flags(w)); first.append(r)
    if len(first) >= 40: break
# phrases
WR = re.compile(r"[A-Za-z']+|[^\sA-Za-z']")


def hw(w): return [x.lower() for x in WR.findall(tok.decode([int(x) for x in w]))][:6]


pe = Counter(); pc = Counter()
for ws, CC_ in ((loop_w, pe), (ctl_w, pc)):
    for w in ws:
        h = hw(w[2]); CC_.update({' '.join(h[:n]) for n in (2, 3, 4) if len(h) >= n})
lop = logodds(pe, pc, pe + pc)
phr = [dict(phrase=w, z=round(z, 2), n_loop=ya, n_ctl=yb) for w, (z, ya, yb) in sorted(lop.items(), key=lambda x: -x[1][0]) if ya >= 30][:50]
# trigger pre-window
pre_e, pre_c = pres(loop_w, 3), pres(ctl_w, 3)
lopre = logodds(pre_e, pre_c, pre_e + pre_c)
pre = [dict(group=w, z=round(z, 2), n_loop=ya, n_ctl=yb) for w, (z, ya, yb) in sorted(lopre.items(), key=lambda x: -x[1][0]) if ya >= 40 and disc(w)][:40]
# markers at loop entries predicting non-resolution: entry windows from unresolved vs resolved trajectories
ue = pres([w for w in loop_w if w[1] == 0]); re_ = pres([w for w in loop_w if w[1] == 1])
lou = logodds(ue, re_, ue + re_)
unres = []
for w, (z, ya, yb) in sorted(lou.items(), key=lambda x: -x[1][0]):
    if ya + yb < 60 or not disc(w): continue
    r = dict(group=w, z_unresolved_vs_resolved=round(z, 2), n_unresolved_entries=ya, n_resolved_entries=yb); r.update(flags(w)); unres.append(r)
    if len(unres) >= 40: break
# trajectory-level: count of loop entries containing marker per 10k thinking tokens -> logit controlling for length & dataset
top_groups = [e['group'] for e in ent[:40]]
per_traj = defaultdict(Counter)
for w in loop_w:
    per_traj[w[0]].update({gkey(int(x)) for x in w[2]} & set(top_groups))
traj_assoc = []
for gname in top_groups:
    ts['m'] = [1e4 * per_traj[t][gname] / n for t, n in zip(ts.tid, ts.ntok)]
    try:
        m = smf.logit('resolved ~ m + log_ntok + np.log(n_steps) + C(dataset) + C(category)', data=ts).fit(disp=0)
        traj_assoc.append(dict(group=gname, coef_per_entry_per_10k=round(m.params['m'], 4), p=float(m.pvalues['m']), mean_rate=round(ts.m.mean(), 4)))
    except Exception as e:
        traj_assoc.append(dict(group=gname, error=str(e)))
ts = ts.drop(columns=['m'])
res = dict(summary=summ, entry_markers=ent, entry_first_token=first, entry_phrases=phr, trigger_pre_window=pre,
           entry_markers_predicting_unresolved=unres, trajectory_level_marker_logit=sorted(traj_assoc, key=lambda x: x.get('coef_per_entry_per_10k', 0)),
           n_windows=dict(loop=len(loop_w), control=len(ctl_w)))
json.dump(res, open(f'{OUT}/swe_summary.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=float)
ts.drop(columns=['top_openers'], errors='ignore').to_parquet(f'{OUT}/trace_loop_stats.parquet')
print(json.dumps(summ, indent=1, default=float)[:12000])
print('ENTRY', [(e['group'], e['z'], e['in_pool'], e['in_candidates']) for e in ent[:40]])
print('FIRST', [(e['group'], e['z']) for e in first[:25]])
print('PHR', [(e['phrase'], e['z'], e['n_loop']) for e in phr[:30]])
print('PRE', [(e['group'], e['z']) for e in pre[:25]])
print('UNRES', [(e['group'], e['z_unresolved_vs_resolved']) for e in unres[:25]])
print('TRAJ', [(e['group'], e.get('coef_per_entry_per_10k'), round(e.get('p', 1), 4)) for e in res['trajectory_level_marker_logit'][:40]])
