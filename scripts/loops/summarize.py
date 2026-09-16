"""Aggregate loop statistics into runs/loops/summary.json (+ per-trace parquet)."""
import json, glob, os, re, pickle, math
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import zstandard
from scipy import stats

R = 'A:/swift/data/external/ukisai-evals'
OUT = 'A:/swift/runs/loops'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
ts = pd.read_json(f'{OUT}/trace_stats.jsonl', lines=True)
it = pd.read_json(f'{OUT}/iterations.jsonl', lines=True)
D = pickle.load(open(f'{OUT}/details.pkl', 'rb'))
T = pickle.load(open(f'{OUT}/traces_light.pkl', 'rb'))
tbt = pd.read_json(f'{OUT}/tb_trial_loops.jsonl', lines=True)
tbs = pd.read_json(f'{OUT}/tb_step_loops.jsonl', lines=True)
pool_ids = {x['id'] for x in json.load(open('A:/swift/runs/markers/ukisai_pool_recovered.json', encoding='utf-8'))}

# topic revisit tokens not overlapping unit loops (approx: window tokens * (1 - unit loop overlap))
topic_extra = {}
for k, d in D.items():
    units = d['units']
    loopmask_ranges = [(u[0], u[1]) for u in units if u[2]]
    tot = 0
    for a, b, v, sa, sb in d['topic_rev']:
        if v < 0.5: continue
        ov = sum(max(0, min(b, y) - max(a, x)) for x, y in loopmask_ranges)
        tot += max(0, (b - a) - ov)
    topic_extra[k] = tot
ts['topic_extra_tok'] = [topic_extra.get((r.bench, r.arm, r.tid), 0) for r in ts.itertuples()]
ts['any_loop_tok'] = ts.loop_tok + ts.topic_extra_tok
ts['any_loop_share'] = ts.any_loop_tok / ts.ntok
ts['loopy15'] = ts.any_loop_share >= 0.15
ts['has_loop'] = ts.n_iters > 0
ts['flipflop'] = ts.flip_returns >= 2
ts['degenerate'] = ts.degen_tok > 0

# pool-token density inside vs outside loop units
pin = Counter(); pout = Counter(); tin = Counter(); tout = Counter()
for t in T:
    k = (t['bench'], t['arm'], t['tid']); d = D.get(k)
    if d is None: continue
    ids = t['ids']; isp = np.isin(ids, list(pool_ids))
    m = np.zeros(len(ids), dtype=bool)
    for u in d['units']:
        if u[2] and not u[9]: m[u[0]:u[1]] = True
    code = np.zeros(len(ids), dtype=bool)
    for u in d['units']:
        if u[9]: code[u[0]:u[1]] = True
    key = (t['bench'], t['arm'])
    pin[key] += int(isp[m].sum()); tin[key] += int(m.sum())
    o = ~m & ~code
    pout[key] += int(isp[o].sum()); tout[key] += int(o.sum())
pool_density = {f'{b}/{a}': dict(per1k_inside_loops=round(1e3 * pin[(b, a)] / max(tin[(b, a)], 1), 2),
                                 per1k_outside=round(1e3 * pout[(b, a)] / max(tout[(b, a)], 1), 2)) for (b, a) in pin}

# ---------- all-sample totals (for length explained & paired analysis) ----------
allrows = []
for arm in ('base', 'swift'):
    for f in glob.glob(f'{R}/09-livecodebench-v6/{arm}/seed*/*.json'):
        x = json.load(open(f, encoding='utf-8'))
        allrows.append(dict(bench='lcb', arm=arm, key=f"s{x['seed']}/{x['question_id']}", pair=f"{x['question_id']}|{x['seed']}", group=str(x['question_id']),
                            tokens=int(x['completion_tokens']), truncated=x['finish_reason'] == 'length', difficulty=x.get('difficulty')))
for arm, dn in (('base', 'base_reference'), ('swift', 'swift')):
    for seed in range(5):
        for l in open(f'{R}/01-gpqa-diamond/adapter_only_dp4x2/{dn}/seed{seed}.raw.jsonl', encoding='utf-8'):
            r = json.loads(l)
            allrows.append(dict(bench='gpqa', arm=arm, key=f"s{seed}/{r['task_id']}", pair=f"{r['task_id']}|{seed}", group=r['task_id'],
                                tokens=int(r['reasoning_tokens']), truncated=r['finish_reason'] == 'length', correct=str(r['correct']) == 'True'))
ratio = {}
for t in T:
    if t['bench'] in ('aime', 'hmmt'):
        ratio.setdefault((t['bench'], t['arm']), []).append(t['ntok'] / max(len(t['text']), 1))
ntok_known = {(t['bench'], t['arm'], t['tid']): t['ntok'] for t in T}
for bench, sub, name in (('aime', '05-aime-2026', 'aime_2026'), ('hmmt', '06-hmmt-nov-2025', 'hmmt_nov_2025')):
    for f in glob.glob(f'{R}/{sub}/matharena_outputs/*/{name}/*_s*/*.json.zst'):
        tag = os.path.basename(os.path.dirname(f)); arm, seed = tag.split('_s')
        d = json.loads(zstandard.ZstdDecompressor().stream_reader(open(f, 'rb')).read())
        cot = ''.join(m.get('content') or '' for m in d['messages'][0] if isinstance(m, dict) and m.get('type') == 'cot')
        tid = f"s{seed}/{d['idx']}"
        ntk = ntok_known.get((bench, arm, tid), int(len(cot) * np.median(ratio[(bench, arm)])))
        allrows.append(dict(bench=bench, arm=arm, key=tid, pair=f"{d['idx']}|{seed}", group=str(d['idx']), tokens=ntk, correct=bool(d['correct'][0])))
csv = pd.read_csv(f'{R}/08-terminal-bench-2.1/swift/analysis/tb21_per_trial.csv')
csv['arm2'] = np.where(csv.arm.str.contains('swift'), 'swift', 'base')
AR = pd.DataFrame(allrows)
tbstep_loop = tbs[tbs.kind.isin(['retry_same_error', 'reread_same_output', 'json_format_fail', 'wait_empty', 'poll'])].groupby(['arm', 'trial']).ct.sum()

summary = dict(notes={}, per_benchmark={}, tb={}, categories={}, pool_token_density=pool_density)
summary['notes']['definitions'] = {
    'unit': 'paragraph (blank-line split; >1200-char blocks split on newlines/sentences)',
    'loop_unit': 'prose/code unit that (a) near-duplicates an earlier unit (Jaccard>=0.5 on number-normalised word 4-gram shingles; adjacent low-raw-overlap matches = enumeration progression, excluded) or (b) is >=50% redundant (>=50% of its shingles already written earlier in the trace, contentful or exact-number shingles)',
    'iteration': 'maximal run of loop units (1 short gap allowed); iteration start = loop entry',
    'topic_revisit': 'window (~300 tok) with tf-idf cosine >=0.5 to a non-adjacent earlier window (paraphrased revisit of same sub-problem); only non-overlapping tokens added to any_loop',
    'loop_share': 'tokens in loop units / trace tokens; any_loop_share adds topic-revisit tokens',
    'degenerate': 'back-to-back exact 32-gram repetition with period<=512 covering >=2 periods',
    'flipflop': 'gpqa/math: committed answer sequence (non-hypothetical "So X."/"answer is X"/boxed) returns to a previously abandoned answer >=2 times',
    'tb_action_loop': 'trial with >=3 futile repeats (same command + same error or same output), or JSON-format-failure run>=2 / >=4 total, or wait/poll run>=4',
    'long_trace': '>=8000 reasoning tokens (TB: per agent call)'}

for b in ['tb', 'lcb', 'gpqa', 'aime', 'hmmt']:
    s = ts[ts.bench == b]
    out = {}
    for arm in ('base', 'swift'):
        a = s[s.arm == arm]
        ar = AR[(AR.bench == b) & (AR.arm == arm)] if b != 'tb' else None
        n_all = len(ar) if ar is not None else int((csv.arm2 == arm).sum())
        tot_all = float(ar.tokens.sum()) if ar is not None else float(csv[csv.arm2 == arm].reasoning_tokens.sum())
        cat_tok = it[(it.bench == b) & (it.arm == arm)].groupby('cat').loop_tok.sum()
        o = dict(n_samples_all=n_all, mean_tokens_all=round(tot_all / max(n_all, 1), 1), n_long=len(a),
                 frac_long=round(len(a) / max(n_all, 1), 4) if b != 'tb' else None,
                 mean_ntok_long=round(a.ntok.mean(), 1),
                 has_iteration=round(a.has_loop.mean(), 4), loopy_ge15pct=round(a.loopy15.mean(), 4),
                 loop_share_mean=round(a.loop_share.mean(), 4), loop_share_prose_mean=round(a.loop_share_prose.mean(), 4),
                 near_dup_share_mean=round(a.nd_share.mean(), 4), topic_revisit_extra_share=round((a.topic_extra_tok / a.ntok).mean(), 4),
                 any_loop_share_mean=round(a.any_loop_share.mean(), 4), any_loop_share_tokw=round(a.any_loop_tok.sum() / a.ntok.sum(), 4),
                 iters_per_10k=round(1e4 * a.n_iters.sum() / a.ntok.sum(), 3), max_visits_mean=round(a.max_visits.mean(), 2),
                 template_opener_share=round(a.tmpl_share.mean(), 4), exact_32gram_rep_share=round(a.exact_rep_share.mean(), 4),
                 degenerate_traces=int(a.degenerate.sum()), degenerate_tok=int(a.degen_tok.sum()),
                 flipflop_rate=round(a.flipflop.mean(), 4) if b in ('gpqa', 'aime', 'hmmt') else None,
                 loop_tokens_per_sample_all=round(a.any_loop_tok.sum() / max(n_all, 1), 1),
                 loop_tokens_by_category_per_sample={c: round(v / max(n_all, 1), 1) for c, v in cat_tok.items()})
        if a.truncated.notna().any():
            o['truncated_rate_long'] = round(a.truncated.mean(), 4)
            o['any_loop_share_truncated'] = round(a[a.truncated == True].any_loop_share.mean(), 4)
            o['any_loop_share_not_truncated'] = round(a[a.truncated == False].any_loop_share.mean(), 4)
        if a.correct.notna().any():
            o['any_loop_share_correct'] = round(a[a.correct == True].any_loop_share.mean(), 4)
            o['any_loop_share_incorrect'] = round(a[a.correct == False].any_loop_share.mean(), 4)
            o['acc_long_loopy'] = round(a[a.loopy15].correct.mean(), 4) if a.loopy15.any() else None
            o['acc_long_not_loopy'] = round(a[~a.loopy15].correct.mean(), 4)
            if b in ('gpqa', 'aime', 'hmmt'):
                o['acc_flipflop'] = round(a[a.flipflop].correct.mean(), 4) if a.flipflop.any() else None
                o['acc_no_flipflop'] = round(a[~a.flipflop].correct.mean(), 4)
        out[arm] = o
    ob, osw = out['base'], out['swift']
    dl = ob['mean_tokens_all'] - osw['mean_tokens_all']
    dloop = ob['loop_tokens_per_sample_all'] - osw['loop_tokens_per_sample_all']
    out['length_explained'] = dict(mean_len_diff_base_minus_swift=round(dl, 1), loop_tok_diff=round(dloop, 1),
                                   frac_of_length_diff_explained_by_loops=round(dloop / dl, 3) if abs(dl) > 1 else None)
    # rank tests on long traces
    out['mannwhitney_any_loop_share_p'] = float(stats.mannwhitneyu(s[s.arm == 'base'].any_loop_share, s[s.arm == 'swift'].any_loop_share).pvalue)
    # paired (non-TB): pair key; short traces count 0 loop tokens
    if b != 'tb':
        ar = AR[AR.bench == b].copy()
        lt = {(r.arm, r.tid): r.any_loop_tok for r in s.itertuples()}
        ar['loop_tok'] = [lt.get((r.arm, r.key), 0) for r in ar.itertuples()]
        pv = ar.pivot_table(index='pair', columns='arm', values=['loop_tok', 'tokens'])
        pv = pv.dropna()
        dlt = pv[('loop_tok', 'base')] - pv[('loop_tok', 'swift')]
        dtk = pv[('tokens', 'base')] - pv[('tokens', 'swift')]
        out['paired'] = dict(n_pairs=len(pv), mean_loop_tok_diff=round(dlt.mean(), 1), mean_token_diff=round(dtk.mean(), 1),
                             wilcoxon_loop_tok_p=float(stats.wilcoxon(dlt).pvalue) if (dlt != 0).sum() > 10 else None)
        if b == 'lcb':
            dd = {}
            for diff in ('easy', 'medium', 'hard'):
                sub = ar[ar.difficulty == diff]
                row = {}
                for arm in ('base', 'swift'):
                    sa = sub[sub.arm == arm]
                    row[arm] = dict(n=len(sa), mean_tokens=round(sa.tokens.mean(), 1), truncated=round(sa.truncated.mean(), 4),
                                    long_frac=round((sa.tokens >= 8000).mean(), 4), loop_tok_per_sample=round(sa.loop_tok.mean(), 1),
                                    any_loop_share_long=round(s[(s.arm == arm) & (s.difficulty == diff)].any_loop_share.mean(), 4))
                dd[diff] = row
            out['by_difficulty'] = dd
    summary['per_benchmark'][b] = out

# ---------- TB extras ----------
tbt['timeout'] = tbt.exc == 'AgentTimeoutError'; tbt['solved'] = tbt.reward == 1.0
calls = ts[ts.bench == 'tb'].copy()
calls['trial'] = calls.tid.str.split('#').str[0]
call_loop = calls.groupby(['arm', 'trial']).agg(call_loop_tok=('any_loop_tok', 'sum'), n_long_calls=('ntok', 'size'), max_call_share=('any_loop_share', 'max')).reset_index()
tbt = tbt.merge(call_loop, on=['arm', 'trial'], how='left').fillna({'call_loop_tok': 0, 'n_long_calls': 0, 'max_call_share': 0})
tbt['within_call_loopy'] = tbt.max_call_share >= 0.15
tbt['action_loop_tok'] = [int(tbstep_loop.get((r.arm, r.trial), 0)) for r in tbt.itertuples()]
tb = {}
for arm in ('base', 'swift'):
    a = tbt[tbt.arm == arm]
    tb[arm] = dict(n_trials=len(a), solved=round(a.solved.mean(), 4), timeout=round(a.timeout.mean(), 4),
                   trials_with_long_call=round((a.n_long_calls > 0).mean(), 4),
                   within_call_loopy_trials=round(a.within_call_loopy.mean(), 4),
                   action_loop_trials=round(a.action_loop.mean(), 4), loop_futile=round(a.loop_futile.mean(), 4), loop_json=round(a.loop_json.mean(), 4),
                   loop_wait=round(a.loop_wait.mean(), 4), loop_complete_repeat=round(a.loop_complete.mean(), 4),
                   json_fail_steps_per_trial=round(a.n_json_fail.mean(), 3), futile_repeats_per_trial=round(a.n_futile.mean(), 3),
                   wait_steps_per_trial=round(a.n_wait_empty.mean(), 3), poll_steps_per_trial=round(a.n_poll.mean(), 3),
                   page_steps_per_trial=round(a.n_page.mean(), 3), rerun_steps_per_trial=round(a.n_rerun.mean(), 3),
                   rewrite_same_file_per_trial=round(a.n_rewrite_same_file.mean(), 3),
                   cross_call_reasoning_redundancy_tokw=round(a.cross_red_tokw.mean(), 4),
                   within_call_loop_tok_per_trial=round(a.call_loop_tok.mean(), 1), action_loop_step_ct_per_trial=round(a.action_loop_tok.mean(), 1),
                   total_ct_per_trial=round(a.total_ct.mean(), 1))
    for flag in ('action_loop', 'loop_json', 'loop_futile', 'loop_wait', 'within_call_loopy'):
        g = a.groupby(flag)[['solved', 'timeout']].mean().round(4)
        tb[arm][f'outcome_by_{flag}'] = {str(k): v for k, v in g.to_dict(orient='index').items()}
# JSON fail trigger: completion tokens of the call that produced the malformed output
tbs_sorted = tbs.sort_values(['trial', 'arm', 'step'])
tb['json_fail_call_ct_median'] = {arm: float(tbs[(tbs.arm == arm) & tbs.json_fail].ct.median()) for arm in ('base', 'swift')}
tb['other_call_ct_median'] = {arm: float(tbs[(tbs.arm == arm) & ~tbs.json_fail].ct.median()) for arm in ('base', 'swift')}
tb['json_fail_rate_by_call_ct'] = {arm: tbs[tbs.arm == arm].groupby(pd.cut(tbs[tbs.arm == arm].ct, [0, 1000, 4000, 8000, 16000, 1e9])).json_fail.mean().round(4).astype(float).to_dict() for arm in ('base', 'swift')}
tb['json_fail_rate_by_call_ct'] = {a: {str(k): v for k, v in d.items()} for a, d in tb['json_fail_rate_by_call_ct'].items()}
# within-call loops vs previous observation error
ERR = re.compile(r"error|failed|traceback|no such file|not found|permission denied|segmentation fault|timed out|cannot|exception|fatal|killed", re.I)
prev_err = {(t['arm'], t['tid']): bool(ERR.search(t.get('prev_obs', '') or '')) for t in T if t['bench'] == 'tb'}
calls['prev_err'] = [prev_err.get((r.arm, r.tid), False) for r in calls.itertuples()]
tb['within_call_loop_share_by_prev_obs_error'] = {arm: calls[calls.arm == arm].groupby('prev_err').any_loop_share.mean().round(4).to_dict() for arm in ('base', 'swift')}
tb['within_call_loop_share_by_prev_obs_error'] = {a: {str(k): v for k, v in d.items()} for a, d in tb['within_call_loop_share_by_prev_obs_error'].items()}
# task-level pairing
tp = tbt.groupby(['task', 'arm']).agg(action_loop=('action_loop', 'mean'), solved=('solved', 'mean'), call_loop_tok=('call_loop_tok', 'mean'), total_ct=('total_ct', 'mean')).unstack()
d_al = (tp[('action_loop', 'base')] - tp[('action_loop', 'swift')]).dropna()
d_cl = (tp[('call_loop_tok', 'base')] - tp[('call_loop_tok', 'swift')]).dropna()
tb['task_paired'] = dict(n_tasks=len(tp), action_loop_rate_diff_base_minus_swift=round(d_al.mean(), 4),
                         wilcoxon_action_loop_p=float(stats.wilcoxon(d_al).pvalue) if (d_al != 0).sum() > 10 else None,
                         within_call_loop_tok_diff=round(d_cl.mean(), 1), wilcoxon_call_loop_tok_p=float(stats.wilcoxon(d_cl).pvalue) if (d_cl != 0).sum() > 10 else None,
                         total_ct_diff=round((tp[('total_ct', 'base')] - tp[('total_ct', 'swift')]).mean(), 1))
summary['tb'] = tb

# ---------- categories ----------
cats = {}
for b in ['tb', 'lcb', 'gpqa', 'aime', 'hmmt', 'all']:
    s_ = it if b == 'all' else it[it.bench == b]
    row = {}
    for arm in ('base', 'swift'):
        a = s_[s_.arm == arm]
        nt = ts[(ts.arm == arm) & ((ts.bench == b) if b != 'all' else True)].ntok.sum()
        c = a.groupby('cat').agg(n=('loop_tok', 'size'), loop_tok=('loop_tok', 'sum'))
        row[arm] = {k: dict(iters_per_100k_long_tokens=round(1e5 * v['n'] / nt, 2), loop_tok_per_1k_long_tokens=round(1e3 * v['loop_tok'] / nt, 2))
                    for k, v in c.to_dict(orient='index').items()}
    cats[b] = row
summary['categories'] = cats
# swift/base ratios per category for all benches (per long token)
rat = {}
for b, row in cats.items():
    rat[b] = {c: round(row['swift'].get(c, {}).get('loop_tok_per_1k_long_tokens', 0) / max(row['base'].get(c, {}).get('loop_tok_per_1k_long_tokens', 1e-9), 1e-9), 3)
              for c in set(row['base']) | set(row['swift'])}
summary['category_swift_over_base_loop_tok_ratio'] = rat
json.dump(summary, open(f'{OUT}/summary.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=float)
ts.drop(columns=['top_openers']).to_parquet(f'{OUT}/trace_loop_stats.parquet')
tbt.to_parquet(f'{OUT}/tb_trial_loop_stats.parquet')
print(json.dumps(summary, indent=1, default=float)[:30000])
