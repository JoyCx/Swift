"""Loop analysis on Open-SWE-Traces (Qwen3.8-27B, mini-swe-agent).

1. Reasoning loops on the trajectory-level thinking trace (all steps concatenated; each step starts a new paragraph),
   using detect.analyse (same detector as UkisAI data). Each iteration is tagged within_step / cross_step.
2. Cross-step action loops: repeated commands (same command + same error / same output), polling, paging,
   edit->test->fail cycles with an unchanged failure signature, tool-call format errors.
3. Outcome association (resolved) controlling for thinking-length decile, dataset, category (logit).
4. Loop-entry markers (matched within-trajectory controls) + markers whose presence at loop entries predicts non-resolution.
"""
import os, re, json, math, pickle, sys
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect

OUT = 'A:/swift/runs/loops/swe'
NUM = re.compile(r"0x[0-9a-f]+|\d+(?:\.\d+)?")
WRE = re.compile(r"[a-z_][a-z_0-9]*|#|[^\sa-z_0-9#]", re.I)
ERRL = re.compile(r"error|failed|failure|traceback|no such file|not found|permission denied|segmentation fault|timed out|cannot|could not|exception|fatal|assert", re.I)
TEST = re.compile(r"pytest|python[0-9.]* -m (pytest|unittest)|\btox\b|go test|cargo test|npm (run )?test|yarn test|jest|mvn .*test|gradle.*test|make test|ctest|phpunit|rspec|bundle exec|mix test|dotnet test|python[0-9.]* [^|;&]*test[^|;&]*\.py|reproduce|repro", re.I)
EDIT = re.compile(r"sed -i|cat\s*>|cat\s*<<.*>|tee\s|python[0-9.]* - <<|python[0-9.]* -c .*(write|open\(.+['\"]w)|perl -pi|git apply|patch\s+-p|>\s*\S+\.(py|js|ts|go|rs|java|rb|c|cpp|h)\b|str_replace|ed -s", re.I)
POLL = re.compile(r"^\s*(sleep\b|tail\b[^|]*\.log|ps\b|pgrep\b|jobs\b|wait\b)", re.M)
PAGE = re.compile(r"^\s*(cd \S+ && )?(sed -n|head\b|tail\b|cat\b|nl\b|awk 'NR|grep -n)", re.M)


def shingle(s, n=3):
    w = WRE.findall(s)
    if len(w) < n: return {' '.join(w)} if w else set()
    return {' '.join(w[i:i+n]) for i in range(len(w) - n + 1)}


def jac(a, b):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)


def err_sig(out):
    lines = [l for l in out.split('\n') if ERRL.search(l)]
    return NUM.sub('#', lines[-1].strip().lower())[:160] if lines else None


def fail_sig(out):
    m = re.findall(r"(FAILED [^\s]+|ERROR [^\s]+|\d+ failed|[A-Za-z]*Error: [^\n]{0,80}|AssertionError[^\n]{0,80})", out)
    return NUM.sub('#', ' | '.join(sorted(set(m))[:6]).lower())[:300] if m else err_sig(out)


def action_loops(steps):
    kinds = []; raws = []; rsh_l = []; obs_sh = []; errs = []
    cycles = []  # (step, fail_sig) for edit->test fail
    last_edit = -99; last_fail_sig = None; same_fail_run = 0; max_same_fail_run = 0; n_etf = 0
    fmt_err = 0; fmt_run = 0; max_fmt_run = 0
    for i, s in enumerate(steps):
        ks = '\n'.join(s['cmds']).strip()
        raw = re.sub(r"\s+", ' ', ks)
        tmp = NUM.sub('#', raw.lower())
        rsh = shingle(raw); tsh = shingle(tmp)
        out = '\n'.join(o for _, o in s['obs'])
        rc = max([abs(int(c)) for c, _ in s['obs'] if c is not None] or [0])
        osh = shingle(NUM.sub('#', out[-1500:].lower()))
        es = err_sig(out[-3000:]) if rc != 0 else None
        best_raw = 0.0; jr = -1; best_t = 0.0
        for j in range(max(0, i - 8), i):
            if not raw or not raws[j]: continue
            vr = 1.0 if raw == raws[j] else jac(rsh, rsh_l[j][0])
            vt = jac(tsh, rsh_l[j][1])
            if vr > best_raw: best_raw, jr = vr, j
            if vt > best_t: best_t = vt
        kind = None
        if 'Tool call error' in s['user'] or 'format error' in s['user'].lower():
            kind = 'format_error'; fmt_err += 1; fmt_run += 1
        else:
            fmt_run = 0
        max_fmt_run = max(max_fmt_run, fmt_run)
        if kind is None and raw:
            if best_raw >= 0.9:
                if POLL.search(ks): kind = 'poll'
                elif errs[jr] is not None and es is not None and es == errs[jr]: kind = 'retry_same_error'
                elif jac(osh, obs_sh[jr]) >= 0.8: kind = 'reread_same_output'
                else: kind = 'rerun'
            elif best_t >= 0.9:
                kind = 'poll' if POLL.search(ks) else ('page' if PAGE.search(ks) else 'template_repeat')
        # edit -> test -> fail cycles
        is_edit = bool(EDIT.search(ks)); is_test = bool(TEST.search(ks))
        if is_edit: last_edit = i
        if is_test and last_edit >= i - 3 and last_edit >= 0:
            failed = rc != 0 or bool(re.search(r"\bFAILED\b|\d+ failed|Traceback|Error:", out))
            if failed:
                fs = fail_sig(out[-4000:])
                n_etf += 1
                if fs is not None and fs == last_fail_sig:
                    same_fail_run += 1
                else:
                    same_fail_run = 1
                last_fail_sig = fs
                max_same_fail_run = max(max_same_fail_run, same_fail_run)
                if kind is None and same_fail_run >= 2: kind = 'edit_test_same_fail'
            else:
                same_fail_run = 0; last_fail_sig = None
        raws.append(raw); rsh_l.append((rsh, tsh)); obs_sh.append(osh); errs.append(es)
        kinds.append(kind)
    return kinds, dict(n_edit_test_fail=n_etf, max_same_fail_run=max_same_fail_run, n_format_error=fmt_err, max_format_error_run=max_fmt_run)


def maxrun(v):
    m = r = 0
    for x in v:
        r = r + 1 if x else 0; m = max(m, r)
    return m


def work(args):
    t, steps = args
    try:
        res, iters, nls, units, commits, topic_rev = detect.analyse(t)
    except Exception as e:
        return None
    st = t['step_tok']
    for it in iters:
        a = int(np.searchsorted(st, it['ta'], side='right') - 1); b = int(np.searchsorted(st, it['src_ta'], side='right') - 1)
        it['step'] = a; it['src_step'] = b; it['cross_step'] = a != b; it['step_gap'] = a - b
    kinds, extra = action_loops(steps)
    ct = []
    for s in steps:
        ct.append(len(s['reasoning']) // 3.3)
    FUT = ('retry_same_error', 'reread_same_output')
    fut = [k in FUT for k in kinds]
    res.update(extra)
    res.update(dataset=t['dataset'], repo=t['repo'], language=t['language'], n_steps=t['n_steps'], ref_lines=t['ref_lines'],
               n_futile=int(sum(fut)), max_run_futile=maxrun(fut), n_poll=kinds.count('poll'), n_page=kinds.count('page'),
               n_rerun=kinds.count('rerun'), n_template_repeat=kinds.count('template_repeat'), n_edit_test_same_fail=kinds.count('edit_test_same_fail'),
               cross_step_loop_tok=sum(it['loop_tok'] for it in iters if it['cross_step']),
               within_step_loop_tok=sum(it['loop_tok'] for it in iters if not it['cross_step']))
    res['topic_extra_tok'] = 0
    lr = [(u[0], u[1]) for u in units if u[2]]
    for a, b, v, sa, sb in topic_rev:
        if v >= 0.5:
            ov = sum(max(0, min(b, y) - max(a, x)) for x, y in lr)
            res['topic_extra_tok'] += max(0, (b - a) - ov)
    res['action_loop'] = res['n_futile'] >= 3 or res['max_same_fail_run'] >= 3 or res['max_format_error_run'] >= 2
    step_kinds = [(i, k) for i, k in enumerate(kinds) if k]
    return res, dict(iters=iters, nonloop_starts=nls, units=units, topic_rev=topic_rev, step_kinds=step_kinds)


if __name__ == '__main__':
    T = pickle.load(open(f'{OUT}/traces_light.pkl', 'rb'))
    S = pickle.load(open(f'{OUT}/steps.pkl', 'rb'))
    T = [t for t in T if t['ntok'] >= 500]
    print('traces', len(T), flush=True)
    with Pool(8) as p:
        out = p.map(work, [(t, S[t['tid']]) for t in T], chunksize=4)
    stats = []; details = {}
    for t, o in zip(T, out):
        if o is None: continue
        stats.append(o[0]); details[t['tid']] = o[1]
    with open(f'{OUT}/trace_stats.jsonl', 'w', encoding='utf-8') as fo:
        for r in stats: fo.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
    pickle.dump(details, open(f'{OUT}/details.pkl', 'wb'), protocol=5)
    print('done', len(stats))
