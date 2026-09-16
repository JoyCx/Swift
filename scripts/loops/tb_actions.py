"""Terminal-Bench cross-step loop analysis: repeated actions / observations / reasoning across agent calls.

Per step:
  raw_sim   = similarity of raw (un-normalised) keystrokes to best earlier step in last 8 steps
  tmpl_sim  = same after number normalisation (catches sed -n 'X,Yp' paging, lr sweeps)
  kind of repeated action:
     poll     : read-only status check of a log/process (grep/tail/cat *.log, ps, sleep, ls -d /proc) -> waiting
     page     : sed -n / head / tail / cat over the same file with different ranges (tmpl dup but raw differs)
     retry    : raw dup whose previous result was an error, and the error repeats (futile retry)
     rerun    : raw dup, result differs (edit-run cycle or state change)
     reread   : raw dup, identical observation, not poll (futile: same command, same output)
     rewrite  : heredoc/cat > same path written again (template dup)
"""
import pickle, re, json, os
from collections import Counter
import numpy as np
import pandas as pd

OUT = 'A:/swift/runs/loops'
T = pickle.load(open(f'{OUT}/tb_trials.pkl', 'rb'))
NUM = re.compile(r"0x[0-9a-f]+|\d+(?:\.\d+)?")
PROMPT = re.compile(r"root@[0-9a-f]+:[^#\n]*#")
ERR = re.compile(r"error|failed|failure|traceback|no such file|not found|permission denied|segmentation fault|timed out|cannot|could not|exception|fatal|killed|unrecognized|invalid|EXIT:[1-9]", re.I)
WRE = re.compile(r"[a-z_][a-z_0-9]*|#|[^\sa-z_0-9#]", re.I)
POLL = re.compile(r"^\s*(sleep\b|tail\b|grep\b[^|]*\.log|cat\b[^|]*\.log|ps\b|pgrep\b|ls -d /proc|nvidia-smi|jobs\b|wait\b|free\b|cat /sys/fs/cgroup|echo\b)", re.M)
PAGE = re.compile(r"^\s*(sed -n|head\b|tail\b|cat\b|less\b|awk 'NR|nl\b|objdump\b[^|]*\|\s*sed)", re.M)
WRITE = re.compile(r"cat\s*>\s*(\S+)\s*<<|tee\s+(\S+)\s*<<|>\s*(\S+\.\w+)\s*<<")


def shingle(s, n=3):
    w = WRE.findall(s)
    if len(w) < n: return {' '.join(w)} if w else set()
    return {' '.join(w[i:i+n]) for i in range(len(w) - n + 1)}


def jac(a, b):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)


def clean_obs(o):
    o = PROMPT.sub('', o)
    o = o.replace('New Terminal Output:', '').replace('Current Terminal Screen:', '')
    return re.sub(r"\s+", ' ', o).strip()[-1500:]


def err_sig(o):
    lines = [l for l in o.split('\n') if ERR.search(l)]
    return NUM.sub('#', lines[-1].strip().lower())[:160] if lines else None


def rshingles(text):
    w = [x for x in WRE.findall(NUM.sub('#', text.lower())) if x.isalnum() or x in '#_']
    return {hash(' '.join(w[i:i+4])) for i in range(max(0, len(w) - 3))}


rows = []; step_rows = []
for tr in T:
    steps = tr['steps']
    raw_sh = []; tmp_sh = []; raws = []; obs_sh = []; errs = []; writes = []; rsh_hist = set()
    kinds = []
    for i, s in enumerate(steps):
        ks = '\n'.join(k for fn, k in s['tools'] if fn == 'bash_command').strip()
        raw = re.sub(r"\s+", ' ', ks)
        tmp = NUM.sub('#', raw.lower())
        rsh = shingle(raw); tsh = shingle(tmp)
        o = s['obs']; osh = shingle(clean_obs(o).lower()); es = err_sig(o[-3000:])
        wpaths = set(x for g in WRITE.findall(ks) for x in g if x)
        best_raw = 0.0; jr = -1; best_t = 0.0; jt = -1
        for j in range(max(0, i - 8), i):
            if not raw or not raws[j]: continue
            vr = 1.0 if raw == raws[j] else jac(rsh, raw_sh[j])
            vt = jac(tsh, tmp_sh[j])
            if vr > best_raw: best_raw, jr = vr, j
            if vt > best_t: best_t, jt = vt, j
        kind = None
        prev_rewrite = any(wpaths & writes[j] for j in range(max(0, i - 8), i)) if wpaths else False
        if raw:
            if best_raw >= 0.9:
                same_obs = jac(osh, obs_sh[jr]) >= 0.8
                if POLL.search(ks) and not wpaths:
                    kind = 'poll'
                elif errs[jr] is not None and es is not None and es == errs[jr]:
                    kind = 'retry_same_error'
                elif same_obs:
                    kind = 'reread_same_output'
                else:
                    kind = 'rerun'
            elif prev_rewrite:
                kind = 'rewrite_same_file'
            elif best_t >= 0.9:
                kind = 'poll' if POLL.search(ks) else ('page' if PAGE.search(ks) else 'template_repeat')
        rs = rshingles(s['reasoning'])
        red = len(rs & rsh_hist) / len(rs) if len(rs) >= 20 else float('nan')
        rsh_hist |= rs
        prev_err = errs[-1] is not None if i > 0 else False
        json_fail = ('provide a proper JSON response' in o) or ('No valid JSON object found' in o)
        is_wait = (not raw) and any(fn == 'bash_command' for fn, k in s['tools']) and not json_fail
        is_complete = any(fn == 'mark_task_complete' for fn, k in s['tools'])
        if json_fail: kind = 'json_format_fail'
        elif is_wait: kind = 'wait_empty'
        raws.append(raw); raw_sh.append(rsh); tmp_sh.append(tsh); obs_sh.append(osh); errs.append(es); writes.append(wpaths)
        kinds.append(kind)
        step_rows.append(dict(arm=tr['arm'], task=tr['task'], trial=tr['trial'], step=i, ct=s['ct'], kind=kind,
                              src=jr if kind in ('poll', 'retry_same_error', 'reread_same_output', 'rerun') else jt,
                              raw_sim=round(best_raw, 3), tmpl_sim=round(best_t, 3), prev_err=prev_err, err_sig=es,
                              cross_red=None if red != red else round(red, 3), noop=not raw, json_fail=json_fail, is_wait=is_wait, is_complete=is_complete, reward=tr['reward'], exc=tr['exc']))
    n = len(steps)
    FUT = ('retry_same_error', 'reread_same_output')

    def maxrun(v):
        m = r = 0
        for x in v:
            r = r + 1 if x else 0; m = max(m, r)
        return m
    ct = np.array([s['ct'] for s in steps], dtype=float) if steps else np.zeros(0)
    fut = [k in FUT for k in kinds]
    rep = [k in FUT for k in kinds]
    jf = [k == 'json_format_fail' for k in kinds]; wt = [k in ('wait_empty', 'poll') for k in kinds]
    ncomp = sum(1 for r in step_rows[-n:] if r['is_complete']) if n else 0
    noop = [not r for r in raws]
    cc = Counter(k for k in kinds if k)
    cr = np.array([r['cross_red'] or 0.0 for r in step_rows[-n:]]) if n else np.zeros(0)
    row = dict(arm=tr['arm'], task=tr['task'], trial=tr['trial'], reward=tr['reward'], exc=tr['exc'], n_steps=n, total_ct=int(ct.sum()),
               n_futile=int(sum(fut)), max_run_futile=maxrun(fut), n_rep=int(sum(rep)), max_run_rep=maxrun(rep), n_noop=int(sum(noop)),
               max_run_noop=maxrun(noop), n_json_fail=int(sum(jf)), max_run_json_fail=maxrun(jf), max_run_wait=maxrun(wt), n_complete_calls=ncomp, ct_in_futile=int(ct[np.array(fut, dtype=bool)].sum()) if n else 0,
               cross_red_tokw=float((cr * ct).sum() / max(ct.sum(), 1)) if n else 0.0)
    for k in ('wait_empty', 'json_format_fail', 'poll', 'page', 'retry_same_error', 'reread_same_output', 'rerun', 'rewrite_same_file', 'template_repeat'):
        row['n_' + k] = cc.get(k, 0)
    row['loop_futile'] = row['n_futile'] >= 3
    row['loop_json'] = row['max_run_json_fail'] >= 2 or row['n_json_fail'] >= 4
    row['loop_wait'] = row['max_run_wait'] >= 4
    row['loop_complete'] = ncomp >= 3
    row['action_loop'] = row['loop_futile'] or row['loop_json'] or row['loop_wait']
    rows.append(row)
df = pd.DataFrame(rows); sd = pd.DataFrame(step_rows)
df.to_json(f'{OUT}/tb_trial_loops.jsonl', orient='records', lines=True, force_ascii=False)
sd.to_json(f'{OUT}/tb_step_loops.jsonl', orient='records', lines=True, force_ascii=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
df['timeout'] = df.exc == 'AgentTimeoutError'; df['solved'] = df.reward == 1.0
print(df.groupby('arm')[[c for c in df.columns if c.startswith('n_') or c.startswith('max_')] + ['cross_red_tokw', 'action_loop']].mean().round(3).T)
print(df.groupby('arm')[['loop_futile', 'loop_json', 'loop_wait', 'loop_complete', 'action_loop']].mean().round(3))
for c in ['loop_futile', 'loop_json', 'loop_wait', 'action_loop']:
    print(c); print(df.groupby(['arm', c])[['solved', 'timeout']].agg(['mean', 'count']).round(3))
print(df.groupby(['arm', 'action_loop'])[['solved', 'timeout', 'n_steps', 'total_ct']].agg(['mean', 'count']).round(3))
print(df.groupby(['arm', 'timeout'])[['n_futile', 'n_rep', 'max_run_rep', 'cross_red_tokw', 'action_loop']].mean().round(3))
