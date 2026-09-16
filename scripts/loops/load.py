"""Build unified trace store for loop analysis. Output: A:/swift/runs/loops/traces.pkl (list of dicts)."""
import json, glob, os, pickle, csv, re, sys
import zstandard
from tokenizers import Tokenizer
R = 'A:/swift/data/external/ukisai-evals'
OUT = 'A:/swift/runs/loops'
MINTOK = int(os.environ.get('MINTOK', 8000))
traces = []  # long reasoning traces
tb_trials = []  # trial-level action sequences (all trials)

def tb():
    for arm in ('base', 'swift'):
        for d in sorted(glob.glob(f'{R}/08-terminal-bench-2.1/{arm}/*__*')):
            tn = os.path.basename(d); task = tn.split('__')[0]
            try:
                res = json.load(open(f'{d}/result.json', encoding='utf-8'))
            except Exception:
                res = {}
            vr = (res.get('verifier_result') or {}).get('rewards') or {}
            reward = vr.get('reward')
            exc = (res.get('exception_info') or {}) or {}
            exc = exc.get('exception_type') if isinstance(exc, dict) else None
            try:
                t = json.load(open(f'{d}/agent/trajectory.json', encoding='utf-8'))
            except Exception as e:
                print('missing traj', d, e); continue
            steps = []
            for s in t['steps']:
                if s.get('source') != 'agent': continue
                ct = (s.get('metrics') or {}).get('completion_tokens') or 0
                if ct == 0: continue
                kc = []
                for tc in s.get('tool_calls') or []:
                    a = tc.get('arguments') or {}
                    kc.append((tc.get('function_name'), a.get('keystrokes', '') if isinstance(a, dict) else str(a)))
                obs = ''
                ob = s.get('observation') or {}
                for r_ in ob.get('results') or []:
                    obs += str(r_.get('content') or '')
                steps.append(dict(step_id=s.get('step_id'), ct=ct, reasoning=s.get('reasoning_content') or '',
                                  message=s.get('message') or '', tools=kc, obs=obs))
            tb_trials.append(dict(bench='tb', arm=arm, task=task, trial=tn, reward=reward, exc=exc, steps=steps))
            for i, st in enumerate(steps):
                if st['ct'] >= MINTOK * 0.9 and st['reasoning']:
                    prev_obs = steps[i-1]['obs'] if i > 0 else ''
                    traces.append(dict(bench='tb', arm=arm, tid=f'{tn}#{st["step_id"]}', group=task, text=st['reasoning'],
                                       ntok_meta=st['ct'], correct=(reward == 1.0) if reward is not None else None,
                                       truncated=None, exc=exc, difficulty=None, prev_obs=prev_obs[-3000:], step_idx=i, nsteps=len(steps)))

def lcb():
    grades = {}
    for gd in ('official_grades_v6', 'official_grades_v1_v5'):
        for f in glob.glob(f'{R}/09-livecodebench-v6/{gd}/*_seed*_codegeneration_output_eval_all.json'):
            m = re.search(r'(base|swift)_seed(\d)', os.path.basename(f)); arm, seed = m.group(1), m.group(2)
            for g in json.load(open(f, encoding='utf-8')):
                grades[(arm, seed, str(g['question_id']))] = bool(g['graded_list'][0])
    for arm in ('base', 'swift'):
        for f in glob.glob(f'{R}/09-livecodebench-v6/{arm}/seed*/*.json'):
            x = json.load(open(f, encoding='utf-8'))
            ct = int(x['completion_tokens'])
            if ct < MINTOK * 0.9: continue
            seed = str(x['seed']); q = str(x['question_id'])
            traces.append(dict(bench='lcb', arm=arm, tid=f's{seed}/{q}', group=q, pair=(q, seed), text=x.get('reasoning') or '',
                               ntok_meta=ct, correct=grades.get((arm, seed, q)), truncated=x['finish_reason'] == 'length',
                               difficulty=x.get('difficulty')))

def math():
    for bench, sub, name in (('aime', '05-aime-2026', 'aime_2026'), ('hmmt', '06-hmmt-nov-2025', 'hmmt_nov_2025')):
        for f in glob.glob(f'{R}/{sub}/matharena_outputs/*/{name}/*_s*/*.json.zst'):
            tag = os.path.basename(os.path.dirname(f)); arm, seed = tag.split('_s')
            d = json.loads(zstandard.ZstdDecompressor().stream_reader(open(f, 'rb')).read())
            msgs = d['messages'][0]
            cot = ''.join(m.get('content') or '' for m in msgs if isinstance(m, dict) and m.get('type') == 'cot')
            if len(cot) < MINTOK * 2.5: continue
            idx = str(d['idx'])
            traces.append(dict(bench=bench, arm=arm, tid=f's{seed}/{idx}', group=idx, pair=(idx, seed), text=cot, ntok_meta=None,
                               correct=bool(d['correct'][0]), truncated=None, difficulty=None))

def gpqa():
    for arm, dn in (('base', 'base_reference'), ('swift', 'swift')):
        for seed in range(5):
            for l in open(f'{R}/01-gpqa-diamond/adapter_only_dp4x2/{dn}/seed{seed}.raw.jsonl', encoding='utf-8'):
                r = json.loads(l)
                rt = int(r['reasoning_tokens'])
                if rt < MINTOK * 0.9: continue
                tid = r['task_id']
                traces.append(dict(bench='gpqa', arm=arm, tid=f's{seed}/{tid}', group=tid, pair=(tid, str(seed)), text=r['reasoning'] or '',
                                   ntok_meta=rt, correct=str(r['correct']) == 'True', truncated=r['finish_reason'] == 'length', difficulty=r.get('subdomain')))

tb(); print('tb', len(traces), flush=True)
lcb(); print('lcb', len(traces), flush=True)
math(); print('math', len(traces), flush=True)
gpqa(); print('gpqa', len(traces), flush=True)
tok = Tokenizer.from_file('A:/models/Qwen3.8-27B/tokenizer.json')
B = 64
for i in range(0, len(traces), B):
    enc = tok.encode_batch([t['text'] for t in traces[i:i+B]], add_special_tokens=False)
    for t, e in zip(traces[i:i+B], enc):
        t['ids'] = e.ids; t['offsets'] = e.offsets; t['ntok'] = len(e.ids)
traces = [t for t in traces if t['ntok'] >= MINTOK]
from collections import Counter
print(Counter((t['bench'], t['arm']) for t in traces))
pickle.dump(traces, open(f'{OUT}/traces.pkl', 'wb'), protocol=5)
pickle.dump(tb_trials, open(f'{OUT}/tb_trials.pkl', 'wb'), protocol=5)
