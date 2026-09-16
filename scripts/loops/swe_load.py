"""Load a stratified subsample of nvidia/Open-SWE-Traces (Qwen3.8-27B, mini-swe-agent) into loop-analysis format.

Sampling: per dataset (scale-swe, swe-rebench-v2) and per resolved label, the same fraction (proportional stratification),
N_PER_DATASET trajectories per dataset, seed 0.
Outputs: runs/loops/swe/traces_light.pkl  (trajectory-level concatenated thinking, one step = starts on a new paragraph)
         runs/loops/swe/steps.pkl         (per trajectory: steps with reasoning, commands, tool results)
"""
import glob, json, os, pickle, random, re
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from tokenizers import Tokenizer

ROOT = 'A:/swift/data/external/open-swe-traces/data/minisweagent/qwen38_27b'
OUT = 'A:/swift/runs/loops/swe'
os.makedirs(OUT, exist_ok=True)
N_PER_DATASET = int(os.environ.get('NPER', 7000))
rng = np.random.default_rng(0)


def nz(x):
    return None if x is None or x == 'None' else x


traces = []; steps_all = {}
for ds in ('scale-swe', 'swe-rebench-v2'):
    files = sorted(glob.glob(f'{ROOT}/{ds}/*.parquet'))
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    frac = min(1.0, N_PER_DATASET / total)
    for f in files:
        meta = pq.read_table(f, columns=['resolved']).column('resolved').to_numpy()
        pick = []
        for lab in np.unique(meta):
            idx = np.where(meta == lab)[0]
            k = int(round(len(idx) * frac))
            pick.extend(rng.choice(idx, size=min(k, len(idx)), replace=False).tolist())
        pick = sorted(pick)
        tab = pq.read_table(f, columns=['instance_id', 'repo', 'language', 'trajectory_id', 'messages', 'resolved', 'metadata']).take(pa.array(pick))
        for r in tab.to_pylist():
            ms = r['messages']
            steps = []
            for i, m in enumerate(ms):
                if m['role'] == 'assistant':
                    cmds = []
                    for tc in m.get('tool_calls') or []:
                        try:
                            a = json.loads(tc['function']['arguments'])
                            cmds.append(a.get('command', '') if isinstance(a, dict) else str(a))
                        except Exception:
                            cmds.append(str(tc['function'].get('arguments', '')))
                    steps.append(dict(reasoning=nz(m.get('reasoning_content')) or '', content=nz(m.get('content')) or '', cmds=cmds, obs=[], user=''))
                elif m['role'] == 'tool' and steps:
                    try:
                        o = json.loads(m['content']); steps[-1]['obs'].append((o.get('returncode'), str(o.get('output', ''))[-4000:]))
                    except Exception:
                        steps[-1]['obs'].append((None, (m['content'] or '')[-4000:]))
                elif m['role'] == 'user' and steps:
                    steps[-1]['user'] += (m['content'] or '')[:2000]
            md = r['metadata'] or {}
            tid = r['trajectory_id']
            # concatenate thinking; record char offset of each step
            parts = []; offs = []; pos = 0
            for s in steps:
                txt = s['reasoning'].strip()
                offs.append(pos)
                if txt:
                    parts.append(txt); pos += len(txt) + 2
            text = '\n\n'.join(parts)
            # recompute offsets for steps precisely
            step_char = []; pos = 0
            for s in steps:
                step_char.append(pos)
                if s['reasoning'].strip(): pos += len(s['reasoning'].strip()) + 2
            traces.append(dict(bench='swe', arm='qwen38', tid=tid, group=r['instance_id'], text=text, correct=bool(r['resolved']), truncated=None,
                               exc=None, difficulty=(md.get('category') if isinstance(md, dict) else None), dataset=ds, repo=r['repo'], language=r['language'],
                               n_steps=len(steps), step_char=step_char,
                               ref_lines=((md.get('reference_patch') or {}).get('num_modified_lines') if isinstance(md, dict) else None)))
            steps_all[tid] = steps
        print(ds, os.path.basename(f), len(traces), flush=True)

tok = Tokenizer.from_file('A:/models/Qwen3.8-27B/tokenizer.json')
B = 64
for i in range(0, len(traces), B):
    chunk = traces[i:i+B]
    enc = tok.encode_batch([t['text'] for t in chunk], add_special_tokens=False)
    for t, e in zip(chunk, enc):
        t['ids'] = np.asarray(e.ids, dtype=np.int32)
        t['starts'] = np.fromiter((o[0] for o in e.offsets), dtype=np.int64, count=len(e.offsets))
        t['ntok'] = len(e.ids)
        t['step_tok'] = np.searchsorted(t['starts'], np.asarray(t['step_char'], dtype=np.int64)).astype(np.int64)
    if i % 3200 == 0: print('tok', i, flush=True)
pickle.dump(traces, open(f'{OUT}/traces_light.pkl', 'wb'), protocol=5)
pickle.dump(steps_all, open(f'{OUT}/steps.pkl', 'wb'), protocol=5)
nt = np.array([t['ntok'] for t in traces])
print('n', len(traces), 'resolved', np.mean([t['correct'] for t in traces]), 'ntok median', np.median(nt), 'p90', np.percentile(nt, 90), '>=8k', (nt >= 8000).mean())
