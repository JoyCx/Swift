"""Build runs/loops/gallery.md from detected loops (UkisAI data + Open-SWE-Traces)."""
import json, pickle, re, os, random
import numpy as np
import pandas as pd

OUT = 'A:/swift/runs/loops'
random.seed(4)
T = {(t['bench'], t['arm'], t['tid']): t for t in pickle.load(open(f'{OUT}/traces_light.pkl', 'rb'))}
D = pickle.load(open(f'{OUT}/details.pkl', 'rb'))
it = pd.read_json(f'{OUT}/iterations.jsonl', lines=True)
ts = pd.read_json(f'{OUT}/trace_stats.jsonl', lines=True)
trials = {tr['trial']: tr for tr in pickle.load(open(f'{OUT}/tb_trials.pkl', 'rb'))}
tbs = pd.read_json(f'{OUT}/tb_step_loops.jsonl', lines=True)
tbt = pd.read_json(f'{OUT}/tb_trial_loops.jsonl', lines=True)


def clip(s, n):
    s = s.strip()
    return s if len(s) <= n else s[:n] + ' [...]'


def span(t, a, b):
    st = t['starts']; L = len(st)
    ca = int(st[a]) if a < L else len(t['text']); cb = int(st[b]) if b < L else len(t['text'])
    return t['text'][ca:cb]


def fence(s):
    return '```text\n' + s.replace('```', "'''") + '\n```'


entries = []


def add_iter(row, title, note, n_early=600, n_late=700):
    k = (row.bench, row.arm, row.tid); t = T[k]
    pre = span(t, max(0, row.ta - 32), row.ta)
    early = span(t, row.src_ta, row.src_tb); late = span(t, row.ta, row.tb)
    md = [f"### {title}", f"- benchmark `{row.bench}` | arm **{row.arm}** | trace `{row.tid}` | trace tokens {row.ntok} | correct={row.correct} truncated={row.truncated}",
          f"- category `{row['cat']}` | iteration tokens {row.ta}-{row.tb} ({row.loop_tok} loop tok) | source span tokens {row.src_ta}-{row.src_tb} | jaccard {row.jac:.2f} redundancy {row.red:.2f} | cluster visits {row.visits}",
          f"- {note}", f"**Earlier span (tok {row.src_ta}):**", fence(clip(early, n_early)),
          f"**Loop iteration (tok {row.ta}); 32 tokens before entry:** `{' '.join(pre.split())[-160:]}`", fence(clip(late, n_late))]
    entries.append('\n'.join(md))


def pick(q, title, note, seed=0, sort=None):
    s = it.query(q)
    if sort: s = s.sort_values(sort, ascending=False).head(30)
    if len(s) == 0:
        print('no match', q); return
    row = s.sample(1, random_state=seed).iloc[0]
    add_iter(row, title, note)


# ---------------- Terminal-Bench (within-call) ----------------
pick("bench=='tb' and arm=='base' and cat=='verification' and code<0.5 and loop_tok>80", "TB within-call: re-verification of an already-checked plan (base)", "Paragraph-level re-check of material written earlier in the same call.", seed=1)
pick("bench=='tb' and arm=='swift' and cat=='hedge_reconsider' and code<0.5 and loop_tok>60", "TB within-call: hedge-led re-derivation (swift)", "Swift still re-enters earlier material behind a hedge opener.", seed=2)
pick("bench=='tb' and arm=='swift' and cat=='code_redraft' and loop_tok>300", "TB within-call: rewriting the same script inside thinking (swift)", "Code redraft is the dominant loop type by tokens in both arms and Swift did not reduce it.", seed=3)
pick("bench=='tb' and arm=='base' and cat=='rederivation_other' and code<0.5 and loop_tok>120 and dist_units>3", "TB within-call: re-deriving the same analysis (base)", "Non-adjacent revisit of the same sub-problem.", seed=4)
dg = ts[(ts.degen_tok > 0)].sort_values('degen_tok', ascending=False)
# ---------------- TB action loops ----------------


def tb_action(trial, kinds, title, note, maxn=6):
    tr = trials[trial]; s = tbs[(tbs.trial == trial)]
    rows = s[s.kind.isin(kinds)].head(maxn)
    md = [f"### {title}", f"- benchmark `tb` | arm **{tr['arm']}** | trial `{trial}` | reward {tr['reward']} | exception {tr['exc']} | steps {len(tr['steps'])}", f"- {note}"]
    body = []
    for r in rows.itertuples():
        st = tr['steps'][r.step]
        cmd = ' || '.join(k for f, k in st['tools'])
        body.append(f"[step {r.step} kind={r.kind} ct={r.ct}]\nREASONING: {clip(' '.join(st['reasoning'].split()), 260)}\nCMD: {clip(cmd, 160)}\nOBS(tail): {clip(' '.join(st['obs'][-220:].split()), 220)}")
    md.append(fence('\n\n'.join(body)))
    entries.append('\n'.join(md))


def pick_trial(cond, kinds, title, note, seed=0):
    s = tbt.query(cond)
    if len(s) == 0: print('no trial', cond); return
    tb_action(s.sample(1, random_state=seed).iloc[0].trial, kinds, title, note)


pick_trial("arm=='base' and max_run_json_fail>=3", ['json_format_fail'], "TB action loop: JSON-format failure loop (base)",
           "Agent's response repeatedly fails the harness JSON parser; each retry re-plans instead of emitting a minimal valid object.", seed=1)
pick_trial("arm=='swift' and n_retry_same_error>=2", ['retry_same_error', 'reread_same_output'], "TB action loop: retry with the same error (swift)",
           "Same command re-issued, same error signature returned.", seed=2)
pick_trial("arm=='swift' and max_run_wait>=5", ['wait_empty', 'poll'], "TB action loop: wait/poll loop (swift)",
           "Empty-keystroke waits / polling while a foreground job runs; Swift has 2x base's wait-loop trial rate.", seed=3)
pick_trial("arm=='base' and n_reread_same_output>=2", ['reread_same_output', 'retry_same_error', 'rerun'], "TB action loop: re-reading identical output (base)",
           "Command repeated with byte-identical observation.", seed=5)

# ---------------- GPQA ----------------
ff = ts[(ts.bench == 'gpqa') & (ts.flip_returns >= 3)].sort_values('flip_returns', ascending=False)
if len(ff):
    r = ff.iloc[0]; k = ('gpqa', r.arm, r.tid); t = T[k]; cm = D[k]['commits']
    seq = ' -> '.join(c for _, c in cm)
    body = []
    for p, c in cm[:10]:
        a = int(t['starts'][p]); body.append(f"[tok {p}] ...{' '.join(t['text'][max(0, a - 160):a + 20].split())}")
    entries.append('\n'.join([f"### GPQA flip-flop between answer options ({r.arm})", f"- benchmark `gpqa` | arm **{r.arm}** | trace `{r.tid}` | tokens {r.ntok} | correct={r.correct}",
                              f"- committed-answer sequence: `{seq}` ({r.flip_returns} returns to an abandoned option)", fence('\n'.join(body))]))
pick("bench=='gpqa' and arm=='base' and truncated==True and code<0.5 and loop_tok>60", "GPQA: hypothesis cycling in a length-truncated trace (base)", "Truncated at 100k tokens; the same candidate encodings are re-tried.", seed=0)
pick("bench=='gpqa' and head.str.lower().str.contains('search memory')", "GPQA: 'Let's search memory' recall loop", "Repeated attempts to recall a fact, each re-listing the same candidates.", seed=1)
# ---------------- LCB ----------------
pick("bench=='lcb' and arm=='base' and truncated==True and cat=='code_redraft' and loop_tok>250", "LCB: micro-optimisation code redraft loop in a truncated trace (base)", "Full solution rewritten again with cosmetic changes (local-variable binding etc.).", seed=0)
pick("bench=='lcb' and arm=='swift' and cat=='verification' and code<0.5 and loop_tok>50", "LCB: edge-case re-verification (swift)", "'Now, let's consider if...' spiral: Swift keeps this pattern.", seed=1)
# ---------------- math ----------------
pick("bench in ['aime','hmmt'] and arm=='swift' and cat=='format_replan'", "Math: answer-format re-planning after the answer is known (swift)", "Repeated 'Need final answer within boxed' planning.", seed=0)
if len(dg):
    r = dg.iloc[0]; k = (r.bench, r.arm, r.tid); d = D[k]
    rows = it[(it.bench == r.bench) & (it.arm == r.arm) & (it.tid == r.tid)].sort_values('exact_frac', ascending=False)
    if len(rows): add_iter(rows.iloc[0], f"Degenerate exact repetition ({r.bench}, {r.arm})", f"Only {int((ts.degen_tok > 0).sum())} long traces in the whole UkisAI set contain back-to-back exact repetition.")
ffm = ts[(ts.bench.isin(['aime', 'hmmt'])) & (ts.flip_returns >= 2)].sort_values('flip_returns', ascending=False)
if len(ffm):
    r = ffm.iloc[0]; k = (r.bench, r.arm, r.tid); t = T[k]; cm = D[k]['commits']
    body = []
    for p, c in cm[:12]:
        a = int(t['starts'][p]); body.append(f"[tok {p}] ...{' '.join(t['text'][max(0, a - 140):a + 20].split())}")
    entries.append('\n'.join([f"### Math answer flip-flop ({r.bench}, {r.arm})", f"- trace `{r.tid}` | tokens {r.ntok} | correct={r.correct}",
                              f"- committed sequence: `{' -> '.join(c for _, c in cm)}`", fence('\n'.join(body))]))

# ---------------- Open-SWE-Traces ----------------
swe_md = []
if os.path.exists(f'{OUT}/swe/iterations.jsonl'):
    ST = {t['tid']: t for t in pickle.load(open(f'{OUT}/swe/traces_light.pkl', 'rb'))}
    SS = pickle.load(open(f'{OUT}/swe/steps.pkl', 'rb'))
    sit = pd.read_json(f'{OUT}/swe/iterations.jsonl', lines=True)
    sts = pd.read_json(f'{OUT}/swe/trace_stats.jsonl', lines=True)
    SD = pickle.load(open(f'{OUT}/swe/details.pkl', 'rb'))

    def swe_iter(q, title, note, seed=0):
        s = sit.query(q)
        if len(s) == 0: print('no swe', q); return
        row = s.sample(1, random_state=seed).iloc[0]; t = ST[row.tid]
        pre = span(t, max(0, row.ta - 32), row.ta)
        md = [f"### {title}", f"- source `open-swe-traces/{t['dataset']}` | repo `{t['repo']}` | trajectory `{row.tid}` | resolved={t['correct']} | thinking tokens {t['ntok']} | steps {t['n_steps']}",
              f"- category `{row['cat']}` | step {row.step} revisits step {row.src_step} | tokens {row.ta}-{row.tb} ({row.loop_tok} loop tok) | jaccard {row.jac:.2f} redundancy {row.red:.2f}",
              f"- {note}", f"**Earlier (step {row.src_step}, tok {row.src_ta}):**", fence(clip(span(t, row.src_ta, row.src_tb), 550)),
              f"**Loop iteration (step {row.step}, tok {row.ta}); 32 tokens before:** `{' '.join(pre.split())[-160:]}`", fence(clip(span(t, row.ta, row.tb), 650))]
        swe_md.append('\n'.join(md))

    def swe_action(cond, kinds, title, note, seed=0, maxn=6):
        s = sts.query(cond)
        if len(s) == 0: print('no swe action', cond); return
        r = s.sample(1, random_state=seed).iloc[0]; steps = SS[r.tid]
        sk = [(i, k) for i, k in SD[r.tid]['step_kinds'] if k in kinds][:maxn]
        body = []
        for i, k in sk:
            st = steps[i]
            out = '\n'.join(o for _, o in st['obs'])
            body.append(f"[step {i} kind={k}]\nREASONING: {clip(' '.join(st['reasoning'].split()), 260)}\nCMD: {clip(' || '.join(st['cmds']), 180)}\nOBS(tail): {clip(' '.join(out[-240:].split()), 240)}")
        swe_md.append('\n'.join([f"### {title}", f"- source `open-swe-traces/{r.dataset}` | repo `{r.repo}` | trajectory `{r.tid}` | resolved={r.correct} | steps {r.n_steps}", f"- {note}", fence('\n\n'.join(body))]))

    swe_iter("cross_step==True and code<0.5 and cat=='verification' and loop_tok>60 and correct==False", "SWE: cross-step re-verification of the same fix (unresolved)", "Reasoning in a later step re-derives an earlier step's analysis.", seed=1)
    swe_iter("cross_step==True and code<0.5 and cat=='hedge_reconsider' and loop_tok>60", "SWE: hedge-led revisit across steps", "", seed=2)
    swe_iter("cross_step==False and code<0.5 and loop_tok>150", "SWE: within-step reasoning loop in a long step", "", seed=3)
    swe_iter("cross_step==True and cat=='code_redraft' and loop_tok>200 and correct==False", "SWE: re-drafting the same patch in thinking (unresolved)", "", seed=4)
    swe_action("max_same_fail_run>=3", ['edit_test_same_fail', 'retry_same_error', 'rerun'], "SWE action loop: edit -> test -> same failure cycle", "Edits do not change the failing signature.", seed=1)
    swe_action("n_futile>=3", ['retry_same_error', 'reread_same_output'], "SWE action loop: retry same command / same error", "", seed=2)
    swe_action("max_format_error_run>=2", ['format_error'], "SWE action loop: tool-call format error loop", "", seed=3)

md = ["# Loop gallery", "",
      "Representative loop excerpts detected automatically (paragraph near-duplicate / redundancy detector, topic revisits, answer flip-flops, cross-step action loops). "
      "Token offsets index the Qwen3.8 tokenisation of the thinking trace (TB: one agent call; Open-SWE: all steps' thinking concatenated). Excerpts are trimmed.", ""]
md += ["## UkisAI evals (Qwen3.8-27B base vs Swift adapter)", ""] + [e + '\n' for e in entries]
if swe_md:
    md += ["## Open-SWE-Traces (Qwen3.8-27B, mini-swe-agent)", ""] + [e + '\n' for e in swe_md]
open(f'{OUT}/gallery.md', 'w', encoding='utf-8').write('\n'.join(md))
print('entries', len(entries), 'swe', len(swe_md))
