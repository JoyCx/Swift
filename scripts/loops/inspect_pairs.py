"""Print matched unit pairs at Jaccard bins for detector validation."""
import pickle, json, random, sys, os
tag = os.environ.get('TAG', '_test')
T = pickle.load(open('A:/swift/runs/loops/traces_light.pkl', 'rb'))
idx = {(t['bench'], t['arm'], t['tid']): t for t in T}
D = pickle.load(open(f'A:/swift/runs/loops/details{tag}.pkl', 'rb'))
mode = sys.argv[1] if len(sys.argv) > 1 else 'nd'
lo, hi = float(sys.argv[2]), float(sys.argv[3])
nshow = int(sys.argv[4]) if len(sys.argv) > 4 else 8
random.seed(int(os.environ.get('SEED', 1)))
cands = []
for k, d in D.items():
    for row in d['units']:
        ta, tb, isl, ist, mj, m, bj, bm, prog, isc, rawj, red, rsrc = row
        bench = os.environ.get('BENCH')
        if bench and k[0] != bench: continue
        if os.environ.get('PROSE') and isc: continue
        if mode == 'nd' and lo <= mj < hi and m >= 0: cands.append((k, row, m))
        if mode == 'red' and lo <= red < hi and rsrc >= 0: cands.append((k, row, rsrc))
        if mode == 'bag' and lo <= bj < hi and mj < 0.5 and bm >= 0: cands.append((k, row, bm))
print('n candidates', len(cands))
for k, row, m in random.sample(cands, min(nshow, len(cands))):
    t = idx[k]; st = t['starts']; units = D[k]['units']
    def txt(r):
        a = int(st[r[0]]); b = int(st[r[1]]) if r[1] < len(st) else len(t['text'])
        return t['text'][a:b]
    print('=' * 100); print(k, 'jac', row[4], 'bag', row[6], 'red', row[11], 'raw', row[10], 'tokpos', row[0], 'src tokpos', units[m][0])
    print('--- EARLIER:'); print(txt(units[m])[:700])
    print('--- LATER:'); print(txt(row)[:700])
