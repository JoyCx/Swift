import pickle, numpy as np
T = pickle.load(open('A:/swift/runs/loops/traces.pkl', 'rb'))
for t in T:
    t['starts'] = np.fromiter((o[0] for o in t['offsets']), dtype=np.int64, count=len(t['offsets']))
    t['ids'] = np.asarray(t['ids'], dtype=np.int32)
    del t['offsets']
pickle.dump(T, open('A:/swift/runs/loops/traces_light.pkl', 'wb'), protocol=5)
print(len(T))
