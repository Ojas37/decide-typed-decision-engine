"""Compare two artifact dirs: weights/vocab bit-identical? calibration + decision identical?"""
import json
import sys

import numpy as np

a, b = sys.argv[1], sys.argv[2]
wa, wb = np.load(a + "/weights.npz"), np.load(b + "/weights.npz")
same_w = sorted(wa.files) == sorted(wb.files) and all(np.array_equal(wa[k], wb[k]) for k in wa.files)
same_v = json.load(open(a + "/vocab.json")) == json.load(open(b + "/vocab.json"))
ma, mb = json.load(open(a + "/meta.json")), json.load(open(b + "/meta.json"))
print("weights.npz bit-identical : %s" % same_w)
print("vocab.json identical      : %s" % same_v)
print("calibration identical     : %s" % (ma["calibration"] == mb["calibration"]))
print("decision identical        : %s" % (ma["decision"] == mb["decision"]))
ok = same_w and same_v and ma["calibration"] == mb["calibration"] and ma["decision"] == mb["decision"]
sys.exit(0 if ok else 1)
