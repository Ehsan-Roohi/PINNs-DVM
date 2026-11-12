"""Size of the reference moments where the final state has the wrong sign (saved arrays only)."""
import sys
import numpy as np

d = np.load(sys.argv[1], allow_pickle=False)
x, ref, fin = d["x"], d["reference"], d["stageB"]
cols = [str(a) for a in d["columns"]]
for n, lo, hi in (("sigma", -0.5, -0.228), ("R", 0.192, 0.5), ("Delta", 0.194, 0.5)):
    j = cols.index(n)
    m = (x >= lo) & (x <= hi)
    wrong = m & (np.sign(fin[:, j]) != np.sign(ref[:, j]))
    ext = abs(ref[:, j]).max()
    print("%-6s region [%.3f, %.3f]: wrong-sign nodes %d of %d; max |ref| there / extremum = %.3f %%; max |model| there / extremum = %.3f %%"
          % (n, lo, hi, wrong.sum(), m.sum(), 100 * abs(ref[wrong, j]).max() / ext, 100 * abs(fin[wrong, j]).max() / ext))
# q and sigma relative deficit near the upstream edge
for n in ("q", "sigma"):
    j = cols.index(n)
    for xx in (-0.12, -0.135, -0.15, -0.16, -0.18):
        a, b = np.interp(xx, x, fin[:, j]), np.interp(xx, x, ref[:, j])
        print("%-6s x=%.3f model/ref - 1 = %+.1f %%" % (n, xx, 100 * (a / b - 1)))
