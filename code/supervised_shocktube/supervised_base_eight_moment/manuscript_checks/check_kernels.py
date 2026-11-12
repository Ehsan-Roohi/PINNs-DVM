"""Gaussian inner products of the velocity kernels quoted in the manuscript (unit Maxwellian weight)."""
import json
import sys
import numpy as np

n = 40
t, w = np.polynomial.hermite_e.hermegauss(n)        # weight exp(-t^2/2)
w = w / np.sqrt(2 * np.pi)
cx, cy, cz = np.meshgrid(t, t, t, indexing="ij")
W = w[:, None, None] * w[None, :, None] * w[None, None, :]
c2 = cx ** 2 + cy ** 2 + cz ** 2
K = {"1": np.ones_like(cx), "cx": cx, "c2": c2,
     "phi_q": (0.5 * c2 - 2.5) * cx / 8, "phi_sigma": (cx ** 2 - c2 / 3) / 4,
     "phi_m": (cx ** 3 - 0.6 * cx * c2) / np.sqrt(12 / 5),
     "phi_R": (c2 - 7) * (cx ** 2 - c2 / 3) / np.sqrt(56 / 3),
     "phi_Delta": (c2 ** 2 - 10 * c2 + 15) / np.sqrt(120)}
names = list(K)
G = np.array([[float((K[a] * K[b] * W).sum()) for b in names] for a in names])
np.set_printoptions(precision=6, suppress=True, linewidth=200)
print(names)
print(G)
# linear response of the moments to the amplitudes: d(moment_a)/d(A_b) = <kernel_a, phi_b> for f = M(1 + sum A_b phi_b)
mom = {"q": 0.5 * c2 * cx, "sigma": cx ** 2 - c2 / 3, "m": cx ** 3 - 0.6 * cx * c2,
       "R": c2 * (cx ** 2 - c2 / 3) - 7 * (cx ** 2 - c2 / 3), "Delta": c2 ** 2 - 15}
resp = np.array([[float((mom[a] * K[b] * W).sum()) for b in names[3:]] for a in mom])
print("response of (q, sigma, m, R, Delta) to unit (A_q, A_sigma, A_m, A_R, A_Delta), linearised about a unit Maxwellian:")
print(resp)
json.dump({"names": names, "gram": G.tolist(), "moments": list(mom), "linear_response": resp.tolist()},
          open(sys.argv[1], "w"), indent=1)
