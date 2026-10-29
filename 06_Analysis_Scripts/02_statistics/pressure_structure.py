"""Structure of the inlet pressure: p_in = p_sys(Q_N) + c * W / A_tube.

Fits, per tube diameter, a quadratic rig characteristic p_sys = a*Q^2 + b*Q (no constant term) plus a
common coefficient c on the weight pressure W/A_tube, to the WFE runs.  Then predicts out-of-fold
(LOUO/LOLO/LOSO) in two ways: with the measured Q_N (diagnostic, upper bound) and with Q_N from the
a-priori GPR flow model (ablation_oof.csv).  Usage:
  python3 -I pressure_structure.py ML_Data_All_v8.csv ablation_oof.csv out.json
"""
import sys, json
import numpy as np, pandas as pd
from numpy.linalg import lstsq

D = pd.read_csv(sys.argv[1]); AO = pd.read_csv(sys.argv[2])
assert (D.Source_File == AO.Source_File).all()
y = D.Pressure_Pa.values; Q = D.Flow_Nm3h.values; tube = D.Tube_D_mm.values
Wp = D.Wire_Weight_N.values / (np.pi / 4 * (tube / 1000.0)**2)


def design(Q, tube, Wp, with_w):
    cols = []
    for t in (8, 10):
        m = (tube == t).astype(float); cols += [m * Q**2, m * Q]
    if with_w: cols.append(Wp)
    return np.column_stack(cols)


def met(y, p):
    e = y - p
    return dict(R2=float(1 - np.sum(e**2) / np.sum((y - y.mean())**2)), MAPE=float(100 * np.mean(np.abs(e / y))), RMSE_kPa=float(np.sqrt(np.mean(e**2)) / 1e3))


K = {}
for w in (False, True):
    X = design(Q, tube, Wp, w); b = lstsq(X, y, rcond=None)[0]
    K[f"insample_{'with_W' if w else 'char_only'}"] = dict(**met(y, X @ b), coef=b.tolist())
for sc, grp in (("LOSO", "Section_mm2"), ("LOUO", "WIRE_uid"), ("LOLO", "Length_m")):
    for qn, qcol in (("Q_measured", None), ("Q_GPR", f"{sc}|Q|GPR geom+phys (script)")):
        for w in (False, True):
            pred = np.full(len(D), np.nan)
            for f in np.unique(D[grp]):
                te = (D[grp] == f).values
                b = lstsq(design(Q[~te], tube[~te], Wp[~te], w), y[~te], rcond=None)[0]
                q = Q[te] if qcol is None else AO[qcol].values[te]
                pred[te] = design(q, tube[te], Wp[te], w) @ b
            K[f"{sc}|{qn}|{'with_W' if w else 'char_only'}"] = met(y, pred)
json.dump(K, open(sys.argv[3], "w"), indent=1)
for k, v in K.items(): print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != 'coef'})
print("coef (in-sample, with W):", np.round(K["insample_with_W"]["coef"], 2))
