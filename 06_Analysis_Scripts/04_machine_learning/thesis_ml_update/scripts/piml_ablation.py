"""Grouped-CV ablation of the physics-informed features (Python replication of
CrossVal_Physics_Calibration_Ablation.m, WFE v8 dataset).

Question answered: does the physics feature add anything beyond the geometry
features?  Same folds, same rows, same GPR (ARD squared exponential), only the
predictor list changes.

Usage: python3 -I piml_ablation.py <ML_Data_All_v8.csv> <out_dir>
"""
import sys, warnings, json
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
data_csv, out_dir = sys.argv[1], sys.argv[2]
T0 = pd.read_csv(data_csv)
n = len(T0)

R_AIR, P_AMB, MU = 287.058, 101325.0, 1.7894e-5
RHO_REF = P_AMB / (R_AIR * 273.15)
CD_LO, CF_LO, CF_START = 0.40, 0.80, 1.145
FERR = {2.1: 2.9, 2.4: 3.5, 2.75: 4.4, 3.95: 5.4, 4.5: 6.5}


def d_ferrule(od):
    out = od / 1000.0
    for k, v in FERR.items():
        out = np.where(np.abs(od - k) < 0.1, v / 1000.0, out)
    return out


def phys(T, rho, cd, cf):
    d = T.Wire_OD_mm.values / 1000; Dh = T.Hydraulic_Diameter_mm.values / 1000
    Dt = T.Tube_D_mm.values / 1000; Af = T.Flow_Area_mm2.values * 1e-6
    At = np.pi / 4 * Dt**2
    Aend = np.pi / 4 * d_ferrule(T.Wire_OD_mm.values)**2
    L = T.Length_m.values; W = T.Wire_Weight_N.values
    v = np.sqrt(2 * W / (rho * cd * Aend))
    for _ in range(500):
        Re = rho * v * Dh / MU
        Cf = cf * 0.079 * Re**-0.25
        vn = np.sqrt(2 * W / (rho * (cd * Aend + Cf * np.pi * d * L)))
        if np.max(np.abs(vn - v) / vn) < 1e-10:
            v = vn; break
        v = vn
    Re = rho * v * Dh / MU
    f = 0.316 * Re**-0.25
    dp = P_AMB + f * (L / Dh) * rho * v**2 / 2
    Q = 3600 * rho * Af * v / RHO_REF
    dyn = 0.5 * rho * (Af * v / At)**2
    return Q, dp, dyn


def calib(T, rho):
    y = T.Flow_Nm3h.values

    def obj(p):
        Q, _, _ = phys(T, rho, max(p[0], CD_LO), max(p[1], CF_LO))
        return np.sum(((y - Q) / y)**2)
    r = minimize(obj, [1.0, CF_START], method="Nelder-Mead",
                 options=dict(xatol=1e-6, fatol=1e-10, maxiter=400))
    return max(r.x[0], CD_LO), max(r.x[1], CF_LO)


def gpr_fit_predict(Xtr, ytr, Xte, log=False):
    """GPR with ARD squared-exponential kernel on the *distinct input vectors*.
    All features are deterministic functions of (wire type, tube), so replicate runs
    share one input vector.  Replicates are collapsed to their mean and enter through
    a known heteroscedastic noise term alpha_i = s2_within / n_i (pooled within-vector
    variance), which is the exact GP posterior for replicated inputs up to the
    estimation of s2_within.  Cost: 40 instead of ~700 points per fit."""
    sc = StandardScaler().fit(Xtr)
    Z = sc.transform(Xtr)
    key = pd.DataFrame(np.round(Z, 8))
    gid = key.groupby(list(key.columns)).ngroup().values
    ng = gid.max() + 1
    y = np.log(ytr) if log else ytr
    mu, sd = y.mean(), y.std()
    yn = (y - mu) / sd
    cnt = np.bincount(gid, minlength=ng)
    ym = np.bincount(gid, weights=yn, minlength=ng) / cnt
    Zg = np.vstack([Z[gid == g][0] for g in range(ng)])
    within = np.sum((yn - ym[gid])**2) / max(len(yn) - ng, 1)
    alpha = within / cnt + 1e-6
    k = C(1.0, (1e-2, 1e2)) * RBF(np.ones(Z.shape[1]), (1e-2, 1e3))
    g = GPR(k, alpha=alpha, normalize_y=False, n_restarts_optimizer=6, random_state=0).fit(Zg, ym)
    p = g.predict(sc.transform(Xte)) * sd + mu
    return np.exp(p) if log else p


GEOM_Q = ["Section_mm2", "Ferrule_to_Tube_Ratio", "Blockage_Ratio", "Length_over_Dh"]
GEOM_P = ["Section_mm2", "Tube_D_mm", "Length_m", "Blockage_Ratio", "Length_over_Dh", "Ferrule_to_Tube_Ratio"]
# the weight is a legitimate a-priori input; geometry-only ablation without any physics
GEOM_QW = GEOM_Q + ["Wire_Weight_N"]
GEOM_PW = GEOM_P + ["Wire_Weight_N"]

schemes = {"LOSO": "Section_mm2", "LOUO": "WIRE_uid", "LOLO": "Length_m"}
variants_Q = {
    "geom": GEOM_Q,
    "geom+W": GEOM_QW,
    "geom+phys (script)": GEOM_Q + ["Flow_Phys_Nm3h"],
}
variants_P = {
    "geom": GEOM_P,
    "geom+W": GEOM_PW,
    "geom+phys (script)": GEOM_P + ["Pressure_Phys_Darcy_Pa", "Sys_Dynamic_Pa"],
    "geom+Qphys only": GEOM_P + ["Flow_Phys_Nm3h"],
}

oof = {s: {} for s in schemes}
logs = []
for sc, grp in schemes.items():
    for f in np.unique(T0[grp]):
        te = (T0[grp] == f).values
        Tr, Te = T0[~te].copy(), T0[te].copy()
        rho = P_AMB / (R_AIR * Tr.Temp_K.mean())
        cd, cf = calib(Tr, rho)
        for T_ in (Tr, Te):
            Q, dp, dyn = phys(T_, rho, cd, cf)
            T_["Flow_Phys_Nm3h"], T_["Pressure_Phys_Darcy_Pa"], T_["Sys_Dynamic_Pa"] = Q, dp, dyn
        logs.append(dict(scheme=sc, fold=float(f), cd=cd, cf=cf, n=int(te.sum())))
        # physics-only
        oof[sc].setdefault("Q|physics (calibrated)", np.full(n, np.nan))[te] = Te.Flow_Phys_Nm3h.values
        oof[sc].setdefault("P|physics (Darcy)", np.full(n, np.nan))[te] = Te.Pressure_Phys_Darcy_Pa.values - P_AMB  # gauge basis: Pressure_Pa is gauge (Build_MAT_Dataset.m l.184)
        for name, cols in variants_Q.items():
            oof[sc].setdefault("Q|GPR " + name, np.full(n, np.nan))[te] = gpr_fit_predict(Tr[cols].values, Tr.Flow_Nm3h.values, Te[cols].values)
        for name, cols in variants_P.items():
            oof[sc].setdefault("P|GPR " + name, np.full(n, np.nan))[te] = gpr_fit_predict(Tr[cols].values, Tr.Pressure_Pa.values, Te[cols].values)
        # linear surrogates for reference
        for name, cols in {"lin geom": GEOM_Q, "lin geom+phys": GEOM_Q + ["Flow_Phys_Nm3h"]}.items():
            m = LinearRegression().fit(Tr[cols].values, Tr.Flow_Nm3h.values)
            oof[sc].setdefault("Q|" + name, np.full(n, np.nan))[te] = m.predict(Te[cols].values)
        for name, cols in {"lin geom": GEOM_P, "lin geom+phys": variants_P["geom+phys (script)"]}.items():
            m = LinearRegression().fit(Tr[cols].values, Tr.Pressure_Pa.values)
            oof[sc].setdefault("P|" + name, np.full(n, np.nan))[te] = m.predict(Te[cols].values)
    print("done", sc, flush=True)


def metrics(y, p):
    e = y - p
    return dict(R2=1 - np.sum(e**2) / np.sum((y - y.mean())**2), RMSE=np.sqrt(np.mean(e**2)),
                MAE=np.mean(np.abs(e)), MAPE=100 * np.mean(np.abs(e / y)), MedAPE=100 * np.median(np.abs(e / y)))


rows = []
for sc in schemes:
    for key, p in oof[sc].items():
        tgt, name = key.split("|")
        y = T0.Flow_Nm3h.values if tgt == "Q" else T0.Pressure_Pa.values
        rows.append(dict(scheme=sc, target=tgt, model=name, **metrics(y, p)))
res = pd.DataFrame(rows)
res.to_csv(f"{out_dir}/ablation_metrics.csv", index=False)
pd.DataFrame(logs).to_csv(f"{out_dir}/fold_coefficients.csv", index=False)
out = T0[["Source_File", "WIRE_uid", "Section_mm2", "Tube_D_mm", "Length_m", "Flow_Nm3h", "Pressure_Pa"]].copy()
for sc in schemes:
    for key, p in oof[sc].items():
        out[f"{sc}|{key}"] = p
out.to_csv(f"{out_dir}/ablation_oof.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 200)
print(res.round(3).to_string())
