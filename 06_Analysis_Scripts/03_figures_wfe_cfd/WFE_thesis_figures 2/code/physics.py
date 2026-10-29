"""Derived fluid-dynamic quantities for the 40 CFD cases (straight wires)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from loaders import CFD_XLSX, cfd_outcome, uid_to_geom

P_ATM = 101_325.0      # Pa
R_AIR = 287.05         # J/(kg K)


def mu_air(T_K):
    """Sutherland's law, dynamic viscosity of air [Pa s]."""
    return 1.716e-5 * (T_K / 273.15) ** 1.5 * (273.15 + 110.4) / (T_K + 110.4)


def cfd_table() -> pd.DataFrame:
    s = pd.read_excel(CFD_XLSX, "Summary_Results")
    o = cfd_outcome().rename(columns={"ID": "Simulation_ID"})
    d = s.merge(o, on="Simulation_ID", how="left")
    d["D_mm"] = d["Tube_Diameter[mm]"]
    d["tube"] = d["Wire_Channel"]
    d["uid"] = d["UID"].str[3:].astype(int)
    d["section"] = d["uid"].map(lambda u: uid_to_geom(u)[0])
    d["L"] = d["uid"].map(lambda u: uid_to_geom(u)[1])
    d["d_mm"] = d["Wire_OD[mm]"]
    d["df_mm"] = d["Ferrule_OD[mm]"]
    d["W"] = d["Wire_Weight[N]"]

    # forces (N). Fluent "drag" reported along +y (upwards = lift on the wire)
    d["F_bot"] = d["wire-drag-force(wall_ferrule_bottom)"]
    d["F_top"] = d["wire-drag-force(wall_ferrule_top)"]
    d["F_wall"] = d["wire-drag-force(wall_wire)"]
    d["F_ferrule"] = d["F_bot"] + d["F_top"]
    d["F_D"] = d["total-wire-drag-force"]
    d["FD_over_W"] = d["F_D"] / d["W"]

    # mass flow
    d["mdot_cfd"] = d["mass-flow-inlet"]
    d["mdot_exp"] = d["CFD_Mass_Flow_Target[kg/s]"]
    d["mdot_err"] = (d["mdot_cfd"] - d["mdot_exp"]) / d["mdot_exp"]

    # pressure budget (Pa). WC08: the launching-station report zone already contains the
    # wire tube (tube-pressure-drop is 0), WC10 reports them separately -> merge for comparability
    d["dp_feed"] = d["start-tube-pressure-drop"]
    d["dp_ls_tube"] = d["launching-station-pressure-drop"] + d["tube-pressure-drop"]
    d["dp_ls"] = d["launching-station-pressure-drop"]
    d["dp_tube"] = d["tube-pressure-drop"]
    d["dp_sil"] = d["silencer-pressure-drop"]
    d["p_in"] = d["pressure-inlet"]
    d["dp_rest"] = d["p_in"] - d["dp_feed"] - d["dp_ls_tube"] - d["dp_sil"]

    # wire-region state (mean static pressure over the zone that contains the wire)
    is08 = d["tube"].eq("WC08")
    p_up = np.where(is08, d["launching_station_inlet_pressure"], d["launching-station-outlet-pressure"])
    p_dn = np.where(is08, d["launching-station-outlet-pressure"], d["silencer-inlet-pressure"])
    d["p_wire_gauge"] = 0.5 * (p_up + p_dn)
    T = d["Operating_Temperature[C]"] + 273.15
    d["rho"] = (P_ATM + d["p_wire_gauge"]) / (R_AIR * T)
    d["mu"] = mu_air(T)

    D = d["D_mm"] / 1e3
    dw = d["d_mm"] / 1e3
    d["A_ann"] = np.pi / 4 * (D**2 - dw**2)
    d["D_h"] = D - dw
    d["U"] = d["mdot_cfd"] / (d["rho"] * d["A_ann"])
    d["Re_h"] = d["rho"] * d["U"] * d["D_h"] / d["mu"]
    d["q_dyn"] = 0.5 * d["rho"] * d["U"] ** 2
    d["tau_w"] = d["F_wall"] / (np.pi * dw * d["L"])
    d["Cf"] = d["tau_w"] / d["q_dyn"]
    d["F_D_per_L"] = d["F_D"] / d["L"]
    d["skin_share"] = d["F_wall"] / d["F_D"]
    d["kappa"] = d["df_mm"] / d["D_mm"]           # ferrule-to-tube diameter ratio
    d["A_ferrule"] = np.pi / 4 * (d["df_mm"] / 1e3) ** 2
    d["K_ferrule"] = d["F_ferrule"] / (d["q_dyn"] * d["A_ferrule"])
    return d
