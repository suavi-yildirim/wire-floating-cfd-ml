"""Cell-count statistics of the 40 straight-wire CFD cases.
Input: 02_Report_Data/<case>/cell-count-rfile.out (last iteration row), CFD_Coefficients_per_case.csv
Usage: python3 -I mesh_stats.py <report_dir> <coef_csv> <out_dir>"""
import sys, os, re, numpy as np, pandas as pd
rep, coef, out = sys.argv[1:4]
rows=[]
for case in sorted(os.listdir(rep)):
    f=os.path.join(rep,case,"cell-count-rfile.out")
    if not os.path.isfile(f): continue
    L=[l for l in open(f).read().splitlines()]
    hdr=re.findall(r'"cell-count\((\w+)\)"',L[2])
    data=[l.split() for l in L[3:] if l.strip() and l.split()[0].isdigit()]
    first=np.array(data[0][1:],float); last=np.array(data[-1][1:],float)
    d=dict(ID=case,n_iter=len(data),const=bool(np.all(first==last)))
    d.update({h:int(v) for h,v in zip(hdr,last)}); d["total"]=int(last.sum())
    rows.append(d)
D=pd.DataFrame(rows).fillna(0)
C=pd.read_csv(coef)[["ID","Section_mm2","Tube_mm","Length_m","Imbalance_pct","mdot_CFD","mdot_exp","Used"]]
D=D.merge(C,on="ID")
D["mdot_err_pct"]=100*(D.mdot_CFD-D.mdot_exp)/D.mdot_exp
D.to_csv(os.path.join(out,"cell_counts.csv"),index=False)
pd.set_option("display.width",220)
print("cases",len(D),"| counts constant over iterations:",D.const.all())
print(D.groupby("Tube_mm").total.describe().round(0))
print(D.total.describe().round(0))
print("min/max case:",D.loc[D.total.idxmin(),["ID","total"]].tolist(),D.loc[D.total.idxmax(),["ID","total"]].tolist())
print(D.groupby("Length_m").total.mean().round(0))
print(D.groupby("Section_mm2").total.mean().round(0))
print("zone shares (%) mean:")
zs=[c for c in D.columns if c in("atmosphere","fluid","launching_station","silencer","tube_buffer","tube_start","tube_wc10")]
print((100*D[zs].div(D.total,axis=0)).mean().round(1))
print("spearman total vs |mdot_err|:",D[["total"]].assign(e=D.mdot_err_pct.abs()).corr("spearman").iloc[0,1].round(2))
print("spearman total vs Length:",D[["total","Length_m"]].corr("spearman").iloc[0,1].round(2))
