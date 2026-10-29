"""Parameter-free physics baseline: drag-balance flow with CFD-derived coefficients
(no WFE fitting).  Uses piml_ablation.phys() definitions.  Usage: python3 -I physics_cfd_constants.py <ML_Data_All_v8.csv> <CFD_Coefficients_per_case.csv>"""
import sys, numpy as np, pandas as pd
sys.argv=[sys.argv[0]]+sys.argv[1:]
data, cfd = sys.argv[1], sys.argv[2]
T=pd.read_csv(data); C=pd.read_csv(cfd)
R,P,MU=287.058,101325.0,1.7894e-5; RREF=P/(R*273.15)
FER={2.1:2.9,2.4:3.5,2.75:4.4,3.95:5.4,4.5:6.5}
def dfer(od):
    o=od/1000.
    for k,v in FER.items(): o=np.where(abs(od-k)<0.1,v/1000.,o)
    return o
def Q(T,cd,cf,rho):
    d=T.Wire_OD_mm.values/1e3; Dh=T.Hydraulic_Diameter_mm.values/1e3; Af=T.Flow_Area_mm2.values*1e-6
    Ae=np.pi/4*dfer(T.Wire_OD_mm.values)**2; L=T.Length_m.values; W=T.Wire_Weight_N.values
    cd=np.broadcast_to(cd,d.shape)
    v=np.sqrt(2*W/(rho*cd*Ae))
    for _ in range(500):
        Re=rho*v*Dh/MU; Cf=cf*0.079*Re**-0.25
        vn=np.sqrt(2*W/(rho*(cd*Ae+Cf*np.pi*d*L)))
        if np.max(abs(vn-v)/vn)<1e-10: v=vn;break
        v=vn
    return 3600*rho*Af*v/RREF
def met(y,p):
    e=y-p; return dict(R2=round(1-(e**2).sum()/((y-y.mean())**2).sum(),3),RMSE=round(np.sqrt((e**2).mean()),2),MAPE=round(100*np.mean(abs(e/y)),1),bias=round((p-y).mean(),2))
rho=P/(R*T.Temp_K.mean()); y=T.Flow_Nm3h.values
print('CFD table: CDend median all/used/by tube, Cf ratio median')
for lab,m in [('all',C.Used>=0),('used',C.Used==1)]:
    print(lab, C[m].CDend.median().round(2), C[m].CfRatio.median().round(3), C[m].groupby('Tube_mm').CDend.median().round(2).to_dict())
rows=[]
for lab,cd,cf in [('script fixed (CD=0.88, CF=1.145)',0.88,1.145),('Build v8 (CD=1.168, CF=1.145)',1.168,1.145),
                  ('CFD median CDend (used), CF=1.156',C[C.Used==1].CDend.median(),C[C.Used==1].CfRatio.median()),
                  ('CFD median CDend (all 40), CF=1.164',C.CDend.median(),C.CfRatio.median())]:
    q=Q(T,cd,cf,rho); r=met(y,q); r['model']=lab; r['cd']=round(float(cd),2); rows.append(r)
    for sec in [1.5]:
        mk=(T.Section_mm2==sec).values; r2=met(y[mk],q[mk]); r2['model']=lab+' | 1.5 mm2 only'; rows.append(r2)
# CFD closure as function of ferrule-to-tube ratio (power law on 'used' CFD cases, applied to all)
U=C[C.Used==1]; 
print(pd.DataFrame(rows).to_string())
