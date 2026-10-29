"""
WFE experimental scatter (UID18, WC10) as noise floor for the CFD helix study - direction-matched.
Stabilisation direction = Final_Trend (wire_rises / wire_drops / unknown). Rising and dropping stabilisations
are NOT pooled: scatter is evaluated inside each direction group.
Outputs: Thesis_Figures/Helical/wfe_vs_cfd_scatter.(pdf|png), Analysis_Output_Helical/wfe_scatter_table.csv, wfe_hysteresis.csv
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
ROOT="/Users/Suavi/Documents_SY/02_Education/University/Master/Masters Thesis"
OUT=f"{ROOT}/03_Experimental_Data/Report_Data/Analysis_Output_Helical"
d=pd.read_csv(f"{ROOT}/03_Experimental_Data/Stable_Phase_Summary.csv").rename(columns={
    'WIRE_uid':'uid','WIRE_instance':'inst','TUBE_diameter (mm)':'tube',
    'Pressure_Stable_Average (mbar)':'p','Flow_Stable_Average (Nm^3/h)':'q','Final_Trend':'dir'})
dr=pd.read_csv(f"{OUT}/wfe_direction_rederived.csv")[['File','dir_fixed','stable_valid']]
d=d.merge(dr,on='File',how='left').drop(columns='dir').rename(columns={'dir_fixed':'dir'})   # direction re-derived (wfe_direction.py)
d=d[(d.System_Stable==True)&(d.tube.isin([8,10]))&(d.stable_valid==True)]

# ---- 1) hysteresis: rises vs drops per wire type ----
rows=[]
for (t,uid),g in d.groupby(['tube','uid']):
    r,dr=g[g.dir=='wire_rises'].p,g[g.dir=='wire_drops'].p
    if len(r)>=2 and len(dr)>=2:
        rows.append(dict(tube=t,uid=uid,n_rise=len(r),n_drop=len(dr),p_rise=r.mean(),p_drop=dr.mean(),ratio=r.mean()/dr.mean()))
hy=pd.DataFrame(rows); hy.to_csv(f"{OUT}/wfe_hysteresis.csv",index=False)
print("rise/drop pressure ratio, median [IQR] per tube:")
print(hy.groupby('tube').ratio.describe()[['count','25%','50%','75%']].round(3))

# ---- 2) scatter inside each direction group (same behaviour only) ----
rows=[]
for (t,uid,dr),g in d[d.dir.isin(['wire_rises','wire_drops'])].groupby(['tube','uid','dir']):
    if len(g)>=3:
        rows.append(dict(tube=t,uid=uid,dir=dr,n=len(g),n_inst=g.inst.nunique(),cv=100*g.p.std()/g.p.mean(),
                         cv_between_inst=100*g.groupby('inst').p.mean().std()/g.p.mean() if g.inst.nunique()>1 else np.nan))
sc=pd.DataFrame(rows)
pool=d.groupby(['tube','uid']).p.agg(lambda s:100*s.std()/s.mean()).rename('cv_pooled').reset_index()
sc.to_csv(f"{OUT}/wfe_scatter_table.csv",index=False)
print("\nmedian CV [%] inside direction groups:"); print(sc.groupby(['tube','dir']).cv.agg(['count','median']).round(2))
print("median CV [%] pooled over directions:"); print(pool.groupby('tube').cv_pooled.median().round(2))

# ---- 3) UID18 / WC10 ----
u=d[(d.uid==18)&(d.tube==10)]
print("\nUID18 WC10 by direction:"); print(u.groupby('dir').agg(n=('p','size'),p=('p','mean'),sd=('p','std'),q=('q','mean')).round(1))

h=pd.read_csv(f"{OUT}/helix_runs_vs_baseline.csv"); lo,hi=h.p_in.min()/100,h.p_in.max()/100
cv_drop=sc[(sc.tube==10)&(sc.dir=='wire_drops')].cv.median(); cv_rise=sc[(sc.tube==10)&(sc.dir=='wire_rises')].cv.median()
cv_pool=pool[pool.tube==10].cv_pooled.median(); hy10=hy[hy.tube==10].ratio.median()
print("\nCFD helix inlet p %.0f-%.0f mbar"%(lo,hi))

plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],"font.size":8,
    "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.8})
fig,ax=plt.subplots(1,2,figsize=(6.8,2.8),gridspec_kw=dict(width_ratios=[1.25,1]))
a=ax[0]; st={'wire_drops':('o','#4B2A7B','dropping (n=%d)'%(u.dir=='wire_drops').sum()),'wire_rises':('^','#E4572E','rising (n=%d)'%(u.dir=='wire_rises').sum())}
Q0=0.008576243953762*3600/1.292248     # CFD mass flow in Nm3/h (density 1.292248 kg/Nm3 at the 0 degC reference of the flow sensor)
qq=np.linspace(u.q.min()*0.97,u.q.max()*1.02,50)
for k,(m,c,lab) in st.items():
    g=u[u.dir==k]; a.scatter(g.q,g.p,marker=m,color=c,s=18,label=lab,zorder=3)
    bexp,aint=np.polyfit(np.log(g.q),np.log(g.p),1); a.plot(qq,np.exp(aint+bexp*np.log(qq)),color=c,lw=0.9,ls='--',zorder=2)
a.scatter(np.full(len(h),Q0),h.p_in/100,marker='D',color='#2E8B57',s=16,label='CFD helix runs (9)',zorder=4)
allr=pd.read_csv(f"{OUT}/helix_all_runs.csv"); pb=float(allr.loc[allr.run=='WC10_UID18','p_in'].iloc[0])/100
a.scatter([Q0],[pb],marker='D',facecolor='white',edgecolor='k',s=26,lw=1.0,label='CFD straight wire (provisional)',zorder=5)
a.axvline(Q0,color='#2E8B57',lw=0.6,ls=':')
a.set(xlabel=r"Stable flow [Nm$^3$/h]",ylabel="Stable levitation pressure [mbar]",title="(a) UID18, WC10: pressure vs flow"); a.legend(frameon=False,fontsize=6.5,loc="upper left")
b=ax[1]
vals=[("Exp. pooled over\ndirections",cv_pool,"#8B6BB1"),("Exp. rising only",cv_rise,"#E4572E"),("Exp. dropping only",cv_drop,"#4B2A7B"),
      ("CFD helix:\nwire drag",100*h.F_wire.std()/h.F_wire.mean(),"#2E8B57")]
b.barh(range(4),[v for _,v,_ in vals],color=[c for *_,c in vals]); b.set_yticks(range(4)); b.set_yticklabels([n for n,*_ in vals]); b.invert_yaxis()
for i,(_,v,_) in enumerate(vals): b.text(v+.1,i,f"{v:.1f} %",va="center",fontsize=7)
b.set(xlabel="Coefficient of variation [%]",title="(b) Scatter (median, WC10 types)")
plt.tight_layout(); fig.savefig(f"{ROOT}/09_Figures/Helical/wfe_vs_cfd_scatter.pdf"); fig.savefig(f"{ROOT}/09_Figures/Helical/wfe_vs_cfd_scatter.png",dpi=300)
print([(n.replace('\n',' '),round(float(v),2)) for n,v,_ in vals], 'hysteresis ratio WC10 median',round(float(hy10),3))
