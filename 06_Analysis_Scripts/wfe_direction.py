"""
Stabilisation direction for every WFE run - re-derived with the sensor-log method of process_data.py, with ONE fix:
the sensor universe is fixed to S16..S20 (all sensors of the lifting section). The original code only knew sensors that
appear in the run's change log. When the wire stabilises low (e.g. at S17), S18-S20 never fire, are missing from the log,
the set of active sensors comes out empty and the run is labelled 'unknown'.
Output: Thesis_Data/Report_Data/Analysis_Output_Helical/wfe_direction_rederived.csv
"""
import sys, glob, os
import numpy as np, pandas as pd
ROOT="/Users/Suavi/Documents_SY/02_Education/University/Master/Masters Thesis"
D=f"{ROOT}/03_Experimental_Data"
sys.path.insert(0,D)
import process_data as pdm

UNIVERSE=set(range(16,21))

def replay_fixed(df, ref_time):
    if df is None or df.empty or pd.isna(ref_time): return {}
    states={s:True for s in sorted(UNIVERSE|set(df["Sensor_Number"].dropna().astype(int)))}
    rel=df[df["Changed_Time"]<=ref_time].sort_values("Changed_Time")
    for _,r in rel.iterrows():
        if r["Changed_Result"] is not None: states[int(r["Sensor_Number"])]=bool(r["Changed_Result"])
    return states

def active_fixed(df, t0, t1):
    if df is None or df.empty or pd.isna(t0) or pd.isna(t1): return []
    states=replay_fixed(df,t0); act={s for s,v in states.items() if v}
    for _,r in df[(df["Changed_Time"]>=t0)&(df["Changed_Time"]<=t1)].sort_values("Changed_Time").iterrows():
        if r["Changed_Result"] is None: continue
        states[int(r["Sensor_Number"])]=bool(r["Changed_Result"])
        if states[int(r["Sensor_Number"])]: act.add(int(r["Sensor_Number"]))
    return sorted(act)

def event_direction(fp):
    """Independent check: was the last controller event before the final stable phase an INCREASE or a DECREASE?"""
    L=open(fp,encoding='utf-8',errors='ignore').read().split('\n')
    def times(title):
        i=next((k for k,l in enumerate(L) if l.startswith(title)),None); out=[]
        if i is None: return out
        j=i+2
        while j<len(L) and L[j].strip() and L[j][0].isdigit(): out.append(pdm.parse_dt(L[j].split(';')[0])); j+=1
        return [t for t in out if t is not None]
    i=next((k for k,l in enumerate(L) if l.startswith('STABLE PHASES')),None)
    if i is None: return None
    ph=[]; j=i+2
    while j<len(L) and L[j][:1].isdigit(): ph.append(L[j].split(';')); j+=1
    ph=[x for x in ph if float(x[3])>=29000]
    if not ph: return None
    st=pdm.parse_dt(ph[-1][1])
    inc=[t for t in times('INCREASES') if t<=st]; dec=[t for t in times('DECREASES') if t<=st]
    if not inc and not dec: return None
    li=max(inc) if inc else None; ld=max(dec) if dec else None
    if ld is None: return 'wire_rises'
    if li is None: return 'wire_drops'
    return 'wire_drops' if ld>=li else 'wire_rises'

if __name__=="__main__":
    pdm.replay_sensor_states_until=replay_fixed
    pdm.get_active_anytime_in_phase=active_fixed
    rows=[]
    for fp in sorted(glob.glob(f"{D}/WFE_*.csv")):
        w,t,s,tot,dst,dsc=pdm.extract_all_from_file(fp)
        if dst is None: continue
        summ,dfl=pdm.summarize_stable_phases(dst)
        if summ is None: continue
        tr,ss,ps,valid,act=pdm.classify_motion_from_sensors(dfl,dsc)
        rows.append(dict(File=os.path.basename(fp),dir_events=event_direction(fp),dir_fixed=tr,stable_sensor=ss,prev_sensor=ps,stable_valid=valid,active=act))
    N=pd.DataFrame(rows)
    S=pd.read_csv(f"{D}/Stable_Phase_Summary.csv")
    R=S.merge(N,on="File",how="left")
    R.to_csv(f"{D}/Report_Data/Analysis_Output_Helical/wfe_direction_rederived.csv",index=False)
    print(pd.crosstab(R.Final_Trend.fillna('NA'),R.dir_fixed.fillna('NA')))
    print(R[R.Final_Trend=='unknown'].groupby('dir_fixed').stable_sensor.value_counts().unstack(fill_value=0))
    k=R.dropna(subset=['dir_events','dir_fixed'])
    print("\nagreement sensor-log(fixed) vs controller events: %.1f%% (n=%d)"%(100*(k.dir_events==k.dir_fixed).mean(),len(k)))
    print(pd.crosstab(k.dir_fixed,k.dir_events))
    k2=R.dropna(subset=['dir_events']); k2=k2[k2.Final_Trend.isin(['wire_rises','wire_drops'])]
    print("agreement OLD label vs controller events: %.1f%% (n=%d)"%(100*(k2.dir_events==k2.Final_Trend).mean(),len(k2)))
    print("\nstill unknown:",(R.dir_fixed=='unknown').sum(),"| stable_valid False:",(R.stable_valid==False).sum())
