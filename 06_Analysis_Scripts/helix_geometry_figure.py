"""Schematic of the helical wire geometry (WC10, UID18) and the parameter table (pitch, slope, eccentricity, wetted-length increase)."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Circle
ROOT="/Users/Suavi/Documents_SY/02_Education/University/Master/Masters Thesis"
OUT=f"{ROOT}/03_Experimental_Data/Report_Data/Analysis_Output_Helical"; FIG=f"{ROOT}/09_Figures/Helical"
R_TUBE,R_WIRE,R_FER,L=5.0,1.975,2.7,960.0
h=pd.read_csv(f"{OUT}/helix_runs_vs_baseline.csv")

plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],"font.size":8,"axes.linewidth":0.8})
fig=plt.figure(figsize=(6.8,2.9)); gs=fig.add_gridspec(1,2,width_ratios=[1.7,1])
a=fig.add_subplot(gs[0]); x=np.linspace(0,L,2000)
for rev,c in [(1,"#E4572E"),(3,"#A8325E"),(5,"#4B2A7B")]:
    a.plot(x,1.0*np.cos(2*np.pi*rev*x/L),color=c,lw=1.1,label=f"{rev} rev. (pitch {L/rev:.0f} mm)")
a.axhline(0,color="gray",lw=0.5,ls="-.")
a.axhline(R_TUBE-R_WIRE,color="k",lw=0.8); a.axhline(-(R_TUBE-R_WIRE),color="k",lw=0.8)
a.text(L*0.5,R_TUBE-R_WIRE+0.25,"limit of centreline before wire touches the wall (3.0 mm)",ha="center",fontsize=6.5)
a.set(xlim=(0,L),ylim=(-3.8,3.8),xlabel="Axial position along the wire body [mm]",ylabel="Lateral centreline position [mm]",
      title=r"(a) Centreline projection, $D_h=2.0$ mm ($a=1.0$ mm)")
a.legend(frameon=False,fontsize=6.5,loc="lower right",ncol=3)
b=fig.add_subplot(gs[1]); b.set_aspect("equal")
b.add_patch(Circle((0,0),R_TUBE,fill=False,lw=1.2,color="k"))
b.add_patch(Circle((1.0,0),R_WIRE,fill=False,lw=1.2,color="#4B2A7B"))
b.add_patch(Circle((1.0,0),R_FER,fill=False,lw=0.8,ls="--",color="gray"))
b.add_patch(Circle((0,0),1.0,fill=False,lw=0.8,ls=":",color="k"))     # path of the centreline (coil diameter D_h)
b.plot([0],[0],"k+",ms=6); b.plot([1.0],[0],".",color="#4B2A7B",ms=4)
b.annotate("",xy=(1.0,3.6),xytext=(0,3.6),arrowprops=dict(arrowstyle="<->",lw=0.7)); b.text(0.5,3.9,"$a=D_h/2$",ha="center",fontsize=7)
b.plot([0,0],[3.4,3.8],"k-",lw=0.5); b.plot([1,1],[3.4,3.8],"k-",lw=0.5)
b.text(0,-R_TUBE-0.9,r"tube $\varnothing$10, wire $\varnothing$3.95, ferrule $\varnothing$5.4 (dashed)",ha="center",fontsize=6.2)
b.set(xlim=(-5.5,5.5),ylim=(-6.3,5.5),title="(b) Cross-section (true scale)"); b.axis("off")
plt.tight_layout(); fig.savefig(f"{FIG}/helix_geometry.pdf"); fig.savefig(f"{FIG}/helix_geometry.png",dpi=300)

g=h[["dia_mm","rev","pitch_mm","max_slope_deg","eccentricity","arc_pct_if_Dia_is_diameter"]]
with open(f"{OUT}/helix_geometry_table.tex","w") as f:
    f.write("\\begin{tabular}{rrrrrr}\n\\toprule\n$D_h$ [mm] & $N_{rev}$ & Pitch [mm] & Max.\\ slope [$^{\\circ}$] & Eccentricity $e$ & Wetted-length increase [\\%]\\\\\n\\midrule\n")
    for _,r in g.iterrows():
        f.write(f"{r.dia_mm:.1f} & {int(r.rev)} & {r.pitch_mm:.0f} & {r.max_slope_deg:.2f} & {r.eccentricity:.2f} & {r.arc_pct_if_Dia_is_diameter:.3f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print(open(f"{OUT}/helix_geometry_table.tex").read())
