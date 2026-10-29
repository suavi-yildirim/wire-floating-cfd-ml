# Script from the 2026-10-19 thesis session (transcript command s177).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
from PIL import Image, ImageDraw, ImageFont
out="Thesis_Latex/Git/6abd2e69062d299b224d6f5b/figures/05_methodology/wfe_hardware.png"
h=Image.open("Thesis_Figures/WFE_CAD_Photos/WFE_Hardware.png").convert("RGB")
h=h.resize((2000,int(h.height*2000/h.width)),Image.LANCZOS)
w,hh=h.size; strip=40
o=Image.new("RGB",(w,hh+strip),(255,255,255)); o.paste(h,(0,0))
d=ImageDraw.Draw(o)
f=ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf",44)
# left photo = inductive ring sensors (b); middle = lower tube with launching station (a); right = silencer (c)
for i,lab in enumerate(["(b)","(a)","(c)"]):
    d.text((int(w*(2*i+1)/6),hh+strip/2),lab,font=f,fill=(0,0,0),anchor="mm")
o.save(out,optimize=True)
print(o.size)
f2="5_0_Methodology.tex"
s=open(f2,encoding="utf-8").read()
old="(a)~Inductive ring sensors on the vertical tube; (b)~sensor row at the lower end of the tube; (c)~exhaust silencer at the end of the channel."
new="(a)~Lower end of the tube with the sensor row and the launching station; (b)~inductive ring sensors on the vertical tube; (c)~exhaust silencer at the end of the channel."
assert s.count(old)==1
open(f2,"w",encoding="utf-8").write(s.replace(old,new))
print("ok")