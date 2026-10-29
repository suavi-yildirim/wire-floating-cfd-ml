# Script from the 2026-10-19 thesis session (transcript command s173).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
from PIL import Image, ImageDraw, ImageFont
out_dir="Thesis_Latex/Git/6abd2e69062d299b224d6f5b/figures/05_methodology/"
# Controller: no panel letters, width 2000
c=Image.open("Thesis_Figures/WFE_Controller.png").convert("RGB")
c=c.resize((2000,int(c.height*2000/c.width)),Image.LANCZOS)
c.save(out_dir+"wfe_controller.png",optimize=True)
# Hardware: three panels with (a)(b)(c) strip labels, as in the previous version
h=Image.open("Thesis_Figures/WFE_CAD_Photos/WFE_Hardware.png").convert("RGB")
h=h.resize((2000,int(h.height*2000/h.width)),Image.LANCZOS)
w,hh=h.size; strip=40
o=Image.new("RGB",(w,hh+strip),(255,255,255)); o.paste(h,(0,0))
d=ImageDraw.Draw(o)
f=ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf",44)
for i,lab in enumerate(["(a)","(b)","(c)"]):
    d.text((int(w*(2*i+1)/6),hh+strip/2),lab,font=f,fill=(0,0,0),anchor="mm")
o.save(out_dir+"wfe_hardware.png",optimize=True)
print(c.size,o.size)