# Script from the 2026-10-19 thesis session (transcript command s076).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
from PIL import Image, ImageChops
import numpy as np
S="ANSYS Figures/"; OUT="Thesis_Latex/Git/6abd2e69062d299b224d6f5b/figures/05_methodology/cfd_air_volume_pair.png"
def crop(path):
    im=Image.open(path).convert("RGB")
    bg=Image.new("RGB", im.size, (255,255,255))
    diff=ImageChops.difference(im,bg).convert("L").point(lambda v: 255 if v>25 else 0)
    x0,y0,x1,y1=diff.getbbox(); pad=12
    return im.crop((max(x0-pad,0),max(y0-pad,0),min(x1+pad,im.width),min(y1+pad,im.height)))
a=crop(S+"Screenshot 2026-10-19 155558.png")  # solid, native scale
b=crop(S+"Screenshot 2026-10-19 155644.png")  # orange air volume + wire, native scale
gap=40; H=max(a.height,b.height); W=a.width+gap+b.width
c=Image.new("RGB",(W,H),(255,255,255)); c.paste(a,(0,0)); c.paste(b,(a.width+gap,0))
c.save(OUT, dpi=(300,300)); print(a.size,b.size,c.size)