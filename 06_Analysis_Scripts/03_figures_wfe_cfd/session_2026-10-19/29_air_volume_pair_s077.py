# Script from the 2026-10-19 thesis session (transcript command s077).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
from PIL import Image, ImageChops
import numpy as np
S="ANSYS Figures/"; OUT="Thesis_Latex/Git/6abd2e69062d299b224d6f5b/figures/05_methodology/cfd_air_volume_pair.png"
def nonwhite(im):
    bg=Image.new("RGB", im.size, (255,255,255))
    return ImageChops.difference(im,bg).convert("L").point(lambda v: 255 if v>25 else 0).getbbox()
A=Image.open(S+"Screenshot 2026-10-19 155558.png").convert("RGB")   # solid body
B=Image.open(S+"Screenshot 2026-10-19 155644.png").convert("RGB")   # air volume + wire
# block tops measured earlier: solid 49, orange 262 (same scale: block heights 666 vs 663)
solid_top, orange_top = 49, 262
Tb = orange_top + 12                      # common block-top row on the canvas
ax0,ay0,ax1,ay1 = nonwhite(A); bx0,by0,bx1,by1 = nonwhite(B)
pad=12
W_A=ax1-ax0+2*pad; W_B=bx1-bx0+2*pad; gap=40
H=2000
canvas=Image.new("RGB",(W_A+gap+W_B,H),(255,255,255))
offA = Tb - solid_top                     # solid block top lands on Tb
offB = Tb - orange_top                    # orange block top lands on Tb
canvas.paste(A.crop((ax0-pad,0,ax1+pad,A.height)),(0,offA))
canvas.paste(B.crop((bx0-pad,0,bx1+pad,B.height)),(W_A+gap,offB))
# trim to content
bb=ImageChops.difference(canvas,Image.new("RGB",canvas.size,(255,255,255))).convert("L").point(lambda v:255 if v>25 else 0).getbbox()
out=canvas.crop((bb[0]-pad,bb[1]-pad,bb[2]+pad,bb[3]+pad))
out.save(OUT, dpi=(300,300)); print(out.size)
# report resulting block tops
print("solid block top on canvas", solid_top+offA-(bb[1]-pad), "orange block top", orange_top+offB-(bb[1]-pad))