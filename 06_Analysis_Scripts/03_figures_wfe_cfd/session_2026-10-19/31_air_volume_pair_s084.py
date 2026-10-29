# Script from the 2026-10-19 thesis session (transcript command s084).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
from PIL import Image, ImageChops
S="ANSYS Figures/"; OUT="Thesis_Latex/Git/6abd2e69062d299b224d6f5b/figures/05_methodology/cfd_air_volume_pair.png"
OUT="Thesis_Latex/Git/6abd2e69062d299b224d6f5b/figures/06_implementation/cfd_air_volume_pair.png"
def nonwhite(im):
    bg=Image.new("RGB",im.size,(255,255,255))
    return ImageChops.difference(im,bg).convert("L").point(lambda v:255 if v>25 else 0).getbbox()
A=Image.open(S+"Screenshot 2026-10-19 155558.png").convert("RGB")
B=Image.open(S+"Screenshot 2026-10-19 155644.png").convert("RGB")
solid_top, orange_top = 49, 262
Tb = orange_top + 4
ax0,ay0,ax1,ay1=nonwhite(A); bx0,by0,bx1,by1=nonwhite(B)
pad=6; gap=10
WA=ax1-ax0+2*pad; WB=bx1-bx0+2*pad
canvas=Image.new("RGB",(WA+gap+WB,2000),(255,255,255))
canvas.paste(A.crop((ax0-pad,0,ax1+pad,A.height)),(0,Tb-solid_top))
canvas.paste(B.crop((bx0-pad,0,bx1+pad,B.height)),(WA+gap,Tb-orange_top))
bb=nonwhite(canvas)
out=canvas.crop((bb[0],bb[1],bb[2],bb[3]))
out.save(OUT, dpi=(300,300)); print(out.size)