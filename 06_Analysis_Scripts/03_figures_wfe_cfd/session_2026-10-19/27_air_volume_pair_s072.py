# Script from the 2026-10-19 thesis session (transcript command s072).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
from PIL import Image, ImageChops
S="ANSYS Figures/"; OUT="Thesis_Latex/Git/6abd2e69062d299b224d6f5b/figures/05_methodology/cfd_air_volume_pair.png"
def crop(path):
    im=Image.open(path).convert("RGB")
    bg=Image.new("RGB", im.size, (255,255,255))
    diff=ImageChops.difference(im,bg).convert("L").point(lambda v: 255 if v>25 else 0)
    box=diff.getbbox(); pad=12
    box=(max(box[0]-pad,0),max(box[1]-pad,0),min(box[2]+pad,im.width),min(box[3]+pad,im.height))
    return im.crop(box)
a=crop(S+"Screenshot 2026-10-19 155558.png"); b=crop(S+"Screenshot 2026-10-19 155644.png")
H=max(a.height,b.height)
a=a.resize((round(a.width*H/a.height),H), Image.LANCZOS); b=b.resize((round(b.width*H/b.height),H), Image.LANCZOS)
gap=30; W=a.width+gap+b.width
canvas=Image.new("RGB",(W,H),(255,255,255)); canvas.paste(a,(0,(H-a.height)//2)); canvas.paste(b,(a.width+gap,0))
canvas.save(OUT, dpi=(300,300)); print(canvas.size, OUT)