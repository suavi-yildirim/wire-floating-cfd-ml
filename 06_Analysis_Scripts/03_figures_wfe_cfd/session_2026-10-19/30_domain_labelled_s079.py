# Script from the 2026-10-19 thesis session (transcript command s079).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
from PIL import Image, ImageChops
im=Image.open("cfd_domain_labelled.png").convert("RGB")
W,H=im.size
# scale of preview was 1100/1571 -> map preview coords to original
k=W/1100
# blank the German UI panel (preview box x 195-412, y 188-490)
from PIL import ImageDraw
d=ImageDraw.Draw(im)
d.rectangle([int(195*k),int(188*k),int(413*k),int(490*k)], fill=(255,255,255))
# drop the status text "Simulation 1" at the bottom
d.rectangle([int(560*k),int(718*k),int(640*k),H], fill=(255,255,255))
bg=Image.new("RGB",im.size,(255,255,255))
bb=ImageChops.difference(im,bg).convert("L").point(lambda v:255 if v>25 else 0).getbbox()
pad=20
out=im.crop((max(bb[0]-pad,0),max(bb[1]-pad,0),min(bb[2]+pad,W),min(bb[3]+pad,H)))
out.save("cfd_domain_labelled.png", dpi=(300,300)); print(out.size)
out.thumbnail((1100,1100)); out.save(S+"/dom_prev2.png")