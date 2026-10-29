# Script from the 2026-10-19 thesis session (transcript command s013).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
src = open('/private/tmp/claude-502/-Users-Suavi-Documents-SY-02-Education-University-Master-Masters-Thesis/b3b7c07f-d28c-4117-ab6d-1952dabe465a/scratchpad/compose_pairs.py').read().split('P = G')[0]
import sys; sys.argv=['x','.']; exec(src)
from PIL import Image
P='figures/07_results/'
a = autocrop(clean(Image.open(P+'cfd_velocity_annulus_UID23.png')))
b = clean(Image.open(P+'cfd_velocity_annulus_UID21.png'))
b = autocrop(b.crop((int(0.2*b.width),0,b.width,b.height)))
gap = 40
H = max(a.height,b.height)
canvas = Image.new('RGB',(a.width+gap+b.width,H),(255,255,255))
canvas.paste(a,(0,(H-a.height)//2)); canvas.paste(b,(a.width+gap,(H-b.height)//2))
canvas.save(P+'cfd_velocity_pair.png'); print(canvas.size)