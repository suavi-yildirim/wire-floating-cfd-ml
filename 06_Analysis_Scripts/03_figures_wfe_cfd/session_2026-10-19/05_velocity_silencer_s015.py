# Script from the 2026-10-19 thesis session (transcript command s015).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
from PIL import Image, ImageDraw, ImageChops
src = Image.open('ANSYS Figures/velocity_silencer_WC10_UID21.png').convert('RGB')
print('source size', src.size)
crop = src.crop((15, 195, 275, 268))
bg = Image.new('RGB', crop.size, (255,255,255))
bbox = ImageChops.difference(crop, bg).getbbox()
crop = crop.crop(bbox)
target_w = 208
s = target_w / crop.width
crop = crop.resize((target_w, round(crop.height*s)), Image.LANCZOS)
print('label size', crop.size)
path = 'Thesis_Latex/Git/6abd2e69062d299b224d6f5b/figures/07_results/cfd_velocity_pair.png'
im = Image.open(path).convert('RGB')
d = ImageDraw.Draw(im)
# wipe the wrapped label above the colour bar
d.rectangle([712, 0, 912, 88], fill=(255,255,255))
im.paste(crop, (728, 4))
im.save(path)
print('saved', im.size)