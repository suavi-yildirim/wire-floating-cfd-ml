# Script from the 2026-10-19 thesis session (transcript command s006).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
import sys
from PIL import Image, ImageDraw, ImageFont, ImageChops

G = sys.argv[1]  # thesis figures folder

FONT = '/System/Library/Fonts/Helvetica.ttc'

def clean(im):
    """Paint over the Ansys logo (top right) and the axis triad (bottom right)."""
    im = im.convert('RGB')
    d = ImageDraw.Draw(im)
    w, h = im.size
    d.rectangle([int(0.78 * w), 0, w, int(0.16 * h)], fill=(255, 255, 255))
    d.rectangle([int(0.90 * w), int(0.80 * h), w, h], fill=(255, 255, 255))
    return im

def autocrop(im, pad=12):
    bg = Image.new('RGB', im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg).convert('L').point(lambda v: 255 if v > 30 else 0)
    x0, y0, x1, y1 = diff.getbbox()
    return im.crop((max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad)))

def compose(out, top, bottom, labels, bottom_crop=None, bottom_left_crop=None):
    a = autocrop(clean(Image.open(top)))
    b = clean(Image.open(bottom))
    if bottom_crop is not None:           # drop a bottom legend from the lower panel
        b = b.crop((0, 0, b.width, int(bottom_crop * b.height)))
    if bottom_left_crop is not None:      # drop a left legend from the lower panel
        b = b.crop((int(bottom_left_crop * b.width), 0, b.width, b.height))
    b = autocrop(b)
    W = max(a.width, b.width)
    font = ImageFont.truetype(FONT, max(28, int(W * 0.035)))
    head = int(font.size * 1.9)
    gap = 18
    H = head + a.height + gap + head + b.height
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    y = 0
    for lab, panel in ((labels[0], a), (labels[1], b)):
        d.text((W // 2, y + head // 2), lab, fill=(0, 0, 0), font=font, anchor='mm')
        y += head
        canvas.paste(panel, ((W - panel.width) // 2, y))
        y += panel.height + gap
    canvas.save(out)
    print(out, canvas.size)

P = G + '/'
compose(P + '06_implementation/cfd_mesh_pair.png', P + '06_implementation/cfd_mesh_overview_UID23.png',
        P + '06_implementation/cfd_mesh_closeup_UID23.png', ('(a) Overview', '(b) Close-up at the ferrule'))
compose(P + '05_methodology/cfd_yplus_pair.png', P + '05_methodology/cfd_yplus_UID23.png',
        P + '05_methodology/cfd_yplus_UID21.png', ('UID23 (long wire)', 'UID21 (short wire)'), bottom_crop=0.74)
compose(P + '06_implementation/cfd_residuals_pair.png', P + '06_implementation/cfd_residuals_UID23.png',
        P + '06_implementation/cfd_residuals_UID21.png', ('UID23 (long wire)', 'UID21 (short wire)'), bottom_crop=0.94)
compose(P + '07_results/cfd_velocity_pair.png', P + '07_results/cfd_velocity_annulus_UID23.png',
        P + '07_results/cfd_velocity_annulus_UID21.png', ('UID23 (long wire)', 'UID21 (short wire)'), bottom_left_crop=0.2)
compose(P + '07_results/cfd_midplane_pair.png', P + '07_results/cfd_pressure_midplane_UID23.png',
        P + '07_results/cfd_pressure_midplane_UID21.png', ('UID23 (long wire)', 'UID21 (short wire)'), bottom_left_crop=0.2)
compose(P + '07_results/cfd_wallshear_pair.png', P + '07_results/cfd_wallshear_UID23.png',
        P + '07_results/cfd_wallshear_UID21.png', ('UID23 (long wire)', 'UID21 (short wire)'), bottom_crop=0.8)