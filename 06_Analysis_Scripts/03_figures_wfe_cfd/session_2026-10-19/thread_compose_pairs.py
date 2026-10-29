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
    d.rectangle([int(0.965 * w), 0, w, h], fill=(255, 255, 255))
    return im

def autocrop(im, pad=12):
    bg = Image.new('RGB', im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg).convert('L').point(lambda v: 255 if v > 30 else 0)
    x0, y0, x1, y1 = diff.getbbox()
    return im.crop((max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad)))

def compose(out, top, bottom, labels, bottom_crop=None, bottom_left_crop=None):
    t = clean(Image.open(top))
    b = clean(Image.open(bottom))
    legend = None
    if bottom_crop is not None:           # one shared legend below both panels
        cut = int(bottom_crop * t.height)
        legend = autocrop(t.crop((0, cut, t.width, t.height)), pad=6)
        t = t.crop((0, 0, t.width, cut))
        b = b.crop((0, 0, b.width, int(bottom_crop * b.height)))
    if bottom_left_crop is not None:      # drop a left legend from the lower panel
        b = b.crop((int(bottom_left_crop * b.width), 0, b.width, b.height))
    a = autocrop(t)
    b = autocrop(b)
    W = max(a.width, b.width, legend.width if legend else 0)
    font = ImageFont.truetype(FONT, max(28, int(W * 0.035)))
    head = int(font.size * 1.9) if labels[0] else 0
    gap = 18
    H = head + a.height + gap + head + b.height + (gap + legend.height if legend else 0)
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    y = 0
    for lab, panel in ((labels[0], a), (labels[1], b)):
        if lab:
            d.text((W // 2, y + head // 2), lab, fill=(0, 0, 0), font=font, anchor='mm')
        y += head
        canvas.paste(panel, ((W - panel.width) // 2, y))
        y += panel.height + gap
    if legend is not None:
        canvas.paste(legend, ((W - legend.width) // 2, y))
    canvas.save(out)
    print(out, canvas.size)

P = G + '/'
compose(P + '06_implementation/cfd_mesh_pair.png', P + '06_implementation/cfd_mesh_overview_UID23.png',
        P + '06_implementation/cfd_mesh_closeup_UID23.png', (None, None))
compose(P + '05_methodology/cfd_yplus_pair.png', P + '05_methodology/cfd_yplus_UID23.png',
        P + '05_methodology/cfd_yplus_UID21.png', (None, None), bottom_crop=0.74)
compose(P + '06_implementation/cfd_residuals_pair.png', P + '06_implementation/cfd_residuals_UID23.png',
        P + '06_implementation/cfd_residuals_UID21.png', (None, None), bottom_crop=0.94)
compose(P + '07_results/cfd_velocity_pair.png', P + '07_results/cfd_velocity_annulus_UID23.png',
        P + '07_results/cfd_velocity_annulus_UID21.png', (None, None), bottom_left_crop=0.2)
compose(P + '07_results/cfd_midplane_pair.png', P + '07_results/cfd_pressure_midplane_UID23.png',
        P + '07_results/cfd_pressure_midplane_UID21.png', (None, None), bottom_left_crop=0.2)
compose(P + '07_results/cfd_wallshear_pair.png', P + '07_results/cfd_wallshear_UID23.png',
        P + '07_results/cfd_wallshear_UID21.png', (None, None), bottom_crop=0.8)
