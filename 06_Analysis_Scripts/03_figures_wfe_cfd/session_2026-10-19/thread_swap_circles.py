import pymupdf, io
from PIL import Image, ImageChops, ImageDraw
P = 'Thesis_Latex/Git/6abd2e69062d299b224d6f5b/figures/07_results/'
FACTOR = 0.9

def white_bbox(img, thr=30):
    diff = ImageChops.difference(img, Image.new('RGB', img.size, (255,255,255))).convert('L').point(lambda v: 255 if v > thr else 0)
    x0, y0, x1, y1 = diff.getbbox(); return img.crop((x0, y0, x1, y1))

def clean(im):
    im = im.convert('RGB'); d = ImageDraw.Draw(im); w, h = im.size
    d.rectangle([int(0.78*w), 0, w, int(0.16*h)], fill=(255,255,255))
    d.rectangle([int(0.90*w), int(0.80*h), w, h], fill=(255,255,255))
    d.rectangle([int(0.965*w), 0, w, h], fill=(255,255,255))
    return im

def parts(pl, pr):
    L = clean(Image.open(pl)); w, h = L.size
    legend = white_bbox(L.crop((0, 0, int(0.2*w), h)))
    dl = white_bbox(L.crop((int(0.2*w), 0, w, h)))
    R = clean(Image.open(pr))
    dr = white_bbox(R.crop((int(0.2*R.width), 0, R.width, R.height)))
    return legend, dl, dr

def sq(img, D):
    s = D / max(img.size); return img.resize((round(img.width*s), round(img.height*s)), Image.LANCZOS), s

def build(legend, left, right):
    D = FACTOR * min(max(left.size), max(right.size))
    left, sl = sq(left, D); right, _ = sq(right, D)
    legend = legend.resize((round(legend.width*sl), round(legend.height*sl)), Image.LANCZOS)
    g = int(0.12*D); H = max(left.height, right.height, legend.height)
    W = left.width + g + legend.width + g + right.width
    c = Image.new('RGB', (W, H), (255,255,255))
    c.paste(left, (0, (H-left.height)//2)); c.paste(legend, (left.width+g, (H-legend.height)//2))
    c.paste(right, (left.width+g+legend.width+g, (H-right.height)//2))
    bl = (0, (H-left.height)//2, left.width, (H-left.height)//2+left.height)
    br = (left.width+g+legend.width+g, (H-right.height)//2, W, (H-right.height)//2+right.height)
    return c, bl, br

doc = pymupdf.open('/private/tmp/claude-502/-Users-Suavi-Documents-SY-02-Education-University-Master-Masters-Thesis/b3b7c07f-d28c-4117-ab6d-1952dabe465a/scratchpad/labels/labels.pdf')
pg = doc[0]
def label_img(txt, zoom):
    for b in pg.get_text('dict')['blocks']:
        for l in b.get('lines', []):
            if ''.join(s['text'] for s in l['spans']).strip() == txt:
                pix = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=pymupdf.Rect(l['bbox']), alpha=False)
                return Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB')

def finish(name, comp, bl, br, left_txt, right_txt, velocity=False):
    W, H = comp.size
    if velocity:
        src = Image.open('ANSYS Figures/velocity_silencer_WC10_UID21.png').convert('RGB')
        c = white_bbox(src.crop((15,195,275,268)))
        c = c.resize((208, round(c.height*208/c.width)), Image.LANCZOS)
        d = ImageDraw.Draw(comp); d.rectangle([int(712*W/1594), 0, int(912*W/1594), int(88*W/1594)], fill=(255,255,255))
        comp.paste(c, (int(728*W/1594), 4))
    zoom = W / 453.5
    la = label_img(left_txt, zoom); lb = label_img(right_txt, zoom)
    cy = max(bl[3], br[3]) + 30
    out = Image.new('RGB', (W, cy+max(la.height, lb.height)+20), (255,255,255))
    out.paste(comp, (0,0))
    out.paste(la, ((bl[0]+bl[2])//2 - la.width//2, cy)); out.paste(lb, ((br[0]+br[2])//2 - lb.width//2, cy))
    out.save(P+name); print(name, out.size)

# UID21 (0.5 m) goes on the left, UID23 (2.0 m) on the right
leg, d23, d21 = parts(P+'cfd_velocity_annulus_UID23.png', P+'cfd_velocity_annulus_UID21.png')
c, bl, br = build(leg, d21, d23)
finish('cfd_velocity_pair.png', c, bl, br, '(a) UID21 (0.5 m)', '(b) UID23 (2.0 m)', velocity=True)
leg, d23, d21 = parts(P+'cfd_pressure_midplane_UID23.png', P+'cfd_pressure_midplane_UID21.png')
c, bl, br = build(leg, d21, d23)
finish('cfd_midplane_pair.png', c, bl, br, '(a) UID21 (0.5 m)', '(b) UID23 (2.0 m)')
