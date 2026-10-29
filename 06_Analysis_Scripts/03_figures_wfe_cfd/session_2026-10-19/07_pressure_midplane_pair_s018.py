# Script from the 2026-10-19 thesis session (transcript command s018).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
import pymupdf, io
from PIL import Image, ImageChops, ImageDraw
P='Thesis_Latex/Git/6abd2e69062d299b224d6f5b/figures/07_results/'
W_CM_PX = None

def white_bbox(img):
    diff = ImageChops.difference(img, Image.new('RGB', img.size,(255,255,255))).convert('L').point(lambda v:255 if v>30 else 0)
    x0,y0,x1,y1 = diff.getbbox(); return img.crop((x0,y0,x1,y1))

def clean(im):
    im = im.convert('RGB'); d = ImageDraw.Draw(im); w,h = im.size
    d.rectangle([int(0.78*w),0,w,int(0.16*h)], fill=(255,255,255))
    d.rectangle([int(0.90*w),int(0.80*h),w,h], fill=(255,255,255))
    d.rectangle([int(0.965*w),0,w,h], fill=(255,255,255))
    return im

def parts(p_left, p_right):
    L = clean(Image.open(p_left)); w,h = L.size
    legend = white_bbox(L.crop((0,0,int(0.2*w),h)))
    disk_l = white_bbox(L.crop((int(0.2*w),0,w,h)))
    R = clean(Image.open(p_right))
    disk_r = white_bbox(R.crop((int(0.2*R.width),0,R.width,R.height)))
    return legend, disk_l, disk_r

def sq(img, D):
    s = D/max(img.size); return img.resize((round(img.width*s), round(img.height*s)), Image.LANCZOS), s

def build(legend, dl, dr):
    D = min(max(dl.size), max(dr.size))
    dl,sl = sq(dl, D); dr,_ = sq(dr, D)
    legend = legend.resize((round(legend.width*sl), round(legend.height*sl)), Image.LANCZOS)
    g = int(0.12*D)
    H = max(dl.height, dr.height, legend.height)
    W = dl.width+g+legend.width+g+dr.width
    c = Image.new('RGB',(W,H),(255,255,255))
    c.paste(dl,(0,(H-dl.height)//2)); c.paste(legend,(dl.width+g,(H-legend.height)//2))
    c.paste(dr,(dl.width+g+legend.width+g,(H-dr.height)//2))
    return c, (0,(H-dl.height)//2, dl.width, (H-dl.height)//2+dl.height), (dl.width+g+legend.width+g, (H-dr.height)//2, W, (H-dr.height)//2+dr.height)

# wire labels from LaTeX (12 pt Computer Modern), scaled to a 16 cm print width
doc = pymupdf.open('/private/tmp/claude-502/-Users-Suavi-Documents-SY-02-Education-University-Master-Masters-Thesis/b3b7c07f-d28c-4117-ab6d-1952dabe465a/scratchpad/labels/labels.pdf')
pg = doc[0]
def label_img(txt, zoom):
    for b in pg.get_text('dict')['blocks']:
        for l in b.get('lines',[]):
            t=''.join(s['text'] for s in l['spans']).strip()
            if t==txt:
                pix=pg.get_pixmap(matrix=pymupdf.Matrix(zoom,zoom), clip=pymupdf.Rect(l['bbox']), alpha=False)
                return Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB')

def finish(name, comp, dl_box, dr_box, zoom, velocity_patch=False):
    if velocity_patch:
        src = Image.open('ANSYS Figures/velocity_silencer_WC10_UID21.png').convert('RGB')
        crop = src.crop((15,195,275,268))
        crop = white_bbox(crop).resize((208, round(white_bbox(src.crop((15,195,275,268))).height*208/white_bbox(src.crop((15,195,275,268))).width)), Image.LANCZOS)
        d = ImageDraw.Draw(comp); d.rectangle([712,0,912,88], fill=(255,255,255)); comp.paste(crop,(728,4))
    W,H = comp.size
    la = label_img('(a) UID23 (2.0 m)', zoom); lb = label_img('(b) UID21 (0.5 m)', zoom)
    cy = max(dl_box[3], dr_box[3]) + 30
    out = Image.new('RGB',(W, cy+max(la.height,lb.height)+20),(255,255,255))
    out.paste(comp,(0,0))
    ca = (dl_box[0]+dl_box[2])//2; cb = (dr_box[0]+dr_box[2])//2
    out.paste(la,(ca-la.width//2, cy)); out.paste(lb,(cb-lb.width//2, cy))
    out.save(P+name); print(name, out.size)

# zoom: 16 cm = 453.5 pt; the composite is shown at the full text width, so 1 pt = W/453.5 px
ZOOM_FOR = lambda W: W/453.5

# velocity
leg, dl, dr = parts(P+'cfd_velocity_annulus_UID23.png', P+'cfd_velocity_annulus_UID21.png')
comp, bl, br = build(leg, dl, dr)
finish('cfd_velocity_pair.png', comp, bl, br, ZOOM_FOR(comp.width), velocity_patch=True)
# midplane
leg, dl, dr = parts(P+'cfd_pressure_midplane_UID23.png', P+'cfd_pressure_midplane_UID21.png')
comp, bl, br = build(leg, dl, dr)
finish('cfd_midplane_pair.png', comp, bl, br, ZOOM_FOR(comp.width))