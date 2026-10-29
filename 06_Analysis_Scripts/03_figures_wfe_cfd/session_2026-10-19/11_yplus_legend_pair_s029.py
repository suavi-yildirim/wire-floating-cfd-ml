# Script from the 2026-10-19 thesis session (transcript command s029).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
import pymupdf, io, sys
from PIL import Image, ImageChops, ImageDraw
S = sys.argv[1]
P = 'Thesis_Latex/Git/6abd2e69062d299b224d6f5b/figures/'

def white_bbox(img, thr=30):
    diff = ImageChops.difference(img, Image.new('RGB', img.size, (255,255,255))).convert('L').point(lambda v: 255 if v > thr else 0)
    bb = diff.getbbox(); return img.crop(bb) if bb else img

def clean(im):
    im = im.convert('RGB'); d = ImageDraw.Draw(im); w, h = im.size
    d.rectangle([int(0.78*w), 0, w, int(0.16*h)], fill=(255,255,255))
    d.rectangle([int(0.90*w), int(0.80*h), w, h], fill=(255,255,255))
    d.rectangle([int(0.965*w), 0, w, h], fill=(255,255,255))
    return im

def pdf_png(pdf, zoom):
    pg = pymupdf.open(pdf)[0]
    pix = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    return white_bbox(Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB'))

def label_pdf(txt_pdf_page, txt, zoom):
    pg = txt_pdf_page
    for b in pg.get_text('dict')['blocks']:
        for l in b.get('lines', []):
            if ''.join(s['text'] for s in l['spans']).strip() == txt:
                pix = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=pymupdf.Rect(l['bbox']), alpha=False)
                return Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB')

labels_pg = pymupdf.open(S + '/labels2/labels.pdf')[0]

def make(top_path, bot_path, cut, legend_pdf, out_name):
    top = clean(Image.open(top_path)); bot = clean(Image.open(bot_path))
    ct = int(cut*top.height); cb = int(cut*bot.height)
    geo_t = white_bbox(top.crop((0, 0, top.width, ct)))
    geo_b = white_bbox(bot.crop((0, 0, bot.width, cb)))
    W = max(geo_t.width, geo_b.width)
    zoom = W / 453.5
    legend = pdf_png(legend_pdf, zoom)
    gap = 26
    la = label_pdf(labels_pg, '(a) UID23 (2.0 m)', zoom)
    lb = label_pdf(labels_pg, '(b) UID21 (0.5 m)', zoom)
    H = la.height + 10 + geo_t.height + gap + legend.height + gap + lb.height + 10 + geo_b.height
    out = Image.new('RGB', (W, H), (255,255,255))
    y = 0
    out.paste(la, (0, y)); y += la.height + 10
    out.paste(geo_t, ((W-geo_t.width)//2, y)); y += geo_t.height + gap
    out.paste(legend, ((W-legend.width)//2, y)); y += legend.height + gap
    out.paste(lb, (0, y)); y += lb.height + 10
    out.paste(geo_b, ((W-geo_b.width)//2, y))
    out.save(P + out_name); print(out_name, out.size, 'legend', legend.size)

make(P+'07_results/cfd_wallshear_UID23.png', P+'07_results/cfd_wallshear_UID21.png', 0.8, S+'/leg/legend_wallshear.pdf', '07_results/cfd_wallshear_pair.png')
make(P+'05_methodology/cfd_yplus_UID23.png', P+'05_methodology/cfd_yplus_UID21.png', 0.74, S+'/leg/legend_yplus.pdf', '05_methodology/cfd_yplus_pair.png')