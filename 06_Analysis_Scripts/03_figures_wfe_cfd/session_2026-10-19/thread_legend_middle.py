import pymupdf, io, sys
from PIL import Image, ImageChops, ImageDraw
P = 'Thesis_Latex/Git/6abd2e69062d299b224d6f5b/figures/'
S = sys.argv[1]

def white_bbox(img, thr=30):
    diff = ImageChops.difference(img, Image.new('RGB', img.size, (255,255,255))).convert('L').point(lambda v: 255 if v > thr else 0)
    bb = diff.getbbox()
    return img.crop(bb) if bb else img

def clean(im):
    im = im.convert('RGB'); d = ImageDraw.Draw(im); w, h = im.size
    d.rectangle([int(0.78*w), 0, w, int(0.16*h)], fill=(255,255,255))
    d.rectangle([int(0.90*w), int(0.80*h), w, h], fill=(255,255,255))
    d.rectangle([int(0.965*w), 0, w, h], fill=(255,255,255))
    return im

def make(top_path, bottom_path, cut, out_name, a_txt_pdf_index):
    top = clean(Image.open(top_path)); bot = clean(Image.open(bottom_path))
    ct = int(cut*top.height); cb = int(cut*bot.height)
    geo_t = white_bbox(top.crop((0,0,top.width,ct)))
    leg_region = top.crop((0,ct,top.width,top.height))
    ImageDraw.Draw(leg_region).rectangle([int(0.9*leg_region.width), 0, leg_region.width, leg_region.height], fill=(255,255,255))
    legend = white_bbox(leg_region)
    geo_b = white_bbox(bot.crop((0,0,bot.width,cb)))
    W = max(geo_t.width, geo_b.width, legend.width)
    gap = 26
    # labels
    doc = pymupdf.open(S+'/labels2/labels.pdf'); pg = doc[0]
    zoom = W / 453.5
    lab = {}
    for b in pg.get_text('dict')['blocks']:
        for l in b.get('lines', []):
            t = ''.join(s['text'] for s in l['spans']).strip()
            if t:
                pix = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=pymupdf.Rect(l['bbox']), alpha=False)
                lab[t[:3]] = Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB')
    la, lb = lab['(a)'], lab['(b)']
    H = la.height + 10 + geo_t.height + gap + legend.height + gap + lb.height + 10 + geo_b.height
    out = Image.new('RGB', (W, H), (255,255,255))
    y = 0
    out.paste(la, (0, y)); y += la.height + 10
    out.paste(geo_t, ((W-geo_t.width)//2, y)); y += geo_t.height + gap
    out.paste(legend, ((W-legend.width)//2, y)); y += legend.height + gap
    out.paste(lb, (0, y)); y += lb.height + 10
    out.paste(geo_b, ((W-geo_b.width)//2, y))
    out.save(P+out_name); print(out_name, out.size)

make(P+'05_methodology/cfd_yplus_UID23.png', P+'05_methodology/cfd_yplus_UID21.png', 0.74, '05_methodology/cfd_yplus_pair.png', 0)
make(P+'07_results/cfd_wallshear_UID23.png', P+'07_results/cfd_wallshear_UID21.png', 0.8, '07_results/cfd_wallshear_pair.png', 0)
