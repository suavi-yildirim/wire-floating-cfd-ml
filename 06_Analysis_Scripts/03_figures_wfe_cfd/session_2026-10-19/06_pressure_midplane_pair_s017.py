# Script from the 2026-10-19 thesis session (transcript command s017).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
import pymupdf, io
from PIL import Image, ImageChops
S='/private/tmp/claude-502/-Users-Suavi-Documents-SY-02-Education-University-Master-Masters-Thesis/b3b7c07f-d28c-4117-ab6d-1952dabe465a/scratchpad/labels/labels.pdf'
doc=pymupdf.open(S); page=doc[0]
ZOOM = 1619/453.5   # px per pt when the 1619 px image is printed at 16 cm
labels=[]
for b in page.get_text('dict')['blocks']:
    for l in b.get('lines',[]):
        txt=''.join(s['text'] for s in l['spans']).strip()
        if txt:
            r=pymupdf.Rect(l['bbox'])
            pix=page.get_pixmap(matrix=pymupdf.Matrix(ZOOM,ZOOM), clip=r, alpha=False)
            labels.append((txt, Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB')))
print([ (t,i.size) for t,i in labels])

def disk_box(im, x0, x1):
    sub = im.crop((x0,0,x1,im.height))
    diff = ImageChops.difference(sub, Image.new('RGB', sub.size,(255,255,255))).convert('L').point(lambda v:255 if v>40 else 0)
    bb = diff.getbbox()
    return (bb[0]+x0, bb[1], bb[2]+x0, bb[3])

for name in ['cfd_velocity_pair.png','cfd_midplane_pair.png']:
    path='figures/07_results/'+name
    im=Image.open(path).convert('RGB')
    W,H=im.size
    L=disk_box(im,0,int(W*0.42)); R=disk_box(im,int(W*0.6),W)
    centers=[(L[0]+L[2])//2, (R[0]+R[2])//2]
    bottom=max(L[3],R[3])
    pad=28
    newH=bottom+pad+sum(i.height for _,i in labels)+10*len(labels)+20
    out=Image.new('RGB',(W,max(H,newH)),(255,255,255))
    out.paste(im,(0,0))
    y=bottom+pad
    for k,(t,lab) in enumerate(labels):
        out.paste(lab,(centers[k]-lab.width//2, y)); y+=lab.height+10
    out=out.crop((0,0,W,y+10))
    out.save(path); print(name, out.size, 'disk centres', centers)