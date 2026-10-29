# Script from the 2026-10-19 thesis session (transcript command s014).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
src = open('/private/tmp/claude-502/-Users-Suavi-Documents-SY-02-Education-University-Master-Masters-Thesis/b3b7c07f-d28c-4117-ab6d-1952dabe465a/scratchpad/compose_pairs.py').read().split('P = G')[0]
import sys; sys.argv=['x','.']; exec(src)
from PIL import Image
P='figures/07_results/'

def parts(path_left_legend, path_right):
    im_l = clean(Image.open(path_left_legend))
    w, h = im_l.size
    legend = autocrop(im_l.crop((0, 0, int(0.2*w), h)), pad=4)
    disk_l = autocrop(im_l.crop((int(0.2*w), 0, w, h)), pad=4)
    im_r = clean(Image.open(path_right))
    disk_r = autocrop(im_r.crop((int(0.2*im_r.width), 0, im_r.width, im_r.height)), pad=4)
    return legend, disk_l, disk_r

def diam(img):
    return max(img.width, img.height)

def square_scale(img, D):
    s = D / diam(img)
    return img.resize((round(img.width*s), round(img.height*s)), Image.LANCZOS), s

def symmetric(out, legend, disk_l, disk_r):
    D = min(diam(disk_l), diam(disk_r))
    disk_l, sl = square_scale(disk_l, D)
    disk_r, sr = square_scale(disk_r, D)
    legend = legend.resize((round(legend.width*sl), round(legend.height*sl)), Image.LANCZOS)
    g = int(0.12 * D)
    H = max(disk_l.height, disk_r.height, legend.height)
    W = disk_l.width + g + legend.width + g + disk_r.width
    canvas = Image.new('RGB', (W, H), (255,255,255))
    canvas.paste(disk_l, (0, (H-disk_l.height)//2))
    canvas.paste(legend, (disk_l.width+g, (H-legend.height)//2))
    canvas.paste(disk_r, (disk_l.width+g+legend.width+g, (H-disk_r.height)//2))
    canvas.save(out); print(out, canvas.size, 'diameters', disk_l.size, disk_r.size)

leg, dl, dr = parts(P+'cfd_velocity_annulus_UID23.png', P+'cfd_velocity_annulus_UID21.png')
symmetric(P+'cfd_velocity_pair.png', leg, dl, dr)
leg, dl, dr = parts(P+'cfd_pressure_midplane_UID23.png', P+'cfd_pressure_midplane_UID21.png')
symmetric(P+'cfd_midplane_pair.png', leg, dl, dr)