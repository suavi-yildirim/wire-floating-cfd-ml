# Script from the 2026-10-19 thesis session (transcript command s027).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
from PIL import Image, ImageChops, ImageDraw
import numpy as np
S='$S'
P='Thesis_Latex/Git/6abd2e69062d299b224d6f5b/figures/'
def legend_crop(path, cut, is_top=True):
    im=Image.open(path).convert('RGB'); w,h=im.size
    leg=im.crop((0,int(cut*h),w,h))
    ImageDraw.Draw(leg).rectangle([int(0.9*leg.width),0,leg.width,leg.height],fill=(255,255,255))
    return leg
def bar_of(leg):
    a=np.asarray(leg).astype(int)
    sat=(a.max(axis=2)-a.min(axis=2))>60
    rows=sat.sum(axis=1); thr=0.5*rows.max()
    ys=np.where(rows>thr)[0]; y0,y1=ys.min(),ys.max()+1
    cols=sat[y0:y1].mean(axis=0)>0.5
    xs=np.where(cols)[0]; x0,x1=xs.min(),xs.max()+1
    return leg.crop((x0,y0,x1,y1))
for name,path,cut in [('wallshear',P+'07_results/cfd_wallshear_UID23.png',0.8),('yplus',P+'05_methodology/cfd_yplus_UID23.png',0.74)]:
    leg=legend_crop(path,cut)
    bar=bar_of(leg); bar.save(f'{S}/leg/{name}_bar.png'); print(name,'bar',bar.size)