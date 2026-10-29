# Script from the 2026-10-19 thesis session (transcript command s034).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
from PIL import Image
import numpy as np
S='$S'
src=Image.open('/Users/Suavi/Documents_SY/02_Education/University/Master/Masters Thesis/05_CFD_Simulation/ANSYS_Figures/S06_wallsheer_WC10_UID23.png').convert('RGB'); w,h=src.size
t=src.crop((int(0.86*w)+75,int(0.80*h)+82,int(0.86*w)+190,int(0.80*h)+195))
a=np.asarray(t).astype(int).copy()
a[a.min(axis=2)>200]=255
a[(a[:,:,1]>120)&(a[:,:,2]>120)&(a[:,:,0]<80)]=255
t=Image.fromarray(a.astype('uint8')).resize((round(115*1.6), round(115*1.6)), Image.LANCZOS)
t.save(S+'/triad.png'); print('triad', t.size)
p=S+'/leg/compose_new_legend.py'
s=open(p).read()
old="    triad = Image.open(S + '/triad.png').convert('RGB')\n    out.paste(triad, (W - triad.width - 14, y + (legend.height - triad.height)//2)); y += legend.height + gap"
assert s.count(old)==1
new='''    triad = Image.open(S + '/triad.png').convert('RGB')
    arr = np.asarray(legend).astype(int)
    sat = (arr.max(axis=2) - arr.min(axis=2)) > 60
    rows = sat.sum(axis=1); ys = np.where(rows > 0.5*rows.max())[0]
    cols = np.where(sat[ys.min():ys.max()+1].mean(axis=0) > 0.5)[0]
    bar_cy = (ys.min() + ys.max() + 1)//2; bar_x1 = cols.max()
    out.paste(triad, (min(bar_x1 + 60, W - triad.width - 10), y + bar_cy - triad.height//2)); y += legend.height + gap'''
s=s.replace(old,new).replace('from PIL import Image, ImageChops, ImageDraw','from PIL import Image, ImageChops, ImageDraw\nimport numpy as np')
open(p,'w').write(s); print('patched')