# Script from the 2026-10-19 thesis session (transcript command s031).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
from PIL import Image
import numpy as np
S='$S'
im=Image.open('ANSYS Figures/S06_wallsheer_WC10_UID23.png').convert('RGB'); w,h=im.size
t=im.crop((int(0.86*w)+75,int(0.80*h)+82,int(0.86*w)+190,int(0.80*h)+195))
a=np.asarray(t).astype(int).copy()
teal=(a[:,:,1]>120)&(a[:,:,2]>120)&(a[:,:,0]<80)
a[teal]=255
t=Image.fromarray(a.astype('uint8'))
t=t.resize((round(t.width*0.7),round(t.height*0.7)), Image.LANCZOS)
t.save(S+'/triad.png'); print(t.size)
src=open(S+'/compose_new_legend.py').read()
src=src.replace("    out.paste(legend, ((W-legend.width)//2, y)); y += legend.height + gap",
"    out.paste(legend, ((W-legend.width)//2, y))\n    triad = Image.open(S + '/triad.png').convert('RGB')\n    out.paste(triad, (W - triad.width - 14, y + (legend.height - triad.height)//2)); y += legend.height + gap")
open(S+'/compose_new_legend.py','w').write(src)
print('triad added', 'triad.png' in src)