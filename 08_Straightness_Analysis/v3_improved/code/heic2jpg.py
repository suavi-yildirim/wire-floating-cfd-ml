"""HEIC -> JPEG via pillow-heif (replaces the ffmpeg-based converter). convert(path, out, max_width)."""
import numpy as np, cv2
from PIL import Image, ImageOps
import pillow_heif
pillow_heif.register_heif_opener()

def convert(path, outp, maxw=None):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    a = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    if maxw and a.shape[1] > maxw:
        s = maxw / a.shape[1]; a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(outp, a)
