# Script from the 2026-10-19 thesis session (transcript command s007).
# Paths are relative to the thesis folder or to the session scratch folder; check them before re-running.
import re
s = open('compose_pairs.py').read()
s = s.replace("d.rectangle([int(0.90 * w), int(0.80 * h), w, h], fill=(255, 255, 255))",
              "d.rectangle([int(0.90 * w), int(0.80 * h), w, h], fill=(255, 255, 255))\n    d.rectangle([int(0.965 * w), 0, w, h], fill=(255, 255, 255))")
new_compose = '''def compose(out, top, bottom, labels, bottom_crop=None, bottom_left_crop=None):
    t = clean(Image.open(top))
    b = clean(Image.open(bottom))
    legend = None
    if bottom_crop is not None:           # one shared legend below both panels
        cut = int(bottom_crop * t.height)
        legend = autocrop(t.crop((0, cut, t.width, t.height)), pad=6)
        t = t.crop((0, 0, t.width, cut))
        b = b.crop((0, 0, b.width, int(bottom_crop * b.height)))
    if bottom_left_crop is not None:      # drop a left legend from the lower panel
        b = b.crop((int(bottom_left_crop * b.width), 0, b.width, b.height))
    a = autocrop(t)
    b = autocrop(b)
    W = max(a.width, b.width, legend.width if legend else 0)
    font = ImageFont.truetype(FONT, max(28, int(W * 0.035)))
    head = int(font.size * 1.9)
    gap = 18
    H = head + a.height + gap + head + b.height + (gap + legend.height if legend else 0)
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    y = 0
    for lab, panel in ((labels[0], a), (labels[1], b)):
        d.text((W // 2, y + head // 2), lab, fill=(0, 0, 0), font=font, anchor='mm')
        y += head
        canvas.paste(panel, ((W - panel.width) // 2, y))
        y += panel.height + gap
    if legend is not None:
        canvas.paste(legend, ((W - legend.width) // 2, y))
    canvas.save(out)
    print(out, canvas.size)
'''
s = re.sub(r"def compose\(.*?\n    print\(out, canvas.size\)\n", new_compose, s, flags=re.S)
s = s.replace("bottom_crop=0.74)", "bottom_crop=0.74)")
s = s.replace("P + '07_results/cfd_wallshear_UID21.png', ('UID23 (long wire)', 'UID21 (short wire)'), bottom_crop=0.8)",
              "P + '07_results/cfd_wallshear_UID21.png', ('UID23 (long wire)', 'UID21 (short wire)'), bottom_crop=0.8)")
open('compose_pairs.py','w').write(s)