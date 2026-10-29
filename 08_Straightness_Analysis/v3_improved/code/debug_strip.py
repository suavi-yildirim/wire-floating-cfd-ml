"""Diagnostic: dump the rule-aligned strip with the tracked path / visibility for one photo. usage: debug_strip.py <jpg cache> <IMG_nnnn> <out.png>"""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire_pipeline as wp
cache, name, out = sys.argv[1:4]
wp.load_templates(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ruler_template_source.jpg"))
im = wp.load_landscape(os.path.join(cache, name + ".jpg"), cache) if False else None
import glob
src = glob.glob(f"../WFE Straightness Photos/{name}.*")[0]
im = wp.load_landscape(src, cache)
p = wp.profile_any(im)
D = wp.DBG
strip = D["strip"]; path = D["path"]; vis = D["vis"]; ps = D["ps_s"]
v = strip.copy()
for q in range(len(path)):
    c = (0, 255, 0) if vis[q] else (0, 0, 255)
    cv2.circle(v, (q, int(path[q])), 1, c, -1)
    if D["is_tape"][q]: v[2, q] = (0, 255, 255)
v = v[::-1]   # up = up
print(name, "hi/lo", D.get("cL"), D.get("cR"), "vis frac", vis.mean(), "n_sel", int(D["sel"].sum()), "full", None if p is None else (p["full"] is not None), "bridged", None if p is None else p.get("bridged_frac"))
cv2.imwrite(out, v)
