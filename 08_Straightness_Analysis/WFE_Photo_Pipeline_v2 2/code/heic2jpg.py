"""Minimal HEIC (iPhone, grid of HEVC tiles) -> JPEG converter using system ffmpeg.
Parses ISOBMFF meta (pitm, iinf, iloc, iref/dimg, iprp/ipco/ipma), writes tiles as an
Annex-B HEVC stream, decodes with ffmpeg, stitches the grid, applies irot/imir.
Usage: heic2jpg.py in.heic out.jpg [max_width]"""
import sys, struct, subprocess, tempfile, os
import numpy as np, cv2

def boxes(b, off, end):
    while off < end:
        size, typ = struct.unpack(">I4s", b[off:off + 8]); hdr = 8
        if size == 1: size = struct.unpack(">Q", b[off + 8:off + 16])[0]; hdr = 16
        elif size == 0: size = end - off
        yield typ.decode("latin-1"), off + hdr, off + size
        off += size

def child(b, s, e, name, full=False):
    s2 = s + (4 if full else 0)
    for t, cs, ce in boxes(b, s2, e):
        if t == name: return cs, ce
    return None

def rd(b, o, n):
    return int.from_bytes(b[o:o + n], "big") if n else 0

def parse(path):
    b = open(path, "rb").read()
    meta = next((cs, ce) for t, cs, ce in boxes(b, 0, len(b)) if t == "meta")
    ms, me = meta[0] + 4, meta[1]                       # full box
    sub = {t: (cs, ce) for t, cs, ce in boxes(b, ms, me)}
    # pitm
    s, _ = sub["pitm"]; v = b[s]; primary = rd(b, s + 4, 2 if v == 0 else 4)
    # iinf
    s, e = sub["iinf"]; v = b[s]; p = s + 4 + (2 if v == 0 else 4)
    types = {}
    for t, cs, ce in boxes(b, p, e):
        v2 = b[cs]; q = cs + 4
        iid = rd(b, q, 2 if v2 < 3 else 4); q += 2 if v2 < 3 else 4
        q += 2  # protection index
        types[iid] = b[q:q + 4].decode("latin-1")
    # iloc
    s, e = sub["iloc"]; v = b[s]; p = s + 4
    os_, ls = b[p] >> 4, b[p] & 15; bs = b[p + 1] >> 4; isz = (b[p + 1] & 15) if v in (1, 2) else 0; p += 2
    n = rd(b, p, 2 if v < 2 else 4); p += 2 if v < 2 else 4
    loc = {}
    for _ in range(n):
        iid = rd(b, p, 2 if v < 2 else 4); p += 2 if v < 2 else 4
        cm = 0
        if v in (1, 2): cm = b[p + 1] & 15; p += 2
        p += 2  # data ref idx
        base = rd(b, p, bs); p += bs
        ec = rd(b, p, 2); p += 2
        ext = []
        for _ in range(ec):
            if v in (1, 2) and isz: p += isz
            eo = rd(b, p, os_); p += os_; el = rd(b, p, ls); p += ls
            ext.append((base + eo, el))
        loc[iid] = (cm, ext)
    def data(iid):
        cm, ext = loc[iid]
        if cm == 1:  # idat
            s, e = sub["iprp"] if False else sub["idat"]
            return b"".join(b[s + o:s + o + l] for o, l in ext)
        return b"".join(b[o:o + l] for o, l in ext)
    # iref dimg
    dimg = {}
    if "iref" in sub:
        s, e = sub["iref"]; v = b[s]; w = 2 if v == 0 else 4
        for t, cs, ce in boxes(b, s + 4, e):
            fid = rd(b, cs, w); c = rd(b, cs + w, 2)
            ids = [rd(b, cs + w + 2 + k * w, w) for k in range(c)]
            if t == "dimg": dimg[fid] = ids
    # iprp
    s, e = sub["iprp"]
    ipco = child(b, s, e, "ipco"); ipma = child(b, s, e, "ipma")
    props = [(t, cs, ce) for t, cs, ce in boxes(b, ipco[0], ipco[1])]
    ps, pe = ipma; v = b[ps]; fl = rd(b, ps + 1, 3); p = ps + 4
    n = rd(b, p, 4); p += 4; assoc = {}
    for _ in range(n):
        iid = rd(b, p, 2 if v < 1 else 4); p += 2 if v < 1 else 4
        c = b[p]; p += 1; lst = []
        for _ in range(c):
            if fl & 1: x = rd(b, p, 2); p += 2; idx = x & 0x7FFF
            else: x = b[p]; p += 1; idx = x & 0x7F
            lst.append(idx)
        assoc[iid] = lst
    def prop(iid, name):
        for idx in assoc.get(iid, []):
            t, cs, ce = props[idx - 1]
            if t == name: return cs, ce
    return b, primary, types, data, dimg, prop

def hvcc_annexb(b, cs, ce):
    p = cs + 22; na = b[p]; p += 1; out = b""
    for _ in range(na):
        p += 1; nn = rd(b, p, 2); p += 2
        for _ in range(nn):
            l = rd(b, p, 2); p += 2; out += b"\x00\x00\x00\x01" + b[p:p + l]; p += l
    return out

def to_annexb(d):
    out, p = b"", 0
    while p < len(d):
        l = rd(d, p, 4); p += 4; out += b"\x00\x00\x00\x01" + d[p:p + l]; p += l
    return out

def convert(path, outp, maxw=None):
    b, primary, types, data, dimg, prop = parse(path)
    if types[primary] == "grid":
        g = data(primary); flags = g[1]; rows, cols = g[2] + 1, g[3] + 1
        fw = 4 if flags & 1 else 2
        W, H = rd(g, 4, fw), rd(g, 4 + fw, fw)
        tiles = dimg[primary]
    else:
        rows = cols = 1; tiles = [primary]; W = H = None
    cs, ce = prop(tiles[0], "hvcC"); ps = hvcc_annexb(b, cs, ce)
    stream = b"".join(ps + to_annexb(data(t)) for t in tiles)
    with tempfile.TemporaryDirectory() as td:
        hp = os.path.join(td, "t.hevc"); open(hp, "wb").write(stream)
        r = subprocess.run(["ffmpeg", "-v", "error", "-f", "hevc", "-i", hp, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           capture_output=True)
        isp = prop(tiles[0], "ispe"); tw, th = rd(b, isp[0] + 4, 4), rd(b, isp[0] + 8, 4)
        fr = np.frombuffer(r.stdout, np.uint8).reshape(-1, th, tw, 3)
    canvas = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for k in range(min(len(fr), rows * cols)):
        rr, cc = divmod(k, cols); canvas[rr * th:(rr + 1) * th, cc * tw:(cc + 1) * tw] = fr[k]
    if W: canvas = canvas[:H, :W]
    ir = prop(primary, "irot")
    if ir:
        a = b[ir[0]] & 3
        for _ in range(a): canvas = cv2.rotate(canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)
    im = prop(primary, "imir")
    if im: canvas = cv2.flip(canvas, 1 if (b[im[0]] & 1) else 0)
    if maxw and canvas.shape[1] > maxw:
        s = maxw / canvas.shape[1]; canvas = cv2.resize(canvas, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(outp, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return canvas.shape

if __name__ == "__main__":
    print(convert(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None))
