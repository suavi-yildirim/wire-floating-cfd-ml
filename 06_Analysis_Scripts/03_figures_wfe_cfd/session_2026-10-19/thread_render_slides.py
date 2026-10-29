import pymupdf, sys
doc = pymupdf.open(sys.argv[1])
for n in [int(x) for x in sys.argv[2].split(",")]:
    pix = doc[n-1].get_pixmap(dpi=110)
    pix.save(f"{sys.argv[3]}_p{n}.png")
print(doc.page_count)
