#!/usr/bin/env bash
# Regenerate figures, build the .docx, render it to PDF and rasterise the pages for review.
set -e
cd "$(dirname "$0")/.."
PY="/c/Users/KIIT/.waste-training/venv/Scripts/python.exe"
SO="/c/Users/KIIT/.waste-training/lo/program/soffice.com"
OUT="paper/Smart_COX2_QSAR_Research_Paper.docx"

if [ "$1" = "figs" ]; then
  "$PY" scripts/06_figures.py 2>&1 | grep -v -i "warn" || true
fi

(cd scripts && "$PY" 07_build_docx.py ../paper/_template.docx ../paper/figures "../$OUT")

rm -rf paper/_render && mkdir -p paper/_render
"$SO" --headless -env:UserInstallation=file:///C:/Users/KIIT/.waste-training/lo-profile \
      --convert-to pdf --outdir paper/_render "$OUT" >/dev/null 2>&1
"$PY" - <<'EOF' 2>&1 | grep -v -i "warn" || true
import glob
from pathlib import Path
import pymupdf
from PIL import Image

pdf = glob.glob("paper/_render/*.pdf")[0]
doc = pymupdf.open(pdf)
ims = []
for i, p in enumerate(doc):
    p.get_pixmap(dpi=75).save(f"paper/_render/p{i+1:02d}.png")
    p.get_pixmap(dpi=150).save(f"paper/_render/hi{i+1:02d}.png")
    ims.append(Image.open(f"paper/_render/p{i+1:02d}.png"))
w, h = ims[0].size
for k in range(0, len(ims), 6):
    chunk = ims[k:k + 6]
    s = Image.new("RGB", (w * 3, h * 2), "white")
    for i, im in enumerate(chunk):
        s.paste(im, ((i % 3) * w, (i // 3) * h))
    s.save(f"paper/_render/sheet{k // 6 + 1}.png")
print("pages:", len(doc))
EOF
