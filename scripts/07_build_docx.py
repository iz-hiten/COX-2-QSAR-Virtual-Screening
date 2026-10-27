import copy, re, sys
from pathlib import Path

import docx
from docx.oxml.ns import qn
from PIL import Image

TEMPLATE, FIG, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
from content import TITLE, BODY  # noqa: E402

# renumber citations by order of first appearance (IEEE style) and reorder the reference list
refs_item = next(it for it in BODY if it[0] == "refs")
order = []


def _scan(s):
    for n in re.findall(r"\[(\d+)\]", s):
        if int(n) not in order:
            order.append(int(n))


for it in BODY:
    if it[0] in ("refs", "fig"):
        continue
    if it[0] == "table":
        for row in it[1]:
            for cell in row:
                _scan(cell)
    elif it[0] == "bul":
        for s in it[1]:
            _scan(s)
    else:
        _scan(it[1])
missing = [i for i in range(1, len(refs_item[1]) + 1) if i not in order]
assert not missing, f"uncited references: {missing}"
assert max(order) <= len(refs_item[1]), "citation to a missing reference"
newnum = {old: i + 1 for i, old in enumerate(order)}


def _renum(s):
    return re.sub(r"\[(\d+)\]", lambda m: f"[{newnum[int(m.group(1))]}]", s)


NEW = []
for it in BODY:
    if it[0] == "refs":
        NEW.append(("refs", [refs_item[1][old - 1] for old in order]))
    elif it[0] == "table":
        NEW.append(("table", [[_renum(c) for c in row] for row in it[1]], it[2], it[3]))
    elif it[0] == "bul":
        NEW.append(("bul", [_renum(s) for s in it[1]]))
    elif it[0] == "fig":
        NEW.append(it)
    else:
        NEW.append((it[0], _renum(it[1])))
BODY = NEW

d = docx.Document(TEMPLATE)
body = d.element.body
els = list(body.iterchildren())

# prototypes taken from the template
P = {
    "abstract": els[5], "keywords": els[6], "blank": els[7], "h1": els[8], "p": els[9],
    "h2": els[26], "bul": els[19], "fig": els[43], "cap": els[44], "tcap": els[85],
    "refhead": els[120], "ref": els[121], "table": els[23],
}
P = {k: copy.deepcopy(v) for k, v in P.items()}
sect = els[-1]

# title: keep paragraph formatting, swap the text
title_ts = els[0].findall(".//" + qn("w:t"))
title_ts[0].text = TITLE
for t in title_ts[1:]:
    t.text = ""

# drop the author block (elements 1-3) and everything after it. Element 4 is a blank
# one-column spacer whose section properties separate the title from the two-column body.
for el in els[1:4] + els[5:-1]:
    body.remove(el)


def first_rpr(p):
    for r in p.iter(qn("w:r")):
        if r.find(qn("w:t")) is not None:
            rp = r.find(qn("w:rPr"))
            return copy.deepcopy(rp) if rp is not None else None
    return None


def set_flag(rpr, tag, on):
    for t in (tag, tag + "Cs"):
        el = rpr.find(qn("w:" + t))
        if el is None:
            el = rpr.makeelement(qn("w:" + t), {})
            rpr.insert(0, el)
        el.set(qn("w:val"), "1" if on else "0")


def segments(text):
    out = []
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part:
            continue
        if part.startswith("**"):
            out.append((part[2:-2], True, False))
        elif part.startswith("*"):
            out.append((part[1:-1], False, True))
        else:
            out.append((part, None, False))
    return out


def para(kind, text, base_bold=None):
    p = copy.deepcopy(P[kind])
    rpr = first_rpr(p)
    for child in list(p):
        if child.tag in (qn("w:r"), qn("w:bookmarkStart"), qn("w:bookmarkEnd"), qn("w:hyperlink")):
            p.remove(child)
    for s, bold, ital in segments(text):
        r = p.makeelement(qn("w:r"), {})
        if rpr is not None:
            rp = copy.deepcopy(rpr)
            b = base_bold if bold is None else bold
            if b is not None:
                set_flag(rp, "b", b)
            set_flag(rp, "i", ital)
            hl = rp.find(qn("w:highlight"))
            if hl is not None:
                rp.remove(hl)
            r.append(rp)
        t = r.makeelement(qn("w:t"), {})
        t.text = s
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        r.append(t)
        p.append(r)
    return p


DOCPR = [500]


def figure(path, width_in):
    p = copy.deepcopy(P["fig"])
    jc = p.find(qn("w:pPr")).find(qn("w:jc"))
    jc.set(qn("w:val"), "center")
    rid, _ = d.part.get_or_add_image(str(path))
    w, h = Image.open(path).size
    cx = int(width_in * 914400)
    cy = int(cx * h / w)
    for ext in p.iter("{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent",
                      "{http://schemas.openxmlformats.org/drawingml/2006/main}ext"):
        ext.set("cx", str(cx)); ext.set("cy", str(cy))
    for blip in p.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}blip"):
        blip.set(qn("r:embed"), rid)
    DOCPR[0] += 1
    for dp in p.iter("{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr"):
        dp.set("id", str(DOCPR[0])); dp.set("name", Path(path).name)
    for nv in p.iter("{http://schemas.openxmlformats.org/drawingml/2006/picture}cNvPr"):
        nv.set("name", Path(path).name)
    return p


def table(rows, widths, bold_last=False):
    t = copy.deepcopy(P["table"])
    trs = t.findall(qn("w:tr"))
    head_tr, body_tr = trs[0], trs[1]
    for tr in trs:
        t.remove(tr)
    grid = t.find(qn("w:tblGrid"))
    for g in list(grid):
        grid.remove(g)
    for wd in widths:
        g = grid.makeelement(qn("w:gridCol"), {qn("w:w"): str(wd)})
        grid.append(g)
    tblw = t.find(qn("w:tblPr")).find(qn("w:tblW"))
    tblw.set(qn("w:w"), str(sum(widths)))
    for ri, row in enumerate(rows):
        proto = head_tr if ri == 0 else body_tr
        tr = copy.deepcopy(proto)
        trpr = tr.find(qn("w:trPr"))
        if trpr is not None:
            for hgt in trpr.findall(qn("w:trHeight")):
                trpr.remove(hgt)
        tc_proto = copy.deepcopy(tr.findall(qn("w:tc"))[0])
        for tc in tr.findall(qn("w:tc")):
            tr.remove(tc)
        for ci, txt in enumerate(row):
            tc = copy.deepcopy(tc_proto)
            tcpr = tc.find(qn("w:tcPr"))
            tcw = tcpr.find(qn("w:tcW"))
            if tcw is None:
                tcw = tcpr.makeelement(qn("w:tcW"), {})
                tcpr.insert(0, tcw)
            tcw.set(qn("w:w"), str(widths[ci])); tcw.set(qn("w:type"), "dxa")
            ps = tc.findall(qn("w:p"))
            for extra in ps[1:]:
                tc.remove(extra)
            p = ps[0]
            ppr = p.find(qn("w:pPr"))
            sp = ppr.find(qn("w:spacing"))
            if sp is None:
                sp = ppr.makeelement(qn("w:spacing"), {}); ppr.append(sp)
            sp.set(qn("w:before"), "30"); sp.set(qn("w:after"), "30"); sp.set(qn("w:line"), "228"); sp.set(qn("w:lineRule"), "auto")
            jc = ppr.find(qn("w:jc"))
            if jc is None:
                jc = ppr.makeelement(qn("w:jc"), {}); ppr.append(jc)
            jc.set(qn("w:val"), "left")
            rpr = first_rpr(p)
            for child in list(p):
                if child.tag == qn("w:r"):
                    p.remove(child)
            r = p.makeelement(qn("w:r"), {})
            if rpr is not None:
                rp = copy.deepcopy(rpr)
                set_flag(rp, "b", ri == 0 or (bold_last and ri == len(rows) - 1))
                set_flag(rp, "i", False)
                r.append(rp)
            tt = r.makeelement(qn("w:t"), {}); tt.text = txt
            tt.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            r.append(tt); p.append(r)
            tr.append(tc)
        t.append(tr)
    return t


for item in BODY:
    kind = item[0]
    if kind == "fig":
        _, name, cap, width = item
        sect.addprevious(figure(FIG / name, width))
        sect.addprevious(para("cap", cap))
    elif kind == "table":
        _, rows, widths, bold_last = item
        sect.addprevious(table(rows, widths, bold_last))
    elif kind == "bul":
        for txt in item[1]:
            sect.addprevious(para("bul", txt, base_bold=False))
    elif kind == "refs":
        for txt in item[1]:
            sect.addprevious(para("ref", txt, base_bold=False))
    else:
        bb = False if kind in ("p", "cap") else None
        sect.addprevious(para(kind, item[1], base_bold=bb))

# drop the template's old images
used = {b.get(qn("r:embed")) for b in body.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}blip")}
for rid, rel in list(d.part.rels.items()):
    if rel.reltype.endswith("/image") and rid not in used:
        del d.part.rels[rid]

cp = d.core_properties
cp.title = TITLE; cp.author = ""; cp.last_modified_by = ""; cp.comments = ""; cp.keywords = ""
d.save(OUT)
print("saved", OUT)
