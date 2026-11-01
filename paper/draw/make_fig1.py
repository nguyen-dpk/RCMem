"""Figure 1: writes draw/fig1_teaser.drawio and figures/fig1_teaser.svg from one layout, so they cannot drift apart.

Run from paper/: python draw/make_fig1.py && rsvg-convert -f pdf -o figures/fig1_teaser.pdf figures/fig1_teaser.svg
The SVG carries the diagram in its `content` attribute, the way draw.io's editable SVG does, so it reopens in draw.io.
shortcut: the SVG is written here, not by draw.io's exporter; re-export from draw.io if its own rendering is wanted.

Real case, subtask 00877-4ok3usBNeis_0_2. MemoryExplorer: run p2_E0_fix_s77 (commits to chair_8, region _0; fails).
RCMem: analysis/p3_nav.py --exp RPcu --seed 77 --viz, zero-shot, decisions 1-6; assets are its viz/ output.
me_*.jpg: draw/render_fig1_me.py, the 5 memories retrieved at its last decision, re-rendered from the recorded poses.
"""
import base64, os
from xml.sax.saxutils import escape, quoteattr

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 500, 462
RED, GREEN, INK, GREY = "#C0392B", "#1E8449", "#222222", "#555555"
cells, svg = [], []


def next_id():
    return str(len(cells) + 2)


def b64(name):
    return base64.b64encode(open(os.path.join(HERE, "assets", name), "rb").read()).decode()


def box(x, y, w, h, color, width=2, fill="none", rx=0):
    round_style = f"rounded=1;absoluteArcSize=1;arcSize={2 * rx};" if rx else "rounded=0;"
    cells.append(f'<mxCell id="{next_id()}" value="" style="{round_style}fillColor={fill};strokeColor={color};strokeWidth={width};" '
                 f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{color}" stroke-width="{width}"/>')


def img(x, y, w, h, name, border=None):
    data = b64(name)
    # draw.io's style string uses ';' as a separator, so its data URIs have no ';base64'
    cells.append(f'<mxCell id="{next_id()}" value="" style="shape=image;imageAspect=0;aspect=fixed;image=data:image/jpeg,{data};" '
                 f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    svg.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="none" '
               f'xlink:href="data:image/jpeg;base64,{data}"/>')
    if border:
        box(x, y, w, h, border)


def text(x, y, w, lines, size=15, color=INK, bold=False):
    """Top-left aligned; `lines` are broken by hand so both outputs wrap the same."""
    h = int(len(lines) * size * 1.2)
    value = escape("\n".join(lines)).replace("\n", "&#10;")
    style = (f"text;html=0;align=left;verticalAlign=top;spacing=0;whiteSpace=nowrap;fontFamily=Helvetica;fontSize={size};"
             f"fontColor={color};fontStyle={1 if bold else 0};")
    cells.append(f'<mxCell id="{next_id()}" value="{value}" style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    for i, line in enumerate(lines):
        svg.append(f'<text x="{x}" y="{y + size + i * size * 1.2:.1f}" font-family="Helvetica, Liberation Sans, Arial, sans-serif" '
                   f'font-size="{size}" fill="{color}" font-weight="{"bold" if bold else "normal"}">{escape(line)}</text>')


def arrow(x1, y1, x2, y2, color=INK):
    cells.append(f'<mxCell id="{next_id()}" value="" style="endArrow=block;endFill=1;html=0;strokeColor={color};strokeWidth=2;" '
                 f'edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>'
                 f'<mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>')
    n = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    ux, uy = (x2 - x1) / n, (y2 - y1) / n
    bx, by = x2 - 9 * ux, y2 - 9 * uy
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" stroke-width="2"/>'
               f'<polygon points="{x2},{y2} {bx - 5 * uy:.1f},{by + 5 * ux:.1f} {bx + 5 * uy:.1f},{by - 5 * ux:.1f}" fill="{color}"/>')


def line(x1, y1, x2, y2, color="#BBBBBB", width=1):
    cells.append(f'<mxCell id="{next_id()}" value="" style="endArrow=none;html=0;strokeColor={color};strokeWidth={width};" edge="1" parent="1">'
                 f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>'
                 f'<mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>')
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')


# input: the task both agents get
box(1, 1, W - 2, 74, INK, width=2.5, fill="#EEEEEE", rx=5)
img(6, 6, 64, 64, "goal.jpg")
text(10, 8, 40, ["goal"], size=12, color="#FFFFFF", bold=True)
text(78, 5, 412, ["INPUT: the task"], size=14, bold=True)
text(78, 22, 412, ["Start in the bedroom and kitchen area to check the refrigerator",
                   "and freezer, then move to the nightstand in the bedroom to see",
                   "what's on it."], size=13)

# MemoryExplorer: top-k images -> one chosen too early -> where that leads
X = 2
text(X, 81, 130, ["MemoryExplorer"], size=14, bold=True)
text(X + 118, 82, 300, ["Top-k retrieved memories: none shows the goal"], size=13, color=GREY)
for i, n in enumerate((1, 2, 3, 4, 0)):
    img(X + i * 69, 102, 66, 66, f"me_top{n}.jpg", RED if n == 0 else None)
MAPX, MAPW, MAPH = 356, 144, 93
img(MAPX, 102, MAPW, MAPH, "me_map.jpg")
arrow(X + 342, 124, MAPX + 33, 121, RED)
text(MAPX + 49, 113, 60, ["chair"], size=12, color=RED, bold=True)
text(MAPX + 63, 125, 78, ["unseen goal"], size=12, color=GREEN, bold=True)
text(X, 175, 340, ["Commits too early to one of them. Failed."], size=15, color=RED, bold=True)
line(0, 204, W, 204, "#BBBBBB", 1)

# RCMem: the decision loop. Context shown as at decision 6; decisions 1-5 went round the loop.
text(X, 209, 190, ["RCMem (ours)"], size=14, bold=True)
box(X, 227, W - 2 * X, 94, "#888888", width=1.5, rx=4)
text(X + 6, 229, 460, ["Context: one panorama per region, unseen space stays grey"], size=13)
for x, name, label, border in ((X + 6, "pano_region0.jpg", "Region 0", None), (X + 170, "pano_region1.jpg", "Region 1", None),
                               (X + 334, "pano_region2.jpg", "Region 2", GREEN)):
    img(x, 247, 160, 68, name, border)
    text(x + 3, 249, 80, [label], size=12, color="#FFFFFF", bold=True)

SY = 342  # top of the step-2 frames
FS = 64  # step-2 frame size
MID = SY + FS // 2
BX, BW = X + 66, 146  # step-1 box
arrow(BX + BW // 2, 321, BX + BW // 2, MID - 23, INK)
box(BX, MID - 23, BW, 46, INK, width=1.5, fill="#F5F5F5", rx=5)
text(BX + 8, MID - 18, 134, ["Step 1, before moving:", "region or explore?"], size=13)

FX = X + 268
arrow(BX + BW, MID, FX - 4, MID, GREEN)
text(BX + BW + 3, MID - 17, 50, ["Region 2"], size=11, color=GREEN, bold=True)
text(FX, 324, 170, ["Step 2: its frames only"], size=13)
for i, n in enumerate((16, 17, 18)):
    img(FX + i * (FS + 4), SY, FS, FS, f"step2_frame{n}.jpg", GREEN if n == 17 else None)

# back to the context: step 1 explores
EX = X + 10
line(BX, MID, EX, MID, GREY, 2)
arrow(EX, MID, EX, 321, GREY)
text(X + 15, MID - 18, 50, ["explore"], size=13, color=GREY, bold=True)

text(X, SY + FS - 4, 260, ["Decisions 1\u20135: no commitment.", "The agent moves and the panoramas grow."], size=13, color=GREY)
arrow(FX + FS + 4 + FS // 2, SY + FS, FX + FS + 4 + FS // 2, SY + FS + 16, GREEN)
text(FX, SY + FS + 17, 212, ["Decision 6: selects the nightstand."], size=13, color=GREEN)
text(FX + 62, SY + FS + 33, 120, ["Succeeded."], size=15, color=GREEN, bold=True)

mxfile = (f'<mxfile host="make_fig1.py"><diagram name="fig1_teaser" id="fig1"><mxGraphModel dx="{W}" dy="{H}" grid="0" page="1" '
          f'pageWidth="{W}" pageHeight="{H}"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) +
          "</root></mxGraphModel></diagram></mxfile>")
open(os.path.join(HERE, "fig1_teaser.drawio"), "w", encoding="utf-8").write(mxfile)
open(os.path.join(HERE, "..", "figures", "fig1_teaser.svg"), "w", encoding="utf-8").write(
    f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}" content={quoteattr(mxfile)}>' + "".join(svg) + "</svg>\n")

# self-check: both outputs parse, and every drawn element reached the draw.io file
import xml.etree.ElementTree as T
root = T.parse(os.path.join(HERE, "fig1_teaser.drawio")).getroot()
assert len(root.findall(".//mxCell")) == len(cells) + 2
assert T.parse(os.path.join(HERE, "..", "figures", "fig1_teaser.svg")).getroot().get("content") == mxfile
