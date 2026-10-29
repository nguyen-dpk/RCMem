"""P3 qualitative pilot report (DEV_PLAN.md "Qualitative HTML reports", P3 row): one self-contained HTML file per run of
analysis/p3_nav.py --viz (RG or RP), docs/qual/p3_<exp>_<YYYYMMDD>.html.

Per episode of the pilot sample (analysis/p3_pilot.py) and per navigation decision (collapsible):
 1 frames captured since the previous decision, kept or dropped, with their share of new voxels;
 2 the top-down map: predicted regions (RP) or the observed GT regions (RG) with walls, centres (numbered as in the
   prompt), the agent's path and the goal (magenta star), next to the GT regions; the share of points assigned to the
   right region (RP);
 3 each region's A1-band panorama with hole fraction, centre and render time;
 4 the exact step-1 prompt text and its token count;
 5 step 1: raw reply, parsed choice, ground-truth target (the goal's region if observed, else "explore") and the regions
   whose frames show the goal, with correct / wrong;
 6 step 2: the chosen region's frames as the model saw them (numbered boxes), the GT goal box in green, the chosen frame
   outlined in red, the reply and whether the chosen object is the goal;
 7 the subtask outcome: success, failure category (analysis/analyze.py taxonomy); the path is on the map.
Images are embedded as compressed JPEG thumbnails; click one to enlarge.
Run (repo root, lmee env): python analysis/p3_qual.py --run p3_pilot_RP
"""
import argparse
import base64
import collections
import html
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
import analyze as A  # noqa: E402
import p2_report as P2  # noqa: E402

OK, BAD = "<b class=ok>✓ correct</b>", "<b class=bad>✗ wrong</b>"


def jpg(img_or_path, width, q=70):
    im = cv2.imread(img_or_path) if isinstance(img_or_path, str) else img_or_path
    if im is None:
        return ""
    if im.shape[1] > width:
        im = cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1])), interpolation=cv2.INTER_AREA)
    ok, b = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(b.tobytes()).decode()


def img(src, w, cls="", title=""):
    return f'<img src="{src}" style="width:{w}px" class="{cls}" title="{html.escape(title)}" onclick="big(this)">' if src else "(missing)"


def mark(ok):
    return OK if ok else BAD


def decision_html(run_dir, vdir, prev_n, ep_dir, seen_prev):
    info = json.load(open(os.path.join(vdir, "info.json")))
    p3 = info["p3"]
    H = []
    # 1 frames
    names = list(info["frac"])
    new = [n for n in names if n not in seen_prev]
    kept = set(info["kept"])
    cells = []
    for n in new:
        f = info["frac"][n]
        k = n in kept
        cells.append(f'<div class=fr>{img(jpg(os.path.join(ep_dir, "memory", n), 120, 60), 96)}<br>'
                     f'{n}<br>{"kept" if k else "dropped"} · new {0 if f is None else 100 * float(f):.0f}%</div>')
    H.append(f"<h4>1. Frames since the previous decision ({len(new)} captured, "
             f"{sum(n in kept for n in new)} kept; {len(kept)} kept in total)</h4><div class=row>{''.join(cells)}</div>")
    # 2 regions
    seg = p3.get("seg") or {}
    segtxt = (f"share of points assigned to the right region {100 * seg['point_acc']:.1f}% · coverage {100 * seg['coverage']:.1f}% · "
              f"predicted regions {seg['n_pred']} vs GT {seg['n_gt']} · merged {seg['merged']} · split {seg['split']}") if seg else "ground-truth regions"
    gtmap = ", ".join(f"Region {i} = {r} (GT {p3.get('region_gt', {}).get(r, r)})" for i, r in enumerate(p3.get("shown", p3["regions"])))
    H.append(f"<h4>2. Regions</h4><p>left: {'predicted regions' if p3['exp'] == 'RP' else 'observed GT regions'} "
             f"(white = wall, grey = observed, unassigned; numbers = prompt Region index at its centre; blue = path; "
             f"magenta star = goal) · right: GT regions of the kept frames' points<br>{segtxt}<br>{gtmap}</p>"
             + img(jpg(os.path.join(vdir, "map.png"), 900, 80), 900))
    # 3 panoramas
    pans = []
    for i, r in enumerate(info["regions"]):
        f = os.path.join(vdir, f"pano_{i}.jpg")
        if os.path.exists(f):
            c = info["centres"][i]
            pans.append(f'<div class=pano><b>{r}</b> · holes {100 * info["holes"][i]:.0f}% · centre '
                        f'({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})<br>{img(jpg(f, 768, 65), 600)}</div>')
    H.append(f"<h4>3. Panoramas ({len(pans)}; built in {p3['render_s']:.2f} s)</h4>" + "".join(pans))
    # 4 context
    H.append(f"<h4>4. Step-1 prompt ({p3.get('step1_tokens', 0)} tokens; images as &lt;image&gt;)</h4>"
             f"<pre>{html.escape(''.join(info['prompt']))}</pre>" if info["prompt"] else
             "<h4>4. Step-1 prompt</h4><p>none: no region observed yet → empty-context rule (nearest frontier, no VLM call)</p>")
    # 5 step 1
    tgt, ch = p3["target"], p3.get("step1_choice")
    if info["prompt"]:
        H.append(f"<h4>5. Step 1 {mark(ch == tgt)}</h4><p>parsed: <b>{ch}</b>"
                 f"{' (Frontier ' + str(p3.get('step1_frontier')) + ')' if p3.get('step1_frontier') is not None else ''}"
                 f" · ground truth: <b>{tgt}</b> (goal's GT region {p3['goal_regions_gt']}; observed: "
                 f"{'yes' if tgt != 'explore' else 'not seen yet'}) · regions whose frames show the goal: {p3['goal_visible_regions'] or 'none'}"
                 f"</p><pre>{html.escape(p3.get('step1_reply', ''))}</pre>")
    # 6 step 2
    if p3.get("step2"):
        c = info.get("choice") or {}
        s2 = info.get("step2_frames", [])
        cells = []
        for f in s2:
            chosen = c.get("type") == "memory" and c.get("frame_index") == f["frame"]
            cells.append(f'<div class=fr>{img(jpg(os.path.join(vdir, "s2_%d.jpg" % f["frame"]), 360, 70), 180, "chosen" if chosen else "")}'
                         f'<br>Frame {f["frame"]}{" ← chosen, Object " + str(c.get("object_index")) if chosen else ""}'
                         f'<br>{html.escape(", ".join(f"{j}:{o}" for j, o in enumerate(f["objects"])))}'
                         f'<br>goal px {f["goal_px"]}</div>')
        H.append(f"<h4>6. Step 2 ({info.get('step2_tokens')} tokens): {c.get('type', 'invalid')} "
                 f"{'→ ' + str(c.get('cluster')) if c.get('type') == 'memory' else ''} · {{GOAL2}}</h4>"
                 f"<div class=row>{''.join(cells)}</div><pre>{html.escape(info.get('step2_reply') or '')}</pre>")
    return "\n".join(H), set(names), p3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run_dir = os.path.join(REPO, "analysis/runs", a.run)
    pilot = json.load(open(os.path.join(REPO, "analysis/out_p3/pilot.json")))
    scenes, subtasks, decisions, nav, qa = P2.load(a.run)
    navd = {r["subtask"]: r for r in nav}
    recs = collections.defaultdict(list)
    for sid, ds in decisions.items():
        recs[sid] = ds
    vdirs = collections.defaultdict(list)
    for d in sorted(os.listdir(os.path.join(run_dir, "viz"))):
        vdirs[d.rsplit("_", 1)[0]].append(os.path.join(run_dir, "viz", d))
    exp = json.loads(open(os.path.join(run_dir, sorted(f for f in os.listdir(run_dir) if f.startswith("records"))[0])).readline()) and \
        next((d.get("p3", {}).get("exp") for ds in decisions.values() for d in ds if d.get("p3")), "?")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "analysis/p3_nav.py", "rcmem"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    log = open(os.path.join(run_dir, sorted(f for f in os.listdir(run_dir) if f.startswith("log_"))[0])).readline().strip()
    run_eps = sorted({st["episode"] for st in subtasks.values()})
    eps = collections.OrderedDict((e, []) for e in ([e for e in pilot["episodes"] if e in run_eps] or run_eps))  # smoke runs: their own
    for sid, st in subtasks.items():
        if st["episode"] in eps:
            eps[st["episode"]].append(sid)
    S, B = [], []
    n_step1 = n_ok = 0
    for ep, sids in eps.items():
        ep_dir = os.path.join(run_dir, ep)
        rows, body, seen = [], [], set()
        for sid in sorted(sids, key=lambda s: int(s.rsplit("_", 1)[-1])):
            st = subtasks[sid]
            if st["goal_type"] == "QA":
                ok = st.get("pred_answer") is not None and str(st.get("pred_answer", "")).strip()[:1] == str((st.get("QA") or {}).get("answer", "?"))
                rows.append(f"<tr><td>{sid}</td><td>QA</td><td colspan=3>{html.escape(str(st['question'])[:150])} → "
                            f"{html.escape(str(st.get('pred_answer')))} (answer {html.escape(str((st.get('QA') or {}).get('answer')))}) {mark(ok)}</td></tr>")
                continue
            r = navd.get(sid, {})
            succ = bool(r.get("success"))
            rows.append(f"<tr><td><a href='#{sid}'>{sid}</a></td><td>{st['goal_type']}</td><td>{html.escape(st['class'])}</td>"
                        f"<td>{mark(succ).replace('correct', 'success').replace('wrong', 'failure')}</td><td>{html.escape(r.get('category', '?'))}</td></tr>")
            blocks = []
            for i, vd in enumerate(vdirs.get(sid, [])):
                h, seen_new, p3 = decision_html(run_dir, vd, i, ep_dir, seen)
                seen |= seen_new
                d = recs[sid][i] if i < len(recs[sid]) else {}
                ch = d.get("choice") or {}
                goal_det = {int(x) for v in (d.get("goal_obj_ids_mapping") or {}).values() for x in v}
                g2 = ch.get("type") == "memory" and bool(set(ch.get("cluster", [])) & goal_det)
                h = h.replace("{GOAL2}", f"chosen object is the goal: {mark(g2)}")
                if p3.get("step1_reply"):
                    n_step1 += 1
                    n_ok += p3.get("step1_choice") == p3["target"]
                summ = (f"decision {i + 1} · step 1: {p3.get('step1_choice') or 'empty-context explore'} (target {p3['target']}) "
                        f"{mark(p3.get('step1_choice') == p3['target']) if p3.get('step1_reply') else ''} · "
                        f"choice {ch.get('type', '-')}{' ' + mark(g2) if ch.get('type') == 'memory' else ''}")
                blocks.append(f"<details><summary>{summ}</summary>{h}</details>")
            body.append(f"<h3 id='{sid}'>{sid} · {st['goal_type']} · {html.escape(st['class'])} · "
                        f"{'success' if succ else 'failure'}: {html.escape(r.get('category', '?'))}</h3>"
                        f"<p>{html.escape(st['question'][:600])}</p>"
                        + (img(jpg(os.path.join(run_dir, sid, 'image_goal.png'), 300), 200) if st["goal_type"] == "image" else "")
                        + "".join(blocks))
        S.append(f"<tr><td><a href='#{ep}'>{ep}</a></td><td>{', '.join(c for c, v in pilot['cases'].items() if ep in v)}</td>"
                 f"<td><table>{''.join(rows)}</table></td></tr>")
        B.append(f"<h2 id='{ep}'>{ep}</h2>" + "".join(body))
    navp = [navd[s] for e in eps for s in eps[e] if s in navd]
    out = a.out or os.path.join(REPO, "docs/qual", f"p3_{exp}_{time.strftime('%Y%m%d')}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    doc = f"""<!doctype html><html><head><meta charset=utf-8><title>P3 pilot {exp}</title><style>
body{{font-family:sans-serif;margin:16px;max-width:1400px}} pre{{white-space:pre-wrap;background:#f4f4f4;padding:6px;font-size:12px}}
.ok{{color:#070}} .bad{{color:#b00}} .row{{display:flex;flex-wrap:wrap;gap:6px}} .fr{{font-size:11px;text-align:center}}
.chosen{{outline:4px solid red}} img{{cursor:zoom-in}} details{{border:1px solid #ccc;margin:4px 0;padding:4px}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:3px;font-size:13px;vertical-align:top}}
#ov{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:9}} #ov img{{max-width:98%;max-height:98%;margin:auto;display:block}}
</style><script>function big(e){{var o=document.getElementById('ov');o.firstChild.src=e.src;o.style.display='block'}}</script></head><body>
<div id=ov onclick="this.style.display='none'"><img></div>
<h1>P3 pilot report: RCMem zero-shot, {exp} ({'predicted' if exp == 'RP' else 'ground-truth'} regions)</h1>
<p>git commit {commit}{' (uncommitted changes in p3_nav.py / rcmem)' if dirty else ''} · run <code>{a.run}</code> ·
config <code>analysis/cfg/p3_pilot.yaml</code> · {html.escape(log)}<br>
command: <code>python analysis/p3_nav.py --exp {exp} --vlm &lt;checkpoint&gt; --run_name {a.run} --cfg_file analysis/cfg/p3_pilot.yaml --seed 77 --viz</code>,
then <code>python analysis/p3_qual.py --run {a.run}</code></p>
<p><b>Pilot sample</b> (analysis/p3_pilot.py, seed {pilot['seed']}): {', '.join(pilot['episodes'])}.<br>{html.escape(pilot['rule'])}</p>
<p><b>Summary:</b> {sum(r['success'] for r in navp)}/{len(navp)} navigation subtasks succeeded; step 1 correct in {n_ok}/{n_step1} decisions that asked it.</p>
<table><tr><th>Episode</th><th>Cases covered</th><th>Subtasks</th></tr>{''.join(S)}</table>
{''.join(B)}</body></html>"""
    open(out, "w").write(doc)
    print(out, f"{os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
