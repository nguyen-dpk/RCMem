"""Room type of every HM3D region in the LMEE scenes (needed by E1b, the room-type gate).

Source 1: LMEE task configs ("Region N: <type>", union over all tasks of a scene).
Source 2 (regions no config labels): a fixed vote over the region's ground-truth object categories (KEYWORDS below);
          regions with no keyword object stay "unknown" and are never hidden by E1b.
Validation: the vote rule is scored against the regions that do have a config label.
Run (repo root): python analysis/room_types.py   ->  analysis/out_p2/room_types.json
"""
import collections
import glob
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "analysis/out_p2")
KEYWORDS = {  # HM3D raw category (lower-case, exact) -> room type
    "bathroom": ["toilet", "bathtub", "shower", "shower glass", "bath", "towel rack", "shower curtain", "bidet"],
    "bedroom": ["bed", "nightstand", "bedside table", "wardrobe", "headboard", "bunk bed"],
    "kitchen": ["refrigerator", "stove", "oven", "microwave", "dishwasher", "kitchen cabinet", "range hood",
                "kitchen counter", "kitchen island"],
    "living room": ["sofa", "couch", "tv", "television", "fireplace", "coffee table", "sofa chair"],
    "dining room": ["dining table", "dining chair"],
    "office": ["desk", "office chair", "computer", "monitor", "printer"],
    "laundry room": ["washing machine", "dryer", "washer", "laundry basket"],
}


def type_set(label):
    """'Tie: Office & Living room' -> {'office', 'living room'}"""
    t = label.strip()
    t = t[4:].strip() if t.lower().startswith("tie:") else t
    return {x.strip().lower() for x in t.split("&") if x.strip()}


def config_labels():
    lab = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(REPO, "data/LMEE-Bench/task_test/*/*/config.json")):
        c = json.load(open(f))
        for k in c["Region_Objects"]:
            n = int(k.split(":")[0].replace("Region", "").strip())
            lab[c["Scene"]][n] = sorted(type_set(k.split(":", 1)[1]))
    return lab


def region_objects(scene):
    sid = scene.split("-")[1]
    f = glob.glob(os.path.join(REPO, f"data/hm3d-0.2/hm3d/*/{scene}/{sid}.semantic.txt"))[0]
    reg = collections.defaultdict(list)
    for line in open(f).read().splitlines()[1:]:
        m = re.match(r'(\d+),[0-9A-Fa-f]+,"(.*)",(-?\d+)', line)
        if m and int(m.group(3)) >= 0:
            reg[int(m.group(3))].append(m.group(2).lower())
    return reg


def vote(cats):
    c = collections.Counter(t for x in cats for t, kws in KEYWORDS.items() if x in kws)
    if not c:
        return []
    top = max(c.values())
    return sorted(t for t, v in c.items() if v == top)


def main():
    os.makedirs(OUT, exist_ok=True)
    lab = config_labels()
    out, n_cfg, n_vote, n_unk, hit, n_val = {}, 0, 0, 0, 0, 0
    for scene in sorted(lab):
        regs = region_objects(scene)
        m = {}
        for r, cats in regs.items():
            v = vote(cats)
            if r in lab[scene]:
                m[r] = {"types": lab[scene][r], "source": "config"}
                n_cfg += 1
                if v:
                    n_val += 1
                    hit += bool(set(v) & set(lab[scene][r]))
            elif v:
                m[r] = {"types": v, "source": "object_vote"}
                n_vote += 1
            else:
                m[r] = {"types": [], "source": "unknown"}
                n_unk += 1
        out[scene] = {str(k): v for k, v in m.items()}
    stats = {"regions": n_cfg + n_vote + n_unk, "config_labelled": n_cfg, "object_vote": n_vote, "unknown": n_unk,
             "vote_accuracy_on_config_labelled": f"{hit}/{n_val} = {100 * hit / max(n_val, 1):.1f}%"}
    json.dump({"stats": stats, "scenes": out}, open(os.path.join(OUT, "room_types.json"), "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
