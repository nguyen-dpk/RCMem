"""Instruction parser (P3): a compound LMEE instruction + its ordered goal names -> ordered (goal, region) pairs.

Rule-based and deterministic. The instruction is split into clauses at sequence words ("then", "next", "finally",
"followed by", sentence ends, ...). Room mentions come from a fixed vocabulary of indoor region types with synonyms;
"X and Y area", "X & Y" or "shared space of X and Y" give one mention with several types. Each goal is located in the
text (after the previous goal when possible) and takes the room mention of its clause (the nearest one before it, else
the first after it), or the previous goal's when its clause names none. Region instances: every mention opens a new
instance, except a mention whose types equal an earlier one's and that carries no "another / other / second / next /
last / different" word, which re-uses that instance (e.g. "back to the kitchen").

Run `python rcmem/instruction.py` to score it against the task configs (data/LMEE-Bench/task_test/*/*/config.json):
goal located, region type correct (overlaps the config's type, ties included), and region-instance grouping (for every
pair of subtasks of a task: same instance in the parse <=> same config region). Writes analysis/out_p3/parser_eval.json.
"""
import glob
import json
import os
import re

TYPES = {  # canonical type (config spelling) -> synonyms, longest first when matching
    "Bedroom": ["master bedroom", "guest bedroom", "bedroom", "bed room"],
    "Bathroom": ["bathroom", "restroom", "washroom", "toilet room", "powder room"],
    "Kitchen": ["kitchen"],
    "Living room": ["living room", "family room", "sitting room", "lounge", "living area"],
    "Dining room": ["dining room", "dining area"],
    "Office": ["office", "study"],
    "Hall": ["hallway", "hall", "corridor", "entryway", "foyer", "entrance"],
    "Laundry room": ["laundry room", "laundry", "utility room"],
    "Rec room": ["rec room", "recreation room", "game room", "playroom"],
}
SYN = sorted(((s, t) for t, ss in TYPES.items() for s in ss), key=lambda x: -len(x[0]))
NEW_WORDS = ("another", "other", "second", "third", "next", "last", "different", "final")
CLAUSE = re.compile(r"[.;!?]\s+|,?\s*\b(?:then|next|finally|afterwards|after that|followed by|before heading|proceed(?:ing)? to|"
                    r"continue to|and end)\b", re.I)


def room_mentions(text):
    """[(start, end, types frozenset, phrase)] in text order; joined mentions ('X and Y area', 'X & Y') merged."""
    low, hits, taken = text.lower(), [], set()
    for s, t in SYN:
        for m in re.finditer(r"\b" + re.escape(s) + r"s?\b", low):
            if not taken & set(range(m.start(), m.end())):
                hits.append([m.start(), m.end(), {t}])
                taken |= set(range(m.start(), m.end()))
    hits.sort()
    out = []
    for h in hits:  # merge "kitchen and living room area", "laundry room & kitchen", "bedroom/living room"
        if out and re.fullmatch(r"\s*(?:and|&|/|or)\s*(?:the\s+)?", low[out[-1][1]:h[0]]):
            out[-1][1], out[-1][2] = h[1], out[-1][2] | h[2]
        else:
            out.append(h)
    return [(a, b, frozenset(t), text[a:b]) for a, b, t in out]


def find_goal(low, name, start):
    for n in (name, name.split()[-1]):
        pat = r"\b" + re.escape(n.lower()) + r"(?:s|es)?\b"
        m = re.search(pat, low[start:]) or re.search(pat, low)
        if m:
            return m.start() + (start if re.search(pat, low[start:]) else 0)
    return None


def parse(instruction, goals):
    """-> [{"goal", "pos", "types", "phrase", "instance"}] in goal order; types empty / instance None when no room."""
    low = instruction.lower()
    ments = room_mentions(instruction)
    cuts = [0] + [m.start() for m in CLAUSE.finditer(instruction)] + [len(instruction) + 1]
    clause_of = lambda p: max(i for i, c in enumerate(cuts[:-1]) if c <= p)
    inst, inst_of = [], {}  # mention index -> instance
    for k, (a, b, t, ph) in enumerate(ments):
        pre = low[max(0, a - 25):a]
        new = any(re.search(r"\b" + w + r"\b", pre) for w in NEW_WORDS)
        prev = [inst_of[j] for j in range(k) if ments[j][2] == t]
        inst_of[k] = prev[-1] if prev and not new else len(inst)
        if inst_of[k] == len(inst):
            inst.append(k)
    out, pos, last = [], 0, None
    for g in goals:
        p = find_goal(low, g, pos)
        m = None
        if p is not None:
            pos = p + 1
            same = [k for k, x in enumerate(ments) if clause_of(x[0]) == clause_of(p)]
            before = [k for k in same if ments[k][0] < p]
            m = before[-1] if before else (same[0] if same else last)
        else:
            m = last
        last = m
        out.append({"goal": g, "pos": p, "types": sorted(ments[m][2]) if m is not None else [],
                    "phrase": ments[m][3].strip() if m is not None else "", "instance": inst_of.get(m)})
    return out


def config_types(label):
    t = label.split(":", 1)[-1].strip()
    t = t[4:].strip() if t.lower().startswith("tie:") else t
    return {x.strip().capitalize() if x.strip().lower() not in ("rec room",) else "Rec room" for x in t.split("&")}


def evaluate(root):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "data/LMEE-Bench/task_test/*/*/config.json"))):
        c = json.load(open(f))
        goals = [o.rsplit("_", 1)[0] for o, _ in c["Object"]]
        gt_t = [config_types(r) for _, r in c["Object"]]
        gt_r = [r.split(":")[0] for _, r in c["Object"]]
        P = parse(c["Task instruction"], goals)
        pairs = [(i, j) for i in range(len(P)) for j in range(i + 1, len(P))]
        same_ok = [(P[i]["instance"] is not None and P[i]["instance"] == P[j]["instance"]) == (gt_r[i] == gt_r[j]) for i, j in pairs]
        rows.append({"task": c["Task instruction"], "difficulty": c["Difficulty"], "n": len(P),
                     "found": [p["pos"] is not None for p in P],
                     "type_ok": [bool({x.lower() for x in p["types"]} & {x.lower() for x in t}) for p, t in zip(P, gt_t)],
                     "pairs_ok": same_ok, "parse": P, "gt": list(zip(gt_r, [sorted(t) for t in gt_t]))})
    flat = lambda k: [x for r in rows for x in r[k]]
    summ = {"tasks": len(rows), "subtasks": len(flat("found")),
            "goal_located": 100 * sum(flat("found")) / len(flat("found")),
            "region_type": 100 * sum(flat("type_ok")) / len(flat("type_ok")),
            "instance_pairs": 100 * sum(flat("pairs_ok")) / max(1, len(flat("pairs_ok"))),
            "task_exact": 100 * sum(all(r["type_ok"]) and all(r["pairs_ok"]) for r in rows) / len(rows)}
    for d in ("easy", "medium", "hard"):
        rr = [r for r in rows if r["difficulty"] == d]
        summ[f"region_type_{d}"] = 100 * sum(x for r in rr for x in r["type_ok"]) / max(1, sum(r["n"] for r in rr))
        summ[f"task_exact_{d}"] = 100 * sum(all(r["type_ok"]) and all(r["pairs_ok"]) for r in rr) / max(1, len(rr))
    return summ, rows


if __name__ == "__main__":
    # self-check on hand-written cases
    p = parse("Start in the first bedroom to look at the painting, then head to the bathroom to check the sink, and finally "
              "go to the second bedroom to examine the painting and the nightstand.", ["painting", "sink", "painting", "nightstand"])
    assert [x["types"] for x in p] == [["Bedroom"], ["Bathroom"], ["Bedroom"], ["Bedroom"]], p
    assert p[0]["instance"] != p[2]["instance"] == p[3]["instance"], p
    p = parse("Check the refrigerator in the kitchen, then visit the dining room and living room area to see the vase, "
              "then go back to the kitchen to inspect the microwave.", ["refrigerator", "vase", "microwave"])
    assert p[0]["types"] == ["Kitchen"] and set(p[1]["types"]) == {"Dining room", "Living room"} and p[0]["instance"] == p[2]["instance"], p
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summ, rows = evaluate(root)
    os.makedirs(os.path.join(root, "analysis/out_p3"), exist_ok=True)
    json.dump({"summary": summ, "errors": [r for r in rows if not (all(r["type_ok"]) and all(r["pairs_ok"]))]},
              open(os.path.join(root, "analysis/out_p3/parser_eval.json"), "w"), indent=1)
    print(json.dumps({k: round(v, 1) for k, v in summ.items()}, indent=1))
