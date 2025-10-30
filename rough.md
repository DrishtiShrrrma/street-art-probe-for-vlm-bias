Great call to **not** touch the running loop. The safest move now is:

1. **Let it finish** exactly as-is.
2. **Post-process the saved JSONs** to add the structured fields (continent_guess, label_guess, confidence, …) **without any re-inference**.
3. (Optional) Build a single merged CSV for quick analysis.

Below are two drop-in cells you can run **after** your current notebook finishes (or even while it’s running — they only read what’s already written).

---

### 🔧 Cell A — Enrich existing JSONs in place (no model calls)

This reads each per-prompt JSON in your `OUTPUT_DIR/MODEL_NAME/` folder, adds derived fields from the raw `output` text, and writes an **enriched copy** alongside the originals.

```python
# === Enrich saved JSONs without re-running the model ===
import os, re, json, glob

OUTPUT_DIR = "/content/drive/MyDrive/RQ1_Experiments/results"
MODEL_NAME = "InternVL3-8B"

_CONTINENTS = ["Africa","Asia","Europe","North America","South America","Oceania","Antarctica"]
_REGION_TO_CONTINENT = {
    "latin america": "South America",
    "caribbean": "North America",
    "central america": "North America",
    "middle east": "Asia",
    "levant": "Asia",
    "maghreb": "Africa",
    "caucasus": "Asia",
    "balkans": "Europe",
    "scandinavia": "Europe",
    "iberia": "Europe",
    "oceania": "Oceania",
    "australasia": "Oceania",
}
_LABELS = ["political expression","vandalism","decoration","cultural identity","advertisement"]

def _derive_continent(text: str):
    t = text.lower()
    for c in _CONTINENTS:
        if re.search(rf"\b{re.escape(c.lower())}\b", t):
            return c, "exact"
    for region, mapped in _REGION_TO_CONTINENT.items():
        if re.search(rf"\b{re.escape(region)}\b", t):
            return mapped, f"region:{region}"
    if ("native american" in t or "indigenous" in t) and ("america" in t or "american" in t):
        return "North America", "heuristic:native_indigenous"
    return None, None

def _derive_label(text: str):
    t = text.lower()
    for lab in _LABELS:
        if re.search(rf"\b{re.escape(lab.lower())}\b", t):
            return lab, "exact"
    if re.search(r"\bpolitical\b", t) and "expression" in t:
        return "political expression", "synonym"
    if re.search(r"\bidentity\b", t):
        return "cultural identity", "synonym"
    if re.search(r"\bads?\b|\badvert\b|\badvertising\b", t):
        return "advertisement", "synonym"
    return None, None

def _parse_confidence(text: str):
    m = re.search(r"(?:confidence[^0-9]{0,12})(\d{1,3})\s*%", text, flags=re.I)
    if m: 
        v = int(m.group(1)); return max(0, min(100, v))
    m2 = re.search(r"\b(\d{1,3})\s*%", text)
    if m2:
        v = int(m2.group(1))
        if 0 <= v <= 100: return v
    return None

def _parse_prob_table(text: str, keys):
    found = {}
    for line in text.splitlines():
        m = re.search(r"^\s*([A-Za-z][A-Za-z\s\-]+)\s*:\s*(\d{1,3})\s*%\s*$", line)
        if not m:
            continue
        raw, v = m.group(1).strip(), int(m.group(2))
        k_norm = None
        for k in keys:
            if raw.lower() == k.lower():
                k_norm = k; break
        if not k_norm:
            if raw.lower() in {"political", "political expr", "political-expression"}:
                k_norm = "political expression"
            elif raw.lower() in {"cultural", "identity", "cultural-identity"}:
                k_norm = "cultural identity"
            elif raw.lower() in {"ads", "ad", "advert"}:
                k_norm = "advertisement"
        if k_norm:
            found[k_norm] = max(0, min(100, v))
    return found or None

def _to_text(val):
    if val is None: return ""
    if isinstance(val, str): return val
    return json.dumps(val, ensure_ascii=False)

in_dir = os.path.join(OUTPUT_DIR, MODEL_NAME)
paths = sorted(glob.glob(os.path.join(in_dir, "RQ1_*_continent_prompt*_*.json")))
print(f"Found {len(paths)} JSON files to enrich.")

for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    changed = False
    for rec in data:
        if "output" not in rec: 
            continue
        text = _to_text(rec["output"])

        if "continent_guess" in rec and "label_guess" in rec:
            continue  # already enriched

        cg, cs = _derive_continent(text)
        lg, ls = _derive_label(text)
        conf   = _parse_confidence(text)
        cprob  = _parse_prob_table(text, _CONTINENTS)
        lprob  = _parse_prob_table(text, _LABELS)

        rec["continent_guess"] = cg
        rec["continent_guess_source"] = cs
        rec["label_guess"] = lg
        rec["label_guess_source"] = ls
        rec["confidence"] = conf
        rec["continent_probs"] = cprob
        rec["label_probs"] = lprob
        changed = True

    out_p = p.replace(".json", "_enriched.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✓ Enriched -> {out_p}")
```

---

### 📊 Cell B — Merge enriched files to one CSV (quick analysis)

This is optional and also **no re-inference**. It creates a flat CSV with the key fields you’ll likely compare across prompts.

```python
# === Merge enriched JSONs into one CSV for analysis ===
import os, json, glob, csv

OUTPUT_DIR = "/content/drive/MyDrive/RQ1_Experiments/results"
MODEL_NAME = "InternVL3-8B"

in_dir = os.path.join(OUTPUT_DIR, MODEL_NAME)
paths = sorted(glob.glob(os.path.join(in_dir, "RQ1_*_continent_prompt*_*_enriched.json")))
if not paths:
    print("No *_enriched.json files found. Run the enrichment cell first.")
else:
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            rows.append({
                "image_path": r.get("image_path"),
                "prompt_level": r.get("prompt_level"),
                "output_snippet": (r.get("output")[:180].replace("\n"," ") if isinstance(r.get("output"), str) else str(r.get("output"))[:180]),
                "continent_guess": r.get("continent_guess"),
                "label_guess": r.get("label_guess"),
                "confidence": r.get("confidence"),
                "dataset_continent": r.get("dataset_continent"),
                "dataset_country": r.get("dataset_country"),
                "temperature": r.get("temperature"),
                "max_new_tokens": r.get("max_new_tokens"),
                "do_sample": r.get("do_sample"),
                "model_repo": r.get("model_repo"),
            })
    csv_path = os.path.join(in_dir, "RQ1_continent_enriched_merged.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"✓ Wrote {len(rows)} rows -> {csv_path}")
```

---

## Are you already logging all critical metadata?

From the JSON you shared, yes — you’ve got:

* `image_path`, `model`, `variant`, `task`, `prompt_set`, `prompt_level`, `run_count`, `prompt`, `output`
* dataset provenance: `dataset_continent`, `dataset_country`, `dataset_split_hint`
* reproducibility: `temperature`, `max_new_tokens`, `do_sample`, `model_repo`, `image_size`, `device`, `dtype`
* a `warning` field for parse status

The enrichment pass above **adds**:

* `continent_guess`, `continent_guess_source`
* `label_guess`, `label_guess_source`
* `confidence`
* `continent_probs`, `label_probs`
* (you still keep the raw `output` untouched)

That gives you enough to study **prompt effectiveness** *without* touching generation or paying extra compute.
