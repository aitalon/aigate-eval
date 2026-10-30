"""PII dataset converters: pinned public datasets -> benchmarks/pii/<name>.jsonl rows.

The mappings are the ones used for our internal evaluation, unchanged. hivetrace / redmadrobot
are read straight from Hugging Face (parquet / csv) and normalized to character spans first.

Row format:
  {id, source, split, category, language, text, entities: [{start, end, type, text}]}
entities == [] -> labeled row without PII. Offsets are Python str offsets and
text[start:end] == entity["text"].

Upstream labels are mapped to our types as is (no manual relabeling). Several sets annotate
only a subset of types: see SPECS[...]["scored_labels"]; predictions of other types on such a
file are not errors of the detector.
"""
import collections
import csv
import hashlib
import json
import os
import random
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SEED = 20260926

LABELS = (
    "PII.PERSON", "PII.ADDRESS", "PII.DATE_OF_BIRTH", "PII.PHONE", "PII.EMAIL", "PII.LOGIN",
    "PII.PASSPORT.RU", "PII.PASSPORT.FOREIGN", "PII.SNILS", "PII.INN", "PII.OMS", "PII.DRIVER_LICENSE",
    "PII.BIRTH_CERTIFICATE", "PII.MILITARY_ID", "PII.VEHICLE_DOCS", "PII.VIN", "PII.VEHICLE_REGISTRATION",
    "PII.BANK_CARD", "PII.CVC", "PII.BANK_ACCOUNT", "PII.BIK", "PII.KPP", "PII.OGRN", "PII.OGRNIP",
    "NETWORK.IP", "NETWORK.MAC",
)
LABEL_SET = frozenset(LABELS)
ALL = sorted(LABELS)


# --------------------------------------------------------------------------- helpers

def lang_of(text):
    cyr = len(re.findall(r"[а-яё]", text, re.I))
    lat = len(re.findall(r"[a-z]", text, re.I))
    tot = cyr + lat
    if not tot or cyr >= 0.8 * tot:
        return "ru"
    if lat >= 0.8 * tot:
        return "en"
    return "mixed"


def ent(text, s, e, t):
    # trim whitespace at the edges, keep text == text[start:end]
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return {"start": s, "end": e, "type": t, "text": text[s:e]}


def finalize(ents):
    """sort, drop exact dups and spans nested inside an equal-typed/longer span."""
    ents = sorted(ents, key=lambda x: (x["start"], -(x["end"] - x["start"])))
    out = []
    for x in ents:
        if x["end"] <= x["start"]:
            continue
        if out and x["start"] < out[-1]["end"]:
            # overlap: keep the longer one (the earlier, by sort order) unless x extends it
            if x["end"] <= out[-1]["end"]:
                continue
            if x["type"] == out[-1]["type"]:  # crossing same-type spans -> union
                p = out[-1]
                out[-1] = {"start": p["start"], "end": x["end"], "type": p["type"],
                           "text": p["text"] + x["text"][p["end"] - x["start"]:]}
                continue
            raise ValueError("crossing spans %r / %r" % (out[-1], x))
        out.append(x)
    return out


def row(id_, source, split, category, text, entities):
    return {"id": id_, "source": source, "split": split, "category": category,
            "language": lang_of(text), "text": text, "entities": entities}


def validate(rows):
    ids = set()
    for r in rows:
        assert r["id"] not in ids, r["id"]
        ids.add(r["id"])
        assert isinstance(r["entities"], list), r["id"]
        prev = -1
        for e in r["entities"]:
            assert e["type"] in LABEL_SET, e
            assert 0 <= e["start"] < e["end"] <= len(r["text"]), (r["id"], e)
            assert r["text"][e["start"]:e["end"]] == e["text"], (r["id"], e)
            assert e["start"] >= prev, (r["id"], e)
            prev = e["end"]


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parquet_rows(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


class Ctx:
    """Per-source run context: cache dir, row limit (None = no limit) and drop/notes counters."""

    def __init__(self, cache, limit=None):
        self.cache = cache
        self.limit = limit
        self.dropped = collections.Counter()
        self.notes = collections.Counter()

    def hf(self, repo, filename, revision):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                               cache_dir=os.path.join(self.cache, "hf"))

    def fetch(self, url, rel):
        """download once into the cache; returns None on 404."""
        path = os.path.join(self.cache, "raw", rel)
        if os.path.exists(path + ".404"):
            return None
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = r.read()
            except urllib.error.HTTPError as ex:
                if ex.code != 404:
                    raise
                open(path + ".404", "w").close()
                return None
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        return path


def seeded_sample(rows, limit):
    """seeded sample of `limit` rows (sorted by id); all rows when limit is None or not exceeded."""
    if limit is not None and len(rows) > limit:
        rows = list(rows)
        random.Random(SEED).shuffle(rows)
        rows = sorted(rows[:limit], key=lambda r: r["id"])
    return rows


FOREIGN_CTX = re.compile(
    r"загран|иностран|foreign|гражданина\s+(?!РФ|России|Российской|Р\.Ф)[A-ZА-ЯЁ]{3,}", re.I)


def passport_type(text, start):
    # sentence-local look-back: foreign passport when the context says so
    ctx = text[max(0, start - 80):start]
    ctx = re.split(r"[.!?\n]\s", ctx)[-1]
    return "PII.PASSPORT.FOREIGN" if FOREIGN_CTX.search(ctx) else "PII.PASSPORT.RU"


# --------------------------------------------------------------------------- hivetrace/pii-bench

HIVE_REPO = "hivetrace/pii-bench"
HIVE_MAP = {"NAME": "PII.PERSON", "ADDRESS": "PII.ADDRESS", "BANK_CARD_NUMBER": "PII.BANK_CARD",
            "CVC": "PII.CVC", "EMAIL": "PII.EMAIL", "INN": "PII.INN", "KPP": "PII.KPP",
            "OGRN": "PII.OGRN", "OGRNIP": "PII.OGRNIP", "PASSPORT_NUMBER": "PII.PASSPORT.*",
            "PHONE_NUMBER": "PII.PHONE", "SNILS": "PII.SNILS"}


def simple_map(rows, mapping, st, source=None):
    out = []
    for r in rows:
        text, ents = r["text"], []
        for e in r.get("entities") or []:
            if not (0 <= e["start"] < e["end"] <= len(text)):
                st.notes["out-of-range/empty upstream spans skipped"] += 1
                continue
            t = mapping.get(e["type"])
            if t == "PII.PASSPORT.*":
                t = passport_type(text, e["start"])
            if t is None:
                st.dropped[e["type"]] += 1
                continue
            ents.append(ent(text, e["start"], e["end"], t))
        out.append(row(r["id"], r.get("source") or source, r.get("split") or "test",
                       r.get("category") or "", text, finalize(ents)))
    return out


def hivetrace_upstream(items_by_split):
    """{split: [parquet rows]} -> intermediate rows (as in aigate's snapshot)."""
    rows = []
    for split in ("entity", "domain"):
        for item in items_by_split[split]:
            entities = [dict(e) for e in item["entities"] or []]
            rows.append({"id": "hivetrace-%s-%s" % (split, item["id"]), "source": HIVE_REPO,
                         "split": split, "category": item["domain"], "text": item["text"],
                         "entities": entities})
    return rows


def conv_hivetrace(rev, st):
    items = {split: parquet_rows(st.hf(HIVE_REPO, "data/%s-00000-of-00001.parquet" % split, rev))
             for split in ("entity", "domain")}
    return simple_map(seeded_sample(hivetrace_upstream(items), st.limit), HIVE_MAP, st)


# --------------------------------------------------------------------------- redmadrobot-rnd/pii_benchmark

RMR_REPO = "redmadrobot-rnd/pii_benchmark"


def token_offsets(text, tokens):
    folded = text.casefold()
    cursor = 0
    offsets = []
    for token in tokens:
        start = folded.find(token.casefold(), cursor)
        if start < 0:
            raise ValueError("cannot align token %r after %d in %r" % (token, cursor, text))
        end = start + len(token)
        offsets.append((start, end))
        cursor = end
    return offsets


def bio_entities(text, tokens, tags):
    if len(tokens) != len(tags):
        raise ValueError("tokens/tags length mismatch in %r" % text)
    entities, current = [], None
    for (start, end), tag in zip(token_offsets(text, tokens), tags):
        if tag.startswith("B-"):
            if current:
                entities.append(current)
            current = {"start": start, "end": end, "type": tag[2:]}
        elif tag.startswith("I-") and current and tag[2:] == current["type"]:
            current["end"] = end
        else:
            if current:
                entities.append(current)
            current = None
    if current:
        entities.append(current)
    for e in entities:
        e["text"] = text[e["start"]:e["end"]]
    return entities


# redmadrobot: fragments are merged into whole PERSON / ADDRESS spans
NAME_PARTS = {"FIRST_NAME", "LAST_NAME", "MIDDLE_NAME"}
ADDR_PARTS = {"COUNTRY", "REGION", "DISTRICT", "CITY", "STREET", "HOUSE"}
ADDR_CORE = {"STREET", "HOUSE"}
RMR_MAP = {"PHONE": "PII.PHONE", "EMAIL": "PII.EMAIL", "INN": "PII.INN", "SNILS": "PII.SNILS",
           "OMS": "PII.OMS", "CREDIT_CARD": "PII.BANK_CARD", "IP_ADDRESS": "NETWORK.IP",
           "DRIVER_LICENSE": "PII.DRIVER_LICENSE", "BIRTH_CERTIFICATE": "PII.BIRTH_CERTIFICATE",
           "MILITARY_ID": "PII.MILITARY_ID", "PASSPORT": "PII.PASSPORT.*"}
# tokens allowed between two address parts (besides , ; whitespace): abbreviations and a 6-digit index
ADDR_GAP_TOKENS = re.compile(
    r"^(г|гор|город|д|дом|ул|улица|обл|область|р-н|район|пос|п|с|пр|пр-т|к|корп|стр|кв|"
    r"g|ul|oblast|d|\d{6})\.?$", re.I)
NAME_GAP = re.compile(r"^\s*,?\s+$|^\s+$")
SAME_GAP = re.compile(r"^[\s\-–—/]+$")  # series + number of one document split in two spans


def addr_gap_ok(gap):
    toks = [t for t in re.split(r"[\s,;]+", gap) if t]
    return all(ADDR_GAP_TOKENS.match(t) for t in toks) and "\n" not in gap


def redmadrobot_upstream(csv_rows):
    """test.csv rows {text, tokens (json), ner_tags (json)} -> intermediate rows with raw BIO spans."""
    return [{"id": "redmadrobot-%04d" % i, "source": RMR_REPO, "split": "test", "category": "mixed",
             "text": r["text"],
             "entities": bio_entities(r["text"], json.loads(r["tokens"]), json.loads(r["ner_tags"]))}
            for i, r in enumerate(csv_rows)]


def redmadrobot_map(rows, st):
    out = []
    for r in rows:
        text = r["text"]
        bad = [e for e in r["entities"] if not (0 <= e["start"] < e["end"] <= len(text))]
        if bad:
            st.notes["out-of-range/empty upstream spans skipped"] += len(bad)
        es = sorted((e for e in r["entities"] if e not in bad), key=lambda e: (e["start"], e["end"]))
        groups = []  # [kind, start, end, set(types)]
        for e in es:
            t = e["type"]
            kind = "NAME" if t in NAME_PARTS else "ADDR" if t in ADDR_PARTS else t
            g = groups[-1] if groups else None
            if g and g[0] == kind and e["start"] >= g[2]:
                gap = text[g[2]:e["start"]]
                if kind == "NAME" and NAME_GAP.match(gap) and t not in g[3]:
                    g[2] = e["end"]; g[3].add(t); st.notes["name parts merged"] += 1; continue
                if kind == "ADDR" and addr_gap_ok(gap) and t not in g[3]:
                    g[2] = e["end"]; g[3].add(t); st.notes["address parts merged"] += 1; continue
                if kind in RMR_MAP and kind not in ("PHONE", "EMAIL") and SAME_GAP.match(gap):
                    g[2] = e["end"]; st.notes["%s parts merged" % kind] += 1; continue
            groups.append([kind, e["start"], e["end"], {t}])
        ents = []
        for kind, s, e_, types in groups:
            if kind == "NAME":
                ents.append(ent(text, s, e_, "PII.PERSON"))
            elif kind == "ADDR":
                if types & ADDR_CORE:
                    ents.append(ent(text, s, e_, "PII.ADDRESS"))
                else:
                    for t in types:
                        st.dropped["%s (standalone, no street/house)" % t] += 1
            elif kind in RMR_MAP:
                t = RMR_MAP[kind]
                if t == "PII.PASSPORT.*":
                    t = passport_type(text, s)
                ents.append(ent(text, s, e_, t))
            else:
                st.dropped[kind] += 1
        out.append(row(r["id"], r["source"], r.get("split") or "test", r.get("category") or "mixed",
                       text, finalize(ents)))
    return out


def conv_redmadrobot(rev, st):
    with open(st.hf(RMR_REPO, "test.csv", rev), encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    return redmadrobot_map(seeded_sample(redmadrobot_upstream(csv_rows), st.limit), st)


# --------------------------------------------------------------------------- alrosait/pii-synthetic-ru

ALROSA_MAP = {"NAME": "PII.PERSON", "ADDRESS": "PII.ADDRESS"}


def alrosait_row(r, st):
    text = r["text"]
    ents, taken = [], []
    # longest first; label every whole-word occurrence of each distinct surface
    for surf, typ in sorted({(e["text"], e["type"]) for e in r["entities"]}, key=lambda x: -len(x[0])):
        n = 0
        for m in re.finditer(r"(?<!\w)" + re.escape(surf) + r"(?!\w)", text):
            if any(m.start() < b and a < m.end() for a, b in taken):
                continue
            taken.append((m.start(), m.end()))
            ents.append(ent(text, m.start(), m.end(), ALROSA_MAP[typ])); n += 1
        if n == 0:
            st.dropped["%s (surface not found as whole word)" % typ] += 1
        elif n > 1:
            st.notes["repeated surface labeled at every occurrence"] += n - 1
    cat = (r.get("domain") or "").lower()
    if not r["entities"]:
        cat += "/negative" + (":%s" % r["neg_category"] if r.get("neg_category") else "")
    return row("alrosait-%s" % r["id"], "alrosait/pii-synthetic-ru", "train(upstream)", cat, text, finalize(ents))


def conv_alrosait(rev, st):
    rows = read_jsonl(st.hf("alrosait/pii-synthetic-ru", "synthetic_pii.jsonl", rev))
    random.Random(SEED).shuffle(rows)
    return [alrosait_row(r, st) for r in rows[:st.limit]]


# --------------------------------------------------------------------------- factRuEval-2016 (persons)

FRE_REPO = "dialogue-evaluation/factRuEval-2016"
FRE_NAME_SPANS = {"name", "surname", "patronymic", "nickname"}


def factrueval_doc(d, text, spans_txt, objects_txt, st, chunk_chars=3000):
    """One FactRuEval document (.txt/.spans/.objects contents) -> rows chunked on paragraphs."""
    text = text.replace("\r\n", "\n")
    spans = {}
    for line in spans_txt.splitlines():
        p = line.split("#")[0].split()
        if len(p) >= 4:
            spans[p[0]] = (p[1], int(p[2]), int(p[2]) + int(p[3]))
    ents = []
    for line in objects_txt.splitlines():
        p = line.split("#")[0].split()
        if len(p) < 3 or p[1] != "Person":
            continue
        parts = sorted(spans[s][1:] for s in p[2:] if s in spans and spans[s][0] in FRE_NAME_SPANS)
        if not parts:
            continue
        groups = [list(parts[0])]
        for s, e in parts[1:]:
            if re.fullmatch(r"[\s.,-]*", text[groups[-1][1]:s]) and s >= groups[-1][1]:
                groups[-1][1] = max(groups[-1][1], e)
            else:
                groups.append([s, e])
        for s, e in groups:
            ents.append(ent(text, s, e, "PII.PERSON"))
    ents = finalize(ents)
    # chunk on paragraph boundaries so rows stay eval-sized
    paras = [m.span() for m in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text)]
    chunks, cur = [], None
    for s, e in paras:
        if cur and e - cur[0] > chunk_chars:
            chunks.append(cur); cur = None
        cur = [cur[0], e] if cur else [s, e]
    if cur:
        chunks.append(cur)
    split = "dev" if d.startswith("devset") else "test"
    out = []
    for i, (cs, ce) in enumerate(chunks):
        sub = [e for e in ents if cs <= e["start"] and e["end"] <= ce]
        crossing = [e for e in ents if e["start"] < ce and e["end"] > cs and e not in sub]
        if crossing:
            st.notes["entities crossing chunk (dropped)"] += len(crossing)
        out.append(row("factrueval-%s-%02d" % (d.replace("/", "-"), i), "factRuEval-2016", split, "news",
                       text[cs:ce], [dict(e, start=e["start"] - cs, end=e["end"] - cs) for e in sub]))
    return out


def conv_factrueval(rev, st):
    base = "https://raw.githubusercontent.com/%s/%s" % (FRE_REPO, rev)
    with open(st.fetch("https://api.github.com/repos/%s/git/trees/%s?recursive=1" % (FRE_REPO, rev),
                       "factrueval/%s/tree.json" % rev), encoding="utf-8") as f:
        tree = json.load(f)["tree"]
    docs = sorted(x["path"][:-4] for x in tree
                  if x["path"].endswith(".txt") and x["path"].split("/")[0] in ("devset", "testset"))
    if st.limit is not None and len(docs) > st.limit:  # every document gives >= 1 row
        docs = sorted(random.Random(SEED).sample(docs, st.limit))
    want = [(d, ext) for d in docs for ext in ("txt", "spans", "objects")]
    with ThreadPoolExecutor(16) as pool:
        got = dict(zip(want, pool.map(
            lambda de: st.fetch("%s/%s.%s" % (base, de[0], de[1]), "factrueval/%s/%s.%s" % (rev, de[0], de[1])), want)))
    out = []
    for d in docs:
        if any(got[(d, ext)] is None for ext in ("txt", "spans", "objects")):
            st.notes["document without .spans/.objects upstream (skipped)"] += 1
            continue

        def rd(ext):
            with open(got[(d, ext)], encoding="utf-8-sig") as f:
                return f.read()

        out.extend(factrueval_doc(d, rd("txt"), rd("spans"), rd("objects"), st))
    return seeded_sample(out, st.limit)


# --------------------------------------------------------------------------- islomov/rubai-NER-150K-Personal

RUBAI_MAP = {"NAME": "PII.PERSON", "PHONE": "PII.PHONE", "ADDRESS": "PII.ADDRESS", "CARD_NUMBER": "PII.BANK_CARD"}
UZ = re.compile(r"[ʻʼ‘’ўқғҳЎҚҒҲ]|\b(va|uchun|bilan|raqam|manzil|kvartira|ko'cha|uy|soat|metr)\b", re.I)


def rubai_candidate(r, st):
    """-> token spans when the row is kept (Russian, no DOCUMENT_ID, tokens align), else None."""
    t = r["original"]
    cyr = len(re.findall("[а-яё]", t, re.I)); lat = len(re.findall("[a-z]", t, re.I))
    if cyr < 0.95 * (cyr + lat) or UZ.search(t):
        return None
    if "DOCUMENT_ID" in r["labels"]:  # can't map reliably (UZ passport / personal no. / other)
        st.dropped["DOCUMENT_ID (whole row skipped)"] += 1
        return None
    toks = [m.span() for m in re.finditer(r"\S+", t)]
    if len(toks) != len(r["types"]):
        st.notes["token/label length mismatch (row skipped)"] += 1
        return None
    return toks


def rubai_row(r, toks, st):
    t, ents, cur = r["original"], [], None
    for (s, e), lab in zip(toks, r["types"]):
        if cur and cur[0] == lab:
            cur[2] = e
        else:
            if cur:
                ents.append(cur)
            cur = [lab, s, e]
    if cur:
        ents.append(cur)
    es = []
    for lab, s, e in ents:
        if lab == "TEXT":
            continue
        if lab not in RUBAI_MAP:
            st.dropped[lab] += 1
            continue
        # strip trailing/leading punctuation carried by whitespace tokens
        while e > s and t[e - 1] in ".,;:!?)»\"'": e -= 1
        while s < e and t[s] in "(«\"'": s += 1
        es.append(ent(t, s, e, RUBAI_MAP[lab]))
    return row("rubai-%s" % r["id"], "islomov/rubai-NER-150K-Personal", "train(upstream)",
               "uz_context_synthetic/%s" % (r.get("domain") or ""), t, finalize(es))


def conv_rubai(rev, st):
    cand = []
    with open(st.hf("islomov/rubai-NER-150K-Personal", "data.jsonl", rev), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            toks = rubai_candidate(r, st)
            if toks is not None:
                cand.append((r, toks))
    random.Random(SEED).shuffle(cand)
    return [rubai_row(r, toks, st) for r, toks in cand[:st.limit]]


# --------------------------------------------------------------------------- span composition (parts -> wholes)

UK_LETTERS = re.compile(r"[іїєґІЇЄҐўЎ]")


def is_ru(text, min_cyr=20):
    """Russian-only row: lang_of == ru, enough Cyrillic, no Ukrainian/Belarusian letters."""
    return (lang_of(text) == "ru" and len(re.findall(r"[а-яё]", text, re.I)) >= min_cyr
            and not UK_LETTERS.search(text))


def compose(text, spans, st, wrappers, parts, keep_part_group, simple):
    """Map upstream spans [(start, end, label)] to our entities.

    wrappers: {label: type}  - whole-entity spans (full name / full address);
    parts:    {label: (cls, type)} - fragments; a fragment inside a wrapper of the same cls is
              dropped, the rest are merged with adjacent fragments of the same cls
              (NAME: whitespace / ', ' gap; ADDR: separators, abbreviations, index) and the
              group becomes `type` if keep_part_group(cls, labels_in_group) else dropped;
    simple:   {label: type | 'PII.PASSPORT.*' | None}; None = dropped (counted).
    Labels in none of the maps are dropped and counted too.
    """
    ok = []
    for s, e, lab in spans:
        if not (0 <= s < e <= len(text)):
            st.notes["out-of-range/empty upstream spans skipped"] += 1
            continue
        ok.append((s, e, lab))
    cls_of_wrapper = {lab: ("NAME" if t == "PII.PERSON" else "ADDR") for lab, t in wrappers.items()}
    wraps = [(s, e, cls_of_wrapper[lab]) for s, e, lab in ok if lab in wrappers]
    ents, frags = [], []
    for s, e, lab in ok:
        if lab in wrappers:
            ents.append(ent(text, s, e, wrappers[lab]))
        elif lab in parts:
            cls = parts[lab][0]
            if any(ws <= s and e <= we and wc == cls for ws, we, wc in wraps):
                st.notes["%s fragment nested in full span (dropped)" % cls] += 1
            else:
                frags.append((s, e, lab))
        elif simple.get(lab):
            t = simple[lab]
            if t == "PII.PASSPORT.*":
                t = passport_type(text, s)
            ents.append(ent(text, s, e, t))
        else:
            st.dropped[lab] += 1
    groups = []  # [cls, start, end, labels]
    for s, e, lab in sorted(frags):
        cls = parts[lab][0]
        g = groups[-1] if groups else None
        if g and g[0] == cls and s >= g[2] and lab not in g[3]:
            gap = text[g[2]:s]
            if (cls == "NAME" and NAME_GAP.match(gap)) or (cls == "ADDR" and addr_gap_ok(gap)):
                g[2] = e; g[3].append(lab); st.notes["%s fragments merged" % cls] += 1
                continue
        groups.append([cls, s, e, [lab]])
    for cls, s, e, labs in groups:
        if keep_part_group(cls, set(labs)):
            ents.append(ent(text, s, e, next(parts[x][1] for x in labs)))
        else:
            for x in labs:
                st.dropped["%s (standalone fragment)" % x] += 1
    return finalize(ents)


ADDR_CORE_LABELS = {"STREET", "HOUSE", "address_street", "address_house",
                    "STREET_ADDRESS", "STREET_NAME", "BUILDING_NUMBER"}


def addr_keep(cls, labs):
    return cls == "NAME" or bool(labs & ADDR_CORE_LABELS)


# --------------------------------------------------------------------------- Powpowpow23/ru-pii-ner-data (validation)

POW_WRAP = {"FULL_NAME": "PII.PERSON", "CARDHOLDER_NAME": "PII.PERSON", "ADDRESS": "PII.ADDRESS"}
POW_PARTS = {k: ("ADDR", "PII.ADDRESS") for k in
             ("COUNTRY", "POSTCODE", "REGION", "DISTRICT", "CITY", "STREET", "HOUSE", "APARTMENT")}
POW_SIMPLE = {"DATE_OF_BIRTH": "PII.DATE_OF_BIRTH", "PASSPORT": "PII.PASSPORT.*",
              "DRIVER_LICENSE": "PII.DRIVER_LICENSE", "EMAIL": "PII.EMAIL", "INN": "PII.INN",
              "CARD_NUMBER": "PII.BANK_CARD", "CVV": "PII.CVC"}


def conv_powpow(rev, st):
    rows = parquet_rows(st.hf("Powpowpow23/ru-pii-ner-data", "data/validation.parquet", rev))
    cand = []
    for r in rows:
        if not is_ru(r["text"]):
            st.notes["non-Russian row (lang != ru) skipped"] += 1
            continue
        cand.append(r)
    random.Random(SEED).shuffle(cand)
    out = []
    for r in cand[:st.limit]:
        text = r["text"]
        spans = [(e["start"], e["end"], e["type"]) for e in r["entities"]]
        fam = "+".join(r["source_families"])
        out.append(row("powpow-%s" % r["id"], "Powpowpow23/ru-pii-ner-data", "validation(upstream)",
                       "synthetic/%s" % fam + ("/negative" if not spans else ""),
                       text, compose(text, spans, st, POW_WRAP, POW_PARTS, addr_keep, POW_SIMPLE)))
    return out


# --------------------------------------------------------------------------- scanpatch/pii-ner-corpus-synthetic-controlled

def label_repeats(text, ents, st):
    """upstream labels a repeated value once ('precision over recall'); label every whole-word repeat."""
    extra = []
    for surf, typ in sorted({(e["text"], e["type"]) for e in ents}, key=lambda x: (-len(x[0]), x)):
        for m in re.finditer(r"(?<!\w)" + re.escape(surf) + r"(?!\w)", text):
            if any(m.start() < e["end"] and e["start"] < m.end() for e in ents + extra):
                continue
            extra.append(ent(text, m.start(), m.end(), typ))
            st.notes["unlabeled repeat of a labeled value (labeled)"] += 1
    return finalize(ents + extra)


SCAN_WRAP = {"name": "PII.PERSON", "address": "PII.ADDRESS"}
SCAN_PARTS = dict(
    [(k, ("NAME", "PII.PERSON")) for k in ("first_name", "middle_name", "last_name", "name_initials")]
    + [(k, ("ADDR", "PII.ADDRESS")) for k in
       ("address_country", "address_region", "address_city", "address_district", "address_street",
        "address_house", "address_building", "address_apartment", "address_postal_code")])
SCAN_SIMPLE = {"mobile_phone": "PII.PHONE", "email": "PII.EMAIL", "ip": "NETWORK.IP", "tin": "PII.INN",
               "snils": "PII.SNILS", "vehicle_number": "PII.VEHICLE_REGISTRATION"}
SCAN_SKIP_ROW = {"document_number", "military_individual_number"}


def scanpatch_row(r, split, i, st):
    text = r["text"]
    if not is_ru(text):
        st.notes["non-Russian row (uk/en/mixed) skipped"] += 1
        return None
    labs = set(r["entity_labels"])
    if labs & SCAN_SKIP_ROW:
        for x in labs & SCAN_SKIP_ROW:
            st.dropped["%s (whole row skipped)" % x] += 1
        return None
    spans = list(zip(r["entity_starts"], r["entity_ends"], r["entity_labels"]))
    ents = label_repeats(text, compose(text, spans, st, SCAN_WRAP, SCAN_PARTS, addr_keep, SCAN_SIMPLE), st)
    h = hashlib.sha1(text.encode()).hexdigest()[:12]
    return row("scanpatch-%s-%05d-%s" % (split, i, h), "scanpatch/pii-ner-corpus-synthetic-controlled",
               "%s(upstream)" % split, "synthetic_controlled" + ("/negative" if not ents else ""), text, ents)


def conv_scanpatch(rev, st):
    out = []
    for split in ("test", "train"):
        path = st.hf("scanpatch/pii-ner-corpus-synthetic-controlled", "data/%s-00000-of-00001.parquet" % split, rev)
        for i, r in enumerate(parquet_rows(path)):
            x = scanpatch_row(r, split, i, st)
            if x is not None:
                out.append(x)
    seen, uniq = set(), []
    for r in out:  # train/test share a few identical sentences
        if r["text"] in seen:
            st.notes["duplicate text skipped"] += 1
            continue
        seen.add(r["text"]); uniq.append(r)
    return seeded_sample(uniq, st.limit)


# --------------------------------------------------------------------------- Wismut/nym-pii-multilingual-data (ru)

NYM_PARTS = dict(
    [("GIVEN_NAME", ("NAME", "PII.PERSON")), ("SURNAME", ("NAME", "PII.PERSON")),
     ("MIDDLE_NAME", ("NAME", "PII.PERSON"))]
    + [(k, ("ADDR", "PII.ADDRESS")) for k in
       ("STREET_ADDRESS", "STREET_NAME", "BUILDING_NUMBER", "SECONDARY_ADDRESS", "CITY", "STATE",
        "ZIP_CODE", "COUNTRY")])
NYM_SIMPLE = {"EMAIL": "PII.EMAIL", "PHONE": "PII.PHONE", "PASSPORT": "PII.PASSPORT.*",
              "DATE_OF_BIRTH": "PII.DATE_OF_BIRTH", "CREDIT_DEBIT_CARD": "PII.BANK_CARD", "CVV": "PII.CVC",
              "MAC_ADDRESS": "NETWORK.MAC", "IP_ADDRESS": "NETWORK.IP", "USERNAME": "PII.LOGIN",
              "LICENSE_PLATE": "PII.VEHICLE_REGISTRATION", "FAX_NUMBER": "PII.PHONE"}


def nym_row(r, split, i, st):
    text = r["text"]
    if not is_ru(text):
        return None
    spans = [(e["start"], e["end"], e["label"]) for e in r["entities"]]
    ents = compose(text, spans, st, {}, NYM_PARTS, addr_keep, NYM_SIMPLE)
    return row("nym-%s-%06d" % (split, i), "Wismut/nym-pii-multilingual-data", "%s(upstream)" % split,
               "synthetic_faker_templates" + ("/negative" if not ents else ""), text, ents)


def conv_nym(rev, st):
    out = []
    for split in ("test", "validation"):
        with open(st.hf("Wismut/nym-pii-multilingual-data", "%s.jsonl" % split, rev), encoding="utf-8") as f:
            for i, line in enumerate(f):
                x = nym_row(json.loads(line), split, i, st)
                if x is not None:
                    out.append(x)
    return seeded_sample(out, st.limit)


# --------------------------------------------------------------------------- registry

PARTIAL = "partial: only scored_labels are gold, ignore predictions of other types"

SPECS = [
    dict(key="hivetrace", name="hivetrace_pii_bench", fn=conv_hivetrace, source=HIVE_REPO,
         url="https://huggingface.co/datasets/hivetrace/pii-bench",
         revision="cd6a18ace16daf23e79247ccf1e2b245d4054654", license="Apache-2.0", limit=None,
         scored_labels=ALL, labeled="full"),
    dict(key="redmadrobot", name="redmadrobot_pii_benchmark", fn=conv_redmadrobot, source=RMR_REPO,
         url="https://huggingface.co/datasets/redmadrobot-rnd/pii_benchmark",
         revision="f77ea831274daf980cc45c61a93c226be9d978d6", license="MIT", limit=None,
         scored_labels=ALL, labeled="full",
         caveat="in-family for redmadrobot-rnd/rubert-base-pii-ner"),
    dict(key="alrosait", name="alrosait_pii_synthetic_ru", fn=conv_alrosait, source="alrosait/pii-synthetic-ru",
         url="https://huggingface.co/datasets/alrosait/pii-synthetic-ru",
         revision="f4ca8b47473e6130fb9ff5b84e41a2844644c9fd", license="MIT", limit=3000,
         scored_labels=["PII.ADDRESS", "PII.PERSON"], labeled=PARTIAL,
         caveat="LLM-generated; emails/phones in text are NOT labeled"),
    dict(key="factrueval", name="factrueval2016_person", fn=conv_factrueval, source="factRuEval-2016",
         url="https://github.com/dialogue-evaluation/factRuEval-2016",
         revision="dfe96237fce79fde13b2021d1938587e4ea3042e", license="MIT", limit=None,
         scored_labels=["PII.PERSON"], labeled=PARTIAL,
         caveat="news; public figures; Location/Org are not addresses"),
    dict(key="rubai", name="rubai_ner_ru_subset", fn=conv_rubai, source="islomov/rubai-NER-150K-Personal",
         url="https://huggingface.co/datasets/islomov/rubai-NER-150K-Personal",
         revision="05f279e6dd8ee50e43f2a4e55bdac05d847983de", license="Apache-2.0", limit=3000,
         scored_labels=["PII.ADDRESS", "PII.BANK_CARD", "PII.PERSON", "PII.PHONE"], labeled=PARTIAL,
         caveat="synthetic Russian with Uzbek context; token-level model-distilled labels"),
    dict(key="powpow", name="powpow_ru_pii_ner_validation", fn=conv_powpow, source="Powpowpow23/ru-pii-ner-data",
         url="https://huggingface.co/datasets/Powpowpow23/ru-pii-ner-data",
         revision="55acbec07fb04a18455111a34337cb515a7f821e", license="Apache-2.0", limit=3000,
         scored_labels=["PII.ADDRESS", "PII.BANK_CARD", "PII.CVC", "PII.DATE_OF_BIRTH", "PII.DRIVER_LICENSE",
                        "PII.EMAIL", "PII.INN", "PII.PASSPORT.FOREIGN", "PII.PASSPORT.RU", "PII.PERSON"],
         labeled=PARTIAL, caveat="phones are not scored (upstream replaced them with non-routable 000... numbers)"),
    dict(key="scanpatch", name="scanpatch_pii_ner_ru", fn=conv_scanpatch,
         source="scanpatch/pii-ner-corpus-synthetic-controlled",
         url="https://huggingface.co/datasets/scanpatch/pii-ner-corpus-synthetic-controlled",
         revision="6af0587dde068d90d8a1080285159c374b46a09b", license="MIT", limit=3000,
         scored_labels=["NETWORK.IP", "PII.ADDRESS", "PII.EMAIL", "PII.INN", "PII.PERSON", "PII.PHONE",
                        "PII.SNILS", "PII.VEHICLE_REGISTRATION"], labeled=PARTIAL,
         caveat="LLM-annotated synthetic; Ukrainian context in many rows"),
    dict(key="nym", name="nym_pii_multilingual_ru", fn=conv_nym, source="Wismut/nym-pii-multilingual-data",
         url="https://huggingface.co/datasets/Wismut/nym-pii-multilingual-data",
         revision="abe23bf08c305f5824b1643dc4f079f06ccb2ad1", license="MIT", limit=3000,
         scored_labels=["NETWORK.IP", "NETWORK.MAC", "PII.ADDRESS", "PII.BANK_CARD", "PII.CVC", "PII.DATE_OF_BIRTH",
                        "PII.EMAIL", "PII.LOGIN", "PII.PASSPORT.FOREIGN", "PII.PASSPORT.RU", "PII.PERSON",
                        "PII.PHONE", "PII.VEHICLE_REGISTRATION"], labeled=PARTIAL,
         caveat="Faker templates; ~20% typo noise inside labeled values"),
]
