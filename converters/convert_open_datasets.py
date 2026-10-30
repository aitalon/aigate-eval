#!/usr/bin/env python3
"""Скачивает открытые датасеты (закреплённые ревизии) и конвертирует их в JSONL-формат бенчмарков.

PII-наборы  -> <out-dir>/pii/<name>.jsonl
инъекции    -> <out-dir>/injection/adr_bench_en.jsonl

Примеры:
    python3 -m pip install -r converters/requirements.txt
    python3 converters/convert_open_datasets.py --list
    python3 converters/convert_open_datasets.py all --adr /path/to/ADR
    python3 converters/convert_open_datasets.py hivetrace redmadrobot --limit 200

Для набора инъекций нужен локальный клон github.com/uber/ADR на закреплённом коммите:
    git clone https://github.com/uber/ADR /path/to/ADR
    git -C /path/to/ADR checkout 3782cea30a2f8ca88f266a5704194d603ae6cf1f
    python3 -I converters/convert_open_datasets.py adr --adr /path/to/ADR
Конвертер только читает файлы клона и ничего из него не исполняет.
"""
import argparse
import collections
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:  # sibling modules; python3 -I does not add the script dir
    sys.path.insert(0, HERE)

import adr_bench  # noqa: E402
import pii_sources  # noqa: E402

REPO = os.path.dirname(HERE)
DEFAULT_OUT = os.path.join(REPO, "benchmarks")
DEFAULT_CACHE = os.path.join(REPO, ".cache")

ADR_SPEC = dict(
    key="adr", name="adr_bench_en", source="uber/ADR (ADR-Bench + bundled AgentDojo run)",
    url=adr_bench.ADR_URL, revision=adr_bench.ADR_COMMIT,
    license="Apache-2.0 (ADR); bundled AgentDojo data MIT (ethz-spylab/agentdojo)")

EPILOG = """\
наборы:
%s
  adr          -> injection/adr_bench_en.jsonl  uber/ADR @ %s (нужен --adr)

Все данные — только для оценки, не для обучения. Метки — исходные метки датасетов,
сведённые к нашим типам (PII.PERSON, PII.PHONE, ...), без ручной переразметки.
Часть наборов размечена не по всем типам: список оцениваемых типов (scored_labels)
выводит --list и записывает <out-dir>/open_datasets_manifest.json.

Русская версия ADR-Bench (adr_bench_ru) — это LLM-перевод EN-строк; из открытых
источников он не воспроизводится и этим скриптом не создаётся.
""" % ("\n".join("  %-12s -> pii/%s.jsonl  %s @ %s" % (s["key"], s["name"], s["source"], s["revision"][:12])
                 for s in pii_sources.SPECS), adr_bench.ADR_COMMIT[:12])


def shown(path):
    try:
        rel = os.path.relpath(path)
    except ValueError:  # another drive on Windows
        return path
    return path if rel.startswith("..") else rel


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def run_pii(spec, out_dir, cache, limit):
    st = pii_sources.Ctx(cache, spec["limit"] if limit is None else limit)
    rows = spec["fn"](spec["revision"], st)
    pii_sources.validate(rows)
    path = os.path.join(out_dir, "pii", spec["name"] + ".jsonl")
    digest = write_jsonl(path, rows)
    counts = collections.Counter(e["type"] for r in rows for e in r["entities"])
    print("  %-30s rows=%5d with_pii=%5d entities=%6d -> %s" % (
        spec["name"], len(rows), sum(1 for r in rows if r["entities"]), sum(counts.values()),
        shown(path)))
    if st.dropped:
        top = st.dropped.most_common(8)
        more = len(st.dropped) - len(top)
        print("      upstream labels not mapped (dropped): %s%s" % (
            ", ".join("%s=%d" % kv for kv in top), " (+%d more)" % more if more > 0 else ""))
    entry = {k: v for k, v in spec.items() if k not in ("fn", "key", "limit")}
    entry.update(kind="pii", file=os.path.relpath(path, out_dir), limit=st.limit, rows=len(rows),
                 rows_with_entities=sum(1 for r in rows if r["entities"]),
                 rows_negative=sum(1 for r in rows if not r["entities"]),
                 entity_counts=dict(counts.most_common()), dropped_types=dict(st.dropped.most_common()),
                 notes=dict(st.notes.most_common()), sha256=digest)
    return entry


def run_adr(adr, out_dir, allow_other_commit):
    adr = os.path.abspath(adr)
    head = adr_bench.check_clone(adr, allow_other_commit)
    rows, log, stats = adr_bench.build(adr)
    rows.sort(key=lambda r: r["pair"])
    path = os.path.join(out_dir, "injection", ADR_SPEC["name"] + ".jsonl")
    digest = write_jsonl(path, rows)
    by = collections.Counter("%s/%s" % (r["kind"], r["label"]) for r in rows)
    print("  %-30s rows=%5d %s -> %s" % (ADR_SPEC["name"], len(rows), dict(sorted(by.items())), shown(path)))
    if log.get("screen_unexplained"):
        print("      excluded as ambiguous (instruction-like text not in spec): %d" % len(log["screen_unexplained"]))
    entry = {k: v for k, v in ADR_SPEC.items() if k != "key"}
    entry.update(kind="injection", file=os.path.relpath(path, out_dir), clone_head=head, rows=len(rows),
                 by_kind_label=dict(sorted(by.items())),
                 by_split=dict(sorted(collections.Counter(r["split"] for r in rows).items())),
                 excluded_rows=stats["excluded"], sha256=digest)
    return entry


def update_manifest(out_dir, entries):
    path = os.path.join(out_dir, "open_datasets_manifest.json")
    old = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                old = {d["name"]: d for d in json.load(f).get("datasets", [])}
        except (ValueError, KeyError, TypeError):
            old = {}
    old.update((e["name"], e) for e in entries)
    manifest = {
        "warning": "Evaluation only. Do not train on these files.",
        "generator": "converters/convert_open_datasets.py",
        "semantics": {"entities=[]": "labeled row without PII",
                      "scored_labels": "only these types are gold for the file; ignore predictions of other types"},
        "datasets": sorted(old.values(), key=lambda d: (d.get("kind", ""), d["name"])),
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=1) + "\n"
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def print_list():
    for s in pii_sources.SPECS:
        lim = "все строки" if s["limit"] is None else "до %d строк" % s["limit"]
        print("%-12s pii/%s.jsonl" % (s["key"], s["name"]))
        print("             %s  rev %s  (%s, %s)" % (s["url"], s["revision"], s["license"], lim))
        print("             scored_labels: %s" % (
            "все типы" if s["scored_labels"] == pii_sources.ALL else ", ".join(s["scored_labels"])))
    print("%-12s injection/%s.jsonl" % ("adr", ADR_SPEC["name"]))
    print("             %s  commit %s  (%s; нужен --adr)" % (ADR_SPEC["url"], ADR_SPEC["revision"], ADR_SPEC["license"]))


def main(argv=None):
    keys = [s["key"] for s in pii_sources.SPECS] + ["adr"]
    by_name = dict((s["name"], s["key"]) for s in pii_sources.SPECS)
    by_name[ADR_SPEC["name"]] = "adr"
    ap = argparse.ArgumentParser(
        description="Скачивает открытые датасеты (закреплённые ревизии) и пишет их в формате бенчмарков.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", metavar="DATASET",
                    help="ключи наборов (%s) или 'all'" % ", ".join(keys))
    ap.add_argument("--list", action="store_true", help="показать наборы, ревизии и оцениваемые типы")
    ap.add_argument("--out-dir", default=DEFAULT_OUT, help="куда писать (по умолчанию: %(default)s)")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE, help="кэш загрузок (по умолчанию: %(default)s)")
    ap.add_argument("--limit", type=int, default=None,
                    help="максимум строк на PII-набор (по умолчанию: лимит набора, обычно 3000 или все); "
                         "на adr не влияет")
    ap.add_argument("--adr", help="путь к клону github.com/uber/ADR на коммите %s" % adr_bench.ADR_COMMIT[:12])
    ap.add_argument("--allow-other-commit", action="store_true",
                    help="не проверять коммит и sha256 файлов клона ADR (результат не будет воспроизводимым)")
    a = ap.parse_args(argv)
    if a.list:
        print_list()
        return 0
    if not a.datasets:
        ap.print_help()
        return 2
    if a.limit is not None and a.limit < 1:
        ap.error("--limit должен быть >= 1")
    want = []
    for d in a.datasets:
        if d == "all":
            want.extend(k for k in keys if k != "adr" or a.adr)
            if not a.adr:
                print("adr пропущен: укажите --adr /path/to/ADR (см. --help)", file=sys.stderr)
            continue
        k = by_name.get(d, d)
        if k not in keys:
            ap.error("неизвестный набор %r; доступны: %s" % (d, ", ".join(keys)))
        want.append(k)
    want = [k for k in keys if k in set(want)]  # registry order, no duplicates
    if "adr" in want and not a.adr:
        ap.error("для adr нужен --adr /path/to/ADR (клон github.com/uber/ADR)")
    out_dir = os.path.abspath(a.out_dir)
    cache = os.path.abspath(a.cache_dir)
    specs = dict((s["key"], s) for s in pii_sources.SPECS)
    entries, failed = [], []
    for k in want:
        t0 = time.time()
        print("%s ..." % k, flush=True)
        try:
            if k == "adr":
                entries.append(run_adr(a.adr, out_dir, a.allow_other_commit))
            else:
                entries.append(run_pii(specs[k], out_dir, cache, a.limit))
        except ImportError as ex:
            print("  ошибка: %s (python3 -m pip install -r converters/requirements.txt)" % ex, file=sys.stderr)
            failed.append(k)
        except Exception as ex:  # one broken source must not hide the others
            print("  ошибка: %s: %s" % (type(ex).__name__, ex), file=sys.stderr)
            failed.append(k)
        else:
            print("  готово за %.1f с" % (time.time() - t0))
    if entries:
        print("manifest: %s" % shown(update_manifest(out_dir, entries)))
    if failed:
        print("не удалось: %s" % ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
