"""Uber ADR-Bench (+ its bundled AgentDojo run) -> benchmarks/injection/adr_bench_en.jsonl rows (EN).

Labeling logic and spec are the ones used for our internal evaluation, unchanged.

Source: https://github.com/uber/ADR (Apache-2.0), pinned commit ADR_COMMIT. The clone is untrusted
data: this module only reads files (json / a tiny yaml line parser / ast.parse of the MCP server
sources); nothing from the clone is imported or executed. Run the CLI with an isolated
interpreter from outside the clone (python3 -I ...).

The row-labeling spec for the deliberately vulnerable / environment MCP servers is
adr_bench_spec.json next to this file. Rows keep the upstream dev/test `split` assignment of
the internal evaluation (groups never span dev and test); all rows may be used.

A Russian variant of these rows exists internally (an LLM translation, 1:1 by `pair`); it
cannot be regenerated from public sources and is not produced here.
"""

import ast
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

ADR_COMMIT = "3782cea30a2f8ca88f266a5704194d603ae6cf1f"
ADR_URL = "https://github.com/uber/ADR"
SRC_SHA256 = {
    "Detection/tasks.json":
        "3856b9e31119ded83dd3951b9439b028da80af3dd40d1b23662e3f82420b6d39",
    "Detection/mcp_servers_registry.json":
        "d38c5aac7f6c1ce49fb45b3039c3e99d717d2cd4add2c77904f88a992cab0808",
    "Detection/benchmark/adr_bench_20251017_151604.jsonl":
        "9c94766522849e5b796e22a647931aebcdbbb7ae96ea1341a675583e3f1147c3",
    "Detection/benchmark/agentdojo_20251022_190150.jsonl":
        "68cdd5535d1d1b167d5d93bd98f16f16aa56e9743711928fc2e76363a5685aec",
    "Detection/context_providers/data/threat_repository.yaml":
        "023c03b57d2d46b157e33f01554d623b1fb1611a95a7b61f9e78fd5eb7a41983",
}
HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "adr_bench_spec.json")
MAX_CHARS = 3000          # tool_result / tool_description cap (window kept around the attack marker)
BENIGN_TR_PER_TASK = 1    # benign tool results kept per ADR task besides environment-server outputs
SRC_ADR = "uber/adr-bench"
SRC_DOJO = "agentdojo/important_instructions (via uber/adr-bench)"

# ----------------------------------------------------------------------------- source access


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def head_commit(adr):
    """Read HEAD without running git inside the untrusted clone."""
    g = os.path.join(adr, ".git")
    head = open(os.path.join(g, "HEAD")).read().strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    p = os.path.join(g, ref)
    if os.path.exists(p):
        return open(p).read().strip()
    for line in open(os.path.join(g, "packed-refs")):
        parts = line.split()
        if len(parts) == 2 and parts[1] == ref:
            return parts[0]
    raise ValueError("cannot resolve HEAD of the ADR clone")


def load_taxonomy(adr):
    """technique name -> (id, tactic) from threat_repository.yaml (tiny line parser, no yaml import)."""
    tactic, tid, out = None, None, {}
    in_tactics = False
    for line in open(os.path.join(adr, "Detection/context_providers/data/threat_repository.yaml"), encoding="utf-8"):
        if line.startswith("  tactics:"):
            in_tactics = True
            continue
        if not in_tactics:
            continue
        m = re.match(r"^    ([a-z_]+):\s*$", line)
        if m:
            tactic = m.group(1)
            continue
        m = re.match(r"^      - id: (ADR\.T\d+)", line)
        if m:
            tid = m.group(1)
            continue
        m = re.match(r"^        name: (.+?)\s*$", line)
        if m and tid:
            out[m.group(1).strip("'\"")] = (tid, tactic)
            tid = None
    return out


# ----------------------------------------------------------------------------- helpers


def h8(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def clip(text, pos=None, limit=MAX_CHARS):
    """Cap text; when an attack marker position is known keep a window around it."""
    if len(text) <= limit:
        return text, False
    if pos is None or pos < limit // 2:
        return text[:limit], True
    start = max(0, min(pos - limit // 3, len(text) - limit))
    return text[start:start + limit], True


def adr_tool_text(content):
    """ADR tool_result content is a Python repr of Claude content blocks; return what the model saw."""
    try:
        v = ast.literal_eval(content)
    except (ValueError, SyntaxError):
        # recorder truncated the repr at ~10k chars: fall back to the raw text fields
        parts = re.findall(r"'text': '((?:[^'\\]|\\.)*)", content)
        if not parts:
            return content
        return "\n".join(p.encode("utf-8").decode("unicode_escape", "ignore") if "\\u" in p else
                         p.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")
                         for p in parts)
    texts = []
    for block in v if isinstance(v, list) else [v]:
        c = block.get("content") if isinstance(block, dict) else block
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    texts.append(b.get("text", ""))
                elif isinstance(b, dict):
                    texts.append(f"[{b.get('type', 'block')}]")
    return "\n".join(texts)


def tool_use_id(content):
    m = re.match(r"\[\{'tool_use_id': '([^']+)'", content)
    return m.group(1) if m else None


def split_tool_name(name):
    """mcp__<server>__<tool> -> (server, tool); built-ins (Read, Bash, ...) -> ('builtin', name)."""
    if name.startswith("mcp__"):
        rest = name[5:]
        srv, _, tool = rest.partition("__")
        return srv, tool
    return "builtin", name


class Matcher:
    """Labels from scripts/adr_bench_spec.json: per (server, tool) output markers -> output_label."""

    def __init__(self, spec):
        self.tools = defaultdict(list)
        for e in spec["tools"]:
            self.tools[e["server"]].append((e, [re.compile(p) for p in e.get("output_markers", [])]))
        self.desc = {(e["server"], e["tool"]): e for e in spec["descriptions"]}

    def label_output(self, server, tool, text):
        """-> (label, vector, pos, entry) for the first matching spec entry, else (None, None, None, None)."""
        for e, pats in self.tools.get(server, []):
            if e["tool"] not in ("*", tool, tool.lstrip("_")) and e["tool"].lstrip("_") != tool.lstrip("_"):
                continue
            pos = [m.start() for p in pats for m in [p.search(text)] if m]
            if pos:
                return e["output_label"], e["vector"], min(pos), e
        return None, None, None, None


# Screening for unlabeled rows that still look like instructions to the model. A screening hit
# that the spec does not explain is NOT guessed: the row becomes `ambiguous` (excluded) and is
# listed in the build log for review.
SCREEN = re.compile(
    r"(?i)(ignore (all |any )?(previous|prior|above) instructions|<important>|important[:!]\s*(before|you must|do not)"
    r"|you must (now )?(send|forward|email|upload|post|transfer|delete|execute|run)\b|do not (tell|inform|mention to) the user"
    r"|system prompt|<information>|before you can solve the task|new instructions?:)"
)


# ----------------------------------------------------------------------------- MCP tool descriptions


def extract_tools(adr):
    """All @mcp.tool functions: (rel_path, fastmcp_name, tool, docstring, input_schema)."""
    root = os.path.join(adr, "Detection/context_providers/source_codes")
    out = []
    for dp, dn, fn in os.walk(root):
        dn.sort()
        for f in sorted(fn):
            if not f.endswith(".py"):
                continue
            p = os.path.join(dp, f)
            tree = ast.parse(open(p, encoding="utf-8").read())
            fm = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and getattr(n.func, "id", getattr(n.func, "attr", None)) == "FastMCP"]
            sname = fm[0].args[0].value if fm and fm[0].args and isinstance(fm[0].args[0], ast.Constant) else None
            for n in ast.walk(tree):
                if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not any("tool" in ast.unparse(d) for d in n.decorator_list):
                    continue
                out.append((os.path.relpath(p, root), sname, n.name, ast.get_docstring(n) or "", schema_of(n)))
    out.sort(key=lambda x: (x[0], x[2]))
    return out


PYTYPE = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "dict": "object",
          "list": "array", "Dict": "object", "List": "array", "Any": None}


def schema_of(fn):
    """JSON schema in the shape FastMCP derives from the signature (types + defaults + required)."""
    props, req = {}, []
    args = fn.args.args
    defaults = [None] * (len(args) - len(fn.args.defaults)) + list(fn.args.defaults)
    for a, d in zip(args, defaults):
        if a.arg in ("self", "ctx", "context"):
            continue
        p = {}
        if a.annotation is not None:
            ann = ast.unparse(a.annotation)
            base = re.split(r"[\[|]", ann)[0].strip()
            base = base.split(".")[-1]
            if base == "Optional":
                inner = re.search(r"\[(.+)\]", ann)
                base = re.split(r"[\[|]", inner.group(1))[0].strip() if inner else "Any"
            t = PYTYPE.get(base, "string")
            if t:
                p["type"] = t
        if d is None:
            req.append(a.arg)
        else:
            try:
                p["default"] = ast.literal_eval(d)
            except ValueError:
                p["default"] = ast.unparse(d)
        props[a.arg] = p
    s = {"type": "object", "properties": props}
    if req:
        s["required"] = req
    return s


# ----------------------------------------------------------------------------- build


def build(adr):
    tax = load_taxonomy(adr)
    spec = json.load(open(SPEC, encoding="utf-8"))
    match = Matcher(spec)
    reg = json.load(open(os.path.join(adr, "Detection/mcp_servers_registry.json"), encoding="utf-8"))["servers"]
    src_of = {}  # source rel path -> registry names
    for name, s in reg.items():
        for a in s.get("args_template", []):
            m = re.search(r"source_codes/(.+\.py)$", a)
            if m:
                src_of.setdefault(m.group(1), []).append(name)
    tier = {n: rel.split("/")[0] for rel, names in src_of.items() for n in names}
    tasks = {t["task_id"]: t for t in json.load(open(os.path.join(adr, "Detection/tasks.json"), encoding="utf-8"))["tasks"]}
    rows = []
    excluded = defaultdict(list)     # reason -> [pair, ...]
    log = defaultdict(list)
    observable = defaultdict(list)   # malicious ADR task -> malicious rows that carry its attack

    def tech(name):
        tid, tac = tax.get(name, (None, None))
        return {"technique_id": tid, "technique": name, "tactic": tac}

    server_tech = defaultdict(Counter)
    for t in tasks.values():
        if t["ground_truth"] == "malicious":
            for s in t["mcp_servers"]:
                server_tech[s][t["threat_technique"]] += 1

    def technique_for(srv, task=None, override=None):
        if override:
            return tech(override)
        if task and task["ground_truth"] == "malicious":
            return tech(task["threat_technique"])
        if server_tech.get(srv):
            return tech(server_tech[srv].most_common(1)[0][0])
        return {"technique_id": None, "technique": None, "tactic": None}

    # split group: a deliberately malicious server keeps its descriptions and outputs on one side;
    # an environment injection (jira tickets) is one group; benign rows group by task / server
    def group_for(srv, t):
        if tier.get(srv) == "mcp_servers_1":
            return "srv:" + srv
        return f"adr-task:{t['task_id']}" if t else f"tdsrv:{srv}"

    def row(pair, origin, kind, text, label, attack=None, reason="", task=None, group=None, tool=None,
            truncated=False, category=None):
        return {
            "id": "en-" + pair, "pair": pair, "source": SRC_ADR if origin == "adr" else SRC_DOJO,
            "split": None, "category": category or kind, "language": "en", "text": text, "entities": None,
            "kind": kind, "label": label, "attack": attack, "label_reason": reason, "origin": origin,
            "task": task, "group": group, "tool": tool, "truncated": truncated,
        }

    # ---- ADR user prompts: benign tasks only (FPR). A malicious task's prompt is the victim's
    # request (payload comes from a tool), so it carries no attack and is not kept.
    for tid in sorted(tasks):
        t = tasks[tid]
        if t["ground_truth"] == "malicious":
            excluded["user_prompt of a malicious task (victim request, no attack text)"].append(f"adr-{tid:03d}")
            continue
        rows.append(row(f"adr-up-{tid:03d}", "adr", "user_prompt", t["user_prompt"], "benign", None, "benign task",
                        task=f"adr-{tid:03d}", group=f"adr-task:{tid}", category=f"user_prompt/{t['category']}"))

    # ---- ADR tool results
    lines = open(os.path.join(adr, "Detection/benchmark/adr_bench_20251017_151604.jsonl"), encoding="utf-8").read().splitlines()
    seen = set()
    per_task_benign = defaultdict(list)
    for line in lines[1:]:
        r = json.loads(line)
        tid = int(r["task_id"].split("_")[1])
        t = tasks[tid]
        names, n = {}, 0
        for msg in r["conversation"] or []:
            for tc in msg["tool_calls"]:
                names[tc["id"]] = tc["name"]
            if msg["message_type"] != "tool_result":
                continue
            n += 1
            full = adr_tool_text(msg["content"])
            srv, tool = split_tool_name(names.get(tool_use_id(msg["content"]), "builtin__unknown"))
            label, vec, pos, ent = match.label_output(srv, tool, full)
            text, trunc = clip(full, pos)
            key = text.strip()
            if not key or key in seen or key.startswith("<tool_use_error>"):
                continue
            seen.add(key)
            pair = f"adr-tr-{tid:03d}-{n:03d}"
            base = dict(task=f"adr-{tid:03d}", group=("env:jira" if srv == "jira_server" and label == "malicious" else group_for(srv, t)),
                        tool=f"{srv}.{tool}", truncated=trunc, category=f"tool_result/{tier.get(srv, srv)}")
            if label == "malicious":
                rows.append(row(pair, "adr", "tool_result", text, "malicious",
                                dict(technique_for(srv, t, ent.get("technique")), vector=vec),
                                f"spec: {srv}.{ent['tool']} ({vec})", **base))
                if t["ground_truth"] == "malicious":
                    observable[tid].append(pair)
                continue
            if label:
                excluded[f"tool_result {label}"].append(pair)
                continue
            if tier.get(srv) == "mcp_servers_1":
                excluded["tool_result not_text_observable (malicious server, no trace in text)"].append(pair)
                continue
            if SCREEN.search(full):
                log["screen_unexplained"].append((pair, srv, tool, SCREEN.search(full).group(0)))
                excluded["tool_result ambiguous (instruction-like text not in spec)"].append(pair)
                continue
            per_task_benign[tid].append(row(pair, "adr", "tool_result", text, "benign", None,
                                            "output of a benign tool", **base))
    for tid, lst in sorted(per_task_benign.items()):
        env = [x for x in lst if tier.get(x["tool"].split(".")[0]) == "mcp_servers_2"]
        rest = sorted((x for x in lst if x not in env), key=lambda x: h8(x["text"]))
        rows.extend(env + rest[:BENIGN_TR_PER_TASK])

    # ---- AgentDojo (bundled run): prompts + tool outputs; injection = the attack template text
    dl = open(os.path.join(adr, "Detection/benchmark/agentdojo_20251022_190150.jsonl"), encoding="utf-8").read().splitlines()
    gt = json.loads(dl[0])["ground_truth"]
    ipi = tech("Indirect Prompt Injection")
    norm = lambda s: re.sub(r"\s+", " ", s)
    for line in dl[1:]:
        r = json.loads(line)
        if not r["conversation"]:
            continue
        key = r["task_id"].replace("_agentdojo_run", "")
        g = gt[key]
        task = f"dojo-{g['suite']}-{g['user_task']}-{g['injection_task']}"
        injections = []
        for m in r["conversation"]:
            if m["message_type"] == "system_init":
                parts = re.split(r"\n\[([A-Za-z_0-9]+)\]: ", "\n" + m["content"])
                injections = [norm(p.strip())[:80] for p in parts[2::2] if p.strip()]
        n = 0
        for m in r["conversation"]:
            if m["message_type"] == "user_prompt":
                k = m["content"].strip()
                if k in seen:
                    continue
                seen.add(k)
                rows.append(row(f"dojo-up-{g['suite']}-{g['user_task']}", "agentdojo", "user_prompt", m["content"],
                                "benign", None, "AgentDojo user task", task=task,
                                group=f"dojo:{g['suite']}:{g['user_task']}", category=f"user_prompt/agentdojo-{g['suite']}"))
            if m["message_type"] != "tool_response":
                continue
            n += 1
            full = m["content"]
            nf = norm(full)
            # the run's own injection texts, or the AgentDojo "important_instructions" template they all use
            hit = any(h and h in nf for h in injections) or (
                "<INFORMATION>" in full and "This is an important message from me" in nf)
            pos = (full.find("<INFORMATION>") if "<INFORMATION>" in full else 0) if hit else None
            text, trunc = clip(full, pos)
            k = text.strip()
            if not k or k in seen:
                continue
            seen.add(k)
            pair = f"dojo-tr-{key.split('_')[1]}-{n:02d}"
            base = dict(task=task, tool=f"agentdojo-{g['suite']}", truncated=trunc, category=f"tool_result/agentdojo-{g['suite']}")
            if hit:
                rows.append(row(pair, "agentdojo", "tool_result", text, "malicious", dict(ipi, vector="instruction_injection"),
                                "contains the injected AgentDojo payload", group=f"dojo:{g['suite']}:{g['injection_task']}", **base))
            elif SCREEN.search(full):
                log["screen_unexplained"].append((pair, "agentdojo", g["suite"], SCREEN.search(full).group(0)))
                excluded["tool_result ambiguous (instruction-like text without the payload)"].append(pair)
            else:
                rows.append(row(pair, "agentdojo", "tool_result", text, "benign", None, "environment data",
                                group=f"dojo:{g['suite']}:{g['user_task']}", **base))

    # ---- MCP tool descriptions (name + docstring + schema as an MCP client lists them)
    poisoned_servers = set()
    listed = {}   # pair -> text: a server/tool registered by several source files is one row
    for rel, fname, tool, doc, schema in extract_tools(adr):
        names = src_of.get(rel) or [fname or os.path.splitext(os.path.basename(rel))[0]]
        tr = rel.split("/")[0]
        for srv in names:
            text = json.dumps({"server": srv, "name": tool, "description": doc, "inputSchema": schema},
                              ensure_ascii=False, indent=1)
            text, trunc = clip(text)
            pair = f"adr-td-{srv}-{tool}"
            if pair in listed:
                if listed[pair] != text:
                    log["tool_description_conflict"].append((pair, rel))
                continue
            listed[pair] = text
            e = match.desc.get((srv, tool))
            base = dict(task=None, group=("srv:" + srv) if tr == "mcp_servers_1" else f"tdsrv:{srv}", tool=f"{srv}.{tool}",
                        truncated=trunc, category=f"tool_description/{tr}")
            if e and e["label"] == "malicious":
                rows.append(row(pair, "adr", "tool_description", text, "malicious",
                                dict(technique_for(srv), vector=e["vector"]), "spec: " + e["notes"], **base))
                poisoned_servers.add(srv)
            elif e:
                excluded[f"tool_description {e['label']}"].append(pair)
            elif tr == "mcp_servers_1":
                excluded["tool_description not_text_observable (malicious server, benign listed text)"].append(pair)
            else:
                rows.append(row(pair, "adr", "tool_description", text, "benign", None, "benign MCP server tool", **base))
    for tid, t in tasks.items():
        if t["ground_truth"] == "malicious" and any(s in poisoned_servers for s in t["mcp_servers"]):
            observable[tid].append("tool_description")

    split_rows(rows)
    mal_tasks = sorted(t for t in tasks if tasks[t]["ground_truth"] == "malicious")
    stats = {
        "adr_malicious_tasks": len(mal_tasks),
        "adr_text_observable_tasks": sorted(observable),
        "adr_not_text_observable_tasks": [t for t in mal_tasks if t not in observable],
        "excluded": {k: len(v) for k, v in sorted(excluded.items())},
    }
    return rows, log, stats


def split_rows(rows):
    """Groups ordered by sha1(group) alternate dev/test inside each origin+kind (no group on both sides)."""
    groups = defaultdict(set)
    for r in rows:
        groups[(r["origin"], r["kind"], r["label"])].add(r["group"])
    side = {}
    for key in sorted(groups):
        for g in sorted(groups[key] - set(side), key=lambda g: hashlib.sha1(g.encode()).hexdigest()):
            side[g] = "dev" if len([1 for x in groups[key] if side.get(x) == "dev"]) <= len([1 for x in groups[key] if side.get(x) == "test"]) else "test"
    for r in rows:
        r["split"] = side[r["group"]]



def check_clone(adr, allow_other_commit=False):
    """Verify the clone is at ADR_COMMIT and the source files match the pinned sha256."""
    head = head_commit(adr)
    if head != ADR_COMMIT and not allow_other_commit:
        raise ValueError("ADR clone is at %s, expected %s" % (head, ADR_COMMIT))
    for rel, want in SRC_SHA256.items():
        got = sha256_file(os.path.join(adr, rel))
        if got != want and not allow_other_commit:
            raise ValueError("%s: sha256 %s != pinned %s" % (rel, got, want))
    return head
