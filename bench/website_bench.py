#!/usr/bin/env python3
# Benchmarks for the website: per model, prompt processing and generation from short prompts to 131K tokens, the
# generation modes (plain, MTP, DFlash2), and the KL against an f32 CPU reference. Writes one JSON object per
# measurement to <out>/results.jsonl and the tables to <out>/summary.md.
# The Python port of bench/website_bench.ps1, for Windows and Linux: the same runs, the same records and the same
# output. The two must stay the same; change both together.
#   Windows: python bench\website_bench.py [--models a.gguf b.gguf] [--sizes 128 512 2048 8192 32768 131072]
#   Linux:   python3 bench/website_bench.py [--models a.gguf b.gguf] [--sizes 128 512 2048 8192 32768 131072]
#   [--out bench/results/website] [--skip-kl] [--skip-modes] [--gen 64] [--cooldown 20]   (--sizes 128,512 works too)
# A. Prompt sizes: runz --bench-pp N --n GEN. Prompt tok/s (best of 2 runs up to 8K, one run above), time to the first
#    token (N / prompt tok/s), and generation tok/s at that depth (GEN tokens after the N-token prompt), plain decoding.
# B. Generation modes, short context: a 256-token coding answer, plain / --mtp / --dflash (when the model has an MTP
#    block or a DFlash2 drafter is in the Hugging Face cache), best of 2.
# C. KL: 480 tokens of USAGE.md through bin/ref_forward (f32 activations, dequantized weights; kept in
#    <out>/ref so later runs reuse it) and runz --kl-ref: mean / max KL and how often the top token agrees.
# Without --models: the usual models found in the Hugging Face cache, the data folder's zero/models
# (%USERPROFILE%\.inferzero on Windows, ~/.inferzero on Linux, or ARC_DATA_DIR) and the folders the server reads.
# Every run checks for other GPU programs first (a second program on the GPU spoils the numbers).
# Programs: bin/runz, bin/ref_forward, bin/serverz, bin/mgpu_probe (.exe on Windows). A relative --out is taken from
# the inferzero folder, as in the .ps1.
# Python 3.8 or newer, standard library only.
import argparse
import csv
import datetime
import decimal
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = ".exe" if os.name == "nt" else ""
RUNZ = os.path.join(ROOT, "bin", "runz" + EXE)
REF = os.path.join(ROOT, "bin", "ref_forward" + EXE)
SERVERZ = os.path.join(ROOT, "bin", "serverz" + EXE)
PROBE = os.path.join(ROOT, "bin", "mgpu_probe" + EXE)
VARS = ["ARC_BENCH_RUNS", "REF_ALL"]  # (the variables the script sets for single runs)
GPU_PROGRAMS = ["serverz", "runz", "zero", "zdec", "ref_forward"]


def int_list(values):
    # (--sizes 128 512 and --sizes 128,512 both work, as -Sizes 128,512 does in the .ps1)
    out = []
    for v in values:
        out += [int(x) for x in v.split(",") if x.strip() != ""]
    return out


ap = argparse.ArgumentParser(
    description="Benchmarks for the website: prompt sizes, generation modes and KL against an f32 reference "
                "(the Python port of bench/website_bench.ps1). Writes <out>/results.jsonl and <out>/summary.md.")
ap.add_argument("--models", nargs="+", default=[], help="the models (.gguf), space-separated (default: the usual "
                "models found in the Hugging Face cache, the data folder and the folders the server reads)")
ap.add_argument("--sizes", nargs="+", default=["128", "512", "2048", "8192", "32768", "131072"],
                help="prompt sizes in tokens (default 128 512 2048 8192 32768 131072; 128,512 works too)")
ap.add_argument("--out", default=os.path.join("bench", "results", "website"),
                help="the results folder (default bench/results/website, from the inferzero folder)")
ap.add_argument("--gen", type=int, default=64, help="tokens generated after each prompt size (default 64)")
ap.add_argument("--cooldown", type=int, default=20, help="seconds of rest before every run (default 20)")
ap.add_argument("--skip-kl", action="store_true", help="no KL against the f32 reference (phase C)")
ap.add_argument("--skip-modes", action="store_true", help="no generation modes (phase B)")
args = ap.parse_args()
SIZES = int_list(args.sizes)
GEN = args.gen
COOLDOWN = args.cooldown
try:
    sys.stdout.reconfigure(errors="replace")
except AttributeError:
    pass

if not os.path.exists(RUNZ):
    sys.exit("bin/runz%s not found" % EXE)
# (model paths as given, from the current folder; --out from the inferzero folder, as the .ps1 sets its location)
models = [os.path.abspath(os.path.expanduser(m)) for m in args.models]
OUT_ARG = args.out
OUT = OUT_ARG if os.path.isabs(OUT_ARG) else os.path.join(ROOT, OUT_ARG)
os.makedirs(os.path.join(OUT, "ref"), exist_ok=True)
JSONL = os.path.join(OUT, "results.jsonl")


def out(s=""):
    print(s, flush=True)


def leaf(path):
    return os.path.basename(path.rstrip("/\\"))


def lines_of(o):
    # (the .ps1 splits runz's output at "\n"; the "\r" left of Windows line ends is dropped here)
    return [l.rstrip("\r") for l in o.split("\n")]


def same_name(a, b):
    # (PowerShell compares names in any case; Windows paths are, Linux paths are not)
    return os.path.normcase(a) == os.path.normcase(b)


def hub_dir():  # the Hugging Face cache, as the engine (src/platform.h hf_home_hub) and scripts/zero_launcher.py find it
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    if os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "hub")
    x = os.environ.get("XDG_CACHE_HOME", "")
    if os.name != "nt" and x.startswith("/"):
        return os.path.join(x, "huggingface", "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def app_dir():
    # the data folder as the engine finds it (src/app_dir.h, scripts/app-dir.ps1): ARC_DATA_DIR, else .inferzero in the
    # user's home folder (%USERPROFILE%, $HOME on Linux); when only an older folder exists, that one (the engine and
    # app-dir.ps1 move it to .inferzero once; this script only reads it and leaves the move to them)
    if os.environ.get("ARC_DATA_DIR"):
        return os.environ["ARC_DATA_DIR"]
    home = os.environ.get("USERPROFILE" if os.name == "nt" else "HOME") or "."
    now = os.path.join(home, ".inferzero")
    if os.path.exists(now):
        return now
    olds = [os.path.join(home, ".arcinferzero"), os.path.join(home, "arcinferzero")]
    if os.environ.get("LOCALAPPDATA"):
        olds += [os.path.join(os.environ["LOCALAPPDATA"], n) for n in ("arcinferzero", "arcinfer")]
    for o in olds:
        if os.path.isdir(o):
            return o
    return now


HUB = hub_dir()
PKG = os.path.join(app_dir(), "zero", "models")


# ---- numbers as .NET formats them (the .ps1's [string]::Format with the invariant culture), so the lines and tables
#      read the same
def fmt_f(x, digits, width=0):
    # {i,width:Fd}: d decimals, midpoints away from zero; nothing (padded) for no value
    if x is None:
        s = ""
    else:
        q = decimal.Decimal(1).scaleb(-digits)
        s = str(decimal.Decimal(repr(float(x))).quantize(q, rounding=decimal.ROUND_HALF_UP))
        if s.startswith("-") and decimal.Decimal(s) == 0:
            s = s[1:]
    return s.rjust(width) if width >= 0 else s.ljust(-width)


def fmt_g(x):
    # a number as "{0}" or "$x" writes it: 80 not 80.0, 1E-05 not 1e-05; nothing for no value
    if x is None:
        return ""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return repr(float(x)).replace("e", "E")


def num(o, rx):
    # the first group of rx in o as a number, or None ([regex]::Match: case matters)
    m = re.search(rx, o)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# ---- the models
# (also every folder the server reads: model-dirs.txt, e.g. C:\hf, through serverz --list-models; GitHub #4: models
#  outside the Hugging Face cache were not found and the script stopped at its summary)
listed = []
if os.path.exists(SERVERZ):
    try:
        p = subprocess.run([SERVERZ, "--list-models"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        data = json.loads(p.stdout.decode("utf-8", "replace"))
        for it in (data if isinstance(data, list) else [data]):
            if isinstance(it, dict) and it.get("supported") and it.get("path"):
                listed.append(it)
    except (OSError, ValueError):
        listed = []


def files_by_name(top):
    # every file under top by name (in normcase), the first one found kept: Get-ChildItem -Recurse -Filter's first match
    found = {}
    for d, dirs, files in os.walk(top):
        dirs.sort()
        for f in sorted(files):
            found.setdefault(os.path.normcase(f), os.path.join(d, f))
    return found


if not models:
    names = ["Qwen3.5-4B-UD-Q4_K_XL.gguf", "Qwen3.5-9B-UD-Q4_K_XL.gguf", "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
             "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S.gguf",
             "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf", "gemma-4-E4B_q4_0-it.gguf",
             "gemma-4-12b-it-qat-q4_0.gguf", "gemma-4-26b-a4b-it-qat-q4_0.gguf", "gemma-4-26B_q4_0-it.gguf",
             "gemma-4-31B_q4_0-it.gguf",
             # (DeepSeek-V4.1-Flash, ISTA's IQ2_XS: the first of six parts; runz finds the others next to it)
             "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00006.gguf"]
    in_pkg = files_by_name(PKG)
    in_hub = files_by_name(HUB)
    for n in names:
        key = os.path.normcase(n)
        f = in_pkg.get(key) or in_hub.get(key)
        if not f:
            f = next((l["path"] for l in listed if same_name(leaf(l["path"]), n)), None)
        # (one Flash Next file only: the single file or the first of its two parts)
        if f and not any("flash-next" in x.lower() and "flash-next" in f.lower() for x in models):
            models.append(f)
if not models:
    out("No model found: none of the usual model files is in the Hugging Face cache, %s" % PKG)
    out("or a folder the server reads (model-dirs.txt). Name them: python bench/website_bench.py --models /path/model.gguf ...")
    sys.exit(1)

# DFlash2 drafters in the Hugging Face cache, by model
DRAFTERS = [("27B", "models--z-lab--Qwen3.8-27B-DFlash2"), ("35B-A3B", "models--incoai--Qwen3.6-35B-A3B-DFlash2"),
            ("Flash-Next", "models--incoai--Qwen3.8-Flash-Next-DFlash2")]


def drafter(model):
    for k, repo in DRAFTERS:
        if k.lower() in model.lower():
            sd = os.path.join(HUB, repo, "snapshots")
            try:
                subs = sorted(c for c in os.listdir(sd) if os.path.isdir(os.path.join(sd, c)))
            except OSError:
                subs = []
            if subs and os.path.exists(os.path.join(sd, subs[0], "config.json")):
                return os.path.join(sd, subs[0])
    # (else the drafter the server found for that model, in any folder it reads)
    l = next((l for l in listed if same_name(l["path"], model) and l.get("dflash_path")), None)
    if l and os.path.exists(os.path.join(l["dflash_path"], "config.json")):
        return l["dflash_path"]
    return None


def gpu_programs_running():
    # another GPU program running: on Windows by tasklist, on Linux by /proc/<pid>/comm
    want = set(GPU_PROGRAMS)
    if os.name == "nt":
        try:
            p = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except OSError:
            return False
        for row in csv.reader(io.StringIO(p.stdout.decode("utf-8", "replace"))):
            if row:
                n = row[0].lower()
                if n.endswith(".exe"):
                    n = n[:-4]
                if n in want:
                    return True
        return False
    try:
        pids = [d for d in os.listdir("/proc") if d.isdigit()]
    except OSError:
        return False
    for pid in pids:
        try:
            with open("/proc/%s/comm" % pid, encoding="utf-8", errors="replace") as f:
                if f.read().strip().lower() in want:
                    return True
        except OSError:
            pass
    return False


def wait_free():
    while gpu_programs_running():
        time.sleep(30)
    if COOLDOWN > 0:
        time.sleep(COOLDOWN)


def env_with(envs):
    # a copy of this process's environment without the script's variables, plus envs
    e = dict(os.environ)
    for v in VARS:
        e.pop(v, None)
    e.update(envs)
    return e


def run(a, envs=None):
    # one runz call once the GPU is free; returns its output (stdout and stderr)
    wait_free()
    p = subprocess.run([RUNZ] + [str(x) for x in a], env=env_with(envs or {}), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    return p.stdout.decode("utf-8", "replace")


def save(obj):
    with open(JSONL, "a", encoding="utf-8", newline="") as f:
        f.write(json.dumps(obj, separators=(",", ":")) + "\n")


q = os.path.join(tempfile.gettempdir(), "inferzero-website-q.txt")
text = ("<|im_start|>user\nWrite a Python module with a class LRUCache (get, put, a capacity, O(1) operations using "
        "a dict and a doubly linked list), docstrings, and unit tests with unittest.<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n")
with open(q, "w", encoding="utf-8", newline="") as f:
    f.write(text)
kltext = os.path.join(tempfile.gettempdir(), "inferzero-website-kl.txt")
with open(os.path.join(ROOT, "USAGE.md"), encoding="utf-8", newline="") as f:
    usage = f.read()
with open(kltext, "w", encoding="utf-8", newline="") as f:
    f.write(usage[:2600])

gpu = "unknown GPU"
if os.path.exists(PROBE):
    p = subprocess.run([PROBE], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for l in p.stdout.decode("utf-8", "replace").splitlines():
        if re.match(r"^\s+GPU 0:", l, re.I):
            gpu = l.strip()
            break
# (the commit: a package's bin/PREBUILT.txt names it; a git folder asks git)
commit = ""
pre = os.path.join(ROOT, "bin", "PREBUILT.txt")
if os.path.exists(pre):
    with open(pre, encoding="utf-8", errors="replace") as f:
        mc = re.search(r"commit ([0-9a-f]+)", f.read())
    if mc:
        commit = mc.group(1)
if not commit and os.path.exists(os.path.join(ROOT, ".git")) and shutil.which("git"):
    try:
        p = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL)
        ls = p.stdout.decode("utf-8", "replace").splitlines()
        commit = ls[0].strip() if ls else ""
    except OSError:
        commit = ""
if not commit:
    commit = "unknown"
stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
with open(os.path.join(OUT, "log.txt"), "a", encoding="utf-8", newline="") as f:
    f.write("# website benchmark %s, commit %s, %s\n" % (stamp, commit, gpu))

for m in models:
    if not os.path.exists(m):
        out("skipped (not found): " + m)
        continue
    name = os.path.splitext(leaf(m))[0]
    out("=== " + name)
    # (a run that is not counted first: a model's first run compiles its kernels (and builds or warms its expert file),
    #  which landed in the first size's generation: a B60's 4B generated 12.8 tok/s at 128 tokens, 61 at 512, issue #4)
    run([m, "--bench-pp", "128", "--n", "8", "--must-fit"], {"ARC_BENCH_RUNS": "1"})
    # ---- A. prompt sizes
    for n in SIZES:
        runs = "2" if n <= 8192 else "1"
        # (--must-fit: a context beyond the GPU share stops at load, so the size is marked "does not fit" instead of
        #  hours of paging: Qwen3.8 27B at 131K on a 16.5 GiB share ran at 8.6 instead of ~120 tok/s)
        o = run([m, "--bench-pp", n, "--n", GEN, "--must-fit"], {"ARC_BENCH_RUNS": runs})
        nf = re.search(r"does not fit: ([\d.]+) GiB", o, re.I)
        if nf:
            save({"time": stamp, "commit": commit, "gpu": gpu, "model": name, "test": "size", "tokens": n, "pp": None,
                  "tg": None, "ttft_s": None, "fits": False, "need_gib": float(nf.group(1)),
                  "error": "does not fit in the GPU share"})
            out("  {0:>7} tokens: does not fit in the GPU share ({1} GiB needed)".format(n, nf.group(1)))
            continue
        pps = [float(x.group(1)) for x in re.finditer(r"pp\d+ run \d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s", o)]
        # (as in the .ps1: no pass, or one pass at 0, gives no prompt speed)
        pp = max(pps) if pps and pps != [0.0] else None
        tg = num(o, r"tg\d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s")
        err = None
        if not pp:
            err = next((l for l in lines_of(o) if re.search(r"error|cannot|failed|exception|out of", l, re.I)), None)
        save({"time": stamp, "commit": commit, "gpu": gpu, "model": name, "test": "size", "tokens": n, "pp": pp,
              "tg": tg, "ttft_s": round(n / pp, 2) if pp else None, "error": err})
        out("  {0:>7} tokens: prompt {1} tok/s, first token {2} s, generation {3} tok/s {4}".format(
            n, fmt_f(pp, 1, 8), fmt_f(n / pp if pp else 0, 2, 7), fmt_f(tg, 2, 6), err or ""))
    # ---- B. generation modes
    if not args.skip_modes:
        d = drafter(m)
        modes = [("plain", []), ("mtp", ["--mtp"])]
        if d:
            modes += [("dflash", ["--dflash", d, "--draft", "dflash"]), ("mtp+dflash", ["--dflash", d, "--draft", "auto"])]
        for k, extra in modes:
            best = None
            acc = None
            for _ in range(2):
                o = run([m, "--text-file", q, "--n", "256"] + extra)
                t = num(o, r"generation: \d+ tokens in [\d.]+ ms \(([\d.]+) t/s\)")
                if t and (best is None or t > best):
                    best = t
                    acc = num(o, r"\(([\d.]+) %\), [\d.]+ tokens per main-model pass")
            if k == "mtp" and best is None:
                continue  # (no MTP block)
            save({"time": stamp, "commit": commit, "gpu": gpu, "model": name, "test": "mode", "mode": k, "tg": best,
                  "accepted_pct": acc})
            out("  {0:<11} {1} tok/s {2}".format(k, fmt_f(best, 2, 6),
                                                 "(%s %% of drafts accepted)" % fmt_g(acc) if acc else ""))
    # ---- C. KL against f32
    if not args.skip_kl and os.path.exists(REF):
        ids = os.path.join(OUT, "ref", name + ".ids.txt")
        rb = os.path.join(OUT, "ref", name + ".f32.bin")
        if not os.path.exists(rb):
            run([m, "--text-file", kltext, "--dump-ids", ids])
            if os.path.exists(ids):
                with open(ids, encoding="utf-8-sig", newline="") as f:
                    t = f.read().strip().split(",")
                with open(ids, "w", encoding="utf-8", newline="") as f:
                    f.write(",".join(t[:480]))
                wait_free()
                subprocess.run([REF, m, "@" + ids, rb], env=env_with({"REF_ALL": "1"}), stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
        if os.path.exists(rb):
            o = run([m, "--tokens-file", ids, "--kl-ref", rb])
            kl = num(o, r"KL\(ref \|\| inferzero\) = ([\d.]+)")
            kmax = num(o, r"max ([\d.]+)\)")
            agree = re.search(r"top token agrees (\d+) / (\d+)", o)
            top = round(100.0 * int(agree.group(1)) / int(agree.group(2)), 1) if agree and int(agree.group(2)) else None
            save({"time": stamp, "commit": commit, "gpu": gpu, "model": name, "test": "kl", "kl": kl, "kl_max": kmax,
                  "top_agree_pct": top})
            out("  KL against f32: mean {0}, max {1}, top token agrees {2} %".format(fmt_g(kl), fmt_g(kmax), fmt_g(top)))
        else:
            out("  KL: no f32 reference (ref_forward failed or does not take this model)")
for f in (q, kltext):
    try:
        os.remove(f)
    except OSError:
        pass

# ---- summary.md from every row of this run
rows = []
with open(JSONL, encoding="utf-8-sig") as f:
    for l in f:
        l = l.strip().lstrip("\ufeff")
        if not l:
            continue
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if isinstance(r, dict) and r.get("time") == stamp:
            rows.append(r)
mnames = []
for r in rows:
    if r.get("model") not in mnames:
        mnames.append(r.get("model"))


def first_row(mn, test, key, value):
    return next((r for r in rows if r.get("model") == mn and r.get("test") == test and r.get(key) == value), None)


head = "| Model | " + " | ".join(str(n) for n in SIZES) + " |"
rule = "|---|" + "|".join("---" for _ in SIZES) + "|"
md = ["# inferzero benchmarks", "", "%s; commit %s; %s." % (gpu, commit, stamp), ""]
md += ["## Prompt processing (tok/s) and time to first token", "", head, rule]
for mn in mnames:
    cells = []
    for n in SIZES:
        r = first_row(mn, "size", "tokens", n)
        if r and r.get("pp"):
            cells.append("%s (%s s)" % (fmt_f(r["pp"], 0), fmt_f(r.get("ttft_s"), 1)))
        elif r and r.get("fits") is False:
            cells.append("does not fit")
        else:
            cells.append("-")
    md.append("| %s | " % mn + " | ".join(cells) + " |")
md += ["", "## Generation (tok/s) at that context depth, plain decoding", "", head, rule]
for mn in mnames:
    cells = []
    for n in SIZES:
        r = first_row(mn, "size", "tokens", n)
        if r and r.get("tg"):
            cells.append(fmt_f(r["tg"], 1))
        elif r and r.get("fits") is False:
            cells.append("does not fit")
        else:
            cells.append("-")
    md.append("| %s | " % mn + " | ".join(cells) + " |")
md += ["", "## Generation modes (256-token coding answer, tok/s)", "", "| Model | plain | MTP | DFlash2 | MTP + DFlash2 |",
       "|---|---|---|---|---|"]
for mn in mnames:
    cells = []
    for k in ("plain", "mtp", "dflash", "mtp+dflash"):
        r = first_row(mn, "mode", "mode", k)
        if r and r.get("tg"):
            a = r.get("accepted_pct")
            cells.append(fmt_f(r["tg"], 1) + (" (%s %%)" % fmt_g(a) if a else ""))
        else:
            cells.append("-")
    md.append("| %s | " % mn + " | ".join(cells) + " |")
md += ["", "## Accuracy: KL against an f32 reference (480 tokens of USAGE.md)", "",
       "| Model | KL mean | KL max | top token agrees |", "|---|---|---|---|"]
for r in rows:
    if r.get("test") == "kl":
        md.append("| %s | %s | %s | %s %% |" % (r.get("model"), fmt_f(r.get("kl"), 5), fmt_f(r.get("kl_max"), 4),
                                               fmt_f(r.get("top_agree_pct"), 1)))
with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8", newline="") as f:
    f.write("".join(l + os.linesep for l in md))  # (WriteAllLines: the system's line end after every line)
out("summary: " + os.path.join(OUT_ARG, "summary.md"))
sys.exit(0)  # (not runz's last exit code: 3 when the last size did not fit)
