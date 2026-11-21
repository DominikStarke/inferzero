# docs/options.md from scripts/options.tsv (the launchers' catalog of opt-in switches, QUEUE O0 / O6). A tool for
# whoever edits the catalog (users never need it: the launchers read the .tsv themselves).
#   python tools/gen_options_md.py
import csv, os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(root, "scripts", "options.tsv"), encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
md = ["# Options you can turn on", "",
      "The switches below are off unless the table says otherwise. Each changes speed or memory; the ones marked "
      "\"same output\" give the same text bit for bit, the others are checked against an f32 reference (KL). The "
      "launchers read this list from `scripts/options.tsv`; this page is made from that file "
      "(`python tools/gen_options_md.py`).", "",
      "Set an environment variable before starting the server (PowerShell: `$env:NAME = \"value\"`, bash: "
      "`export NAME=value`), or add the flag to serverz's command line.", ""]
for d in rows:
    how = d["name"] + ("=" + d["on"] if d["kind"] == "env" else (" " + d["on"] if d["on"] else ""))
    link = d["docs"][5:] if d["docs"].startswith("docs/") else "../" + d["docs"]
    md += ["## " + d["title"] + " (`" + how + "`)", "",
           "- **What it does:** " + d["what"] + ".",
           "- **Default:** " + d["default"] + ".",
           "- **Best for:** " + d["best_for"].replace(";", ", ") + ".",
           "- **Cost:** " + d["cost"] + ".",
           "- **Output:** " + ("the same, bit for bit" if d["output"] == "same" else "slightly different, checked against f32 (KL)") + ".",
           "- **Measured:** " + d["measured"] + "."]
    if d["conflicts"]:
        md.append("- **Note:** " + d["conflicts"] + ".")
    md += ["- **More:** [" + d["docs"] + "](" + link + "), " + d["status"] + ".", ""]
with open(os.path.join(root, "docs", "options.md"), "w", encoding="utf-8", newline="") as f:
    f.write("\n".join(md))
print("docs/options.md:", len(rows), "options")
