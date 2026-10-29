import re, os, subprocess, glob, sys
ROOT = "/Users/Suavi/Documents_SY/02_Education/University/Master/Masters Thesis/01_Thesis_Latex/Git/6abd2e69062d299b224d6f5b"
os.chdir(ROOT)
FIG = "figures"
chap_dir = {
    "Cover.tex": "00_cover",
    "2_0_Theoretical_Foundations.tex": "02_theory",
    "3_0_State_of_the_Art.tex": "03_state_of_the_art",
    "5_0_Methodology.tex": "05_methodology",
    "6_0_Implementation.tex": "06_implementation",
    "7_0_Results_and_Evaluation.tex": "07_results",
    "A_Appendix.tex": "A_appendix",
}
# 1. collect references (non-comment lines) in the live chapter sources
refs = {}   # basename -> (folder, tex file, line)
for tex, folder in chap_dir.items():
    if tex == "Cover.tex":
        continue
    for i, line in enumerate(open(tex, encoding="utf-8", errors="replace"), 1):
        if line.lstrip().startswith("%"):
            continue
        for m in re.finditer(r"includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}", line):
            name = m.group(1)
            if name in refs and refs[name][0] != folder:
                sys.exit(f"conflict: {name} used in two chapters")
            refs.setdefault(name, (folder, tex, i))
# cover image(s): titelbild is set via \TITELBILD in FAPS-Thesis.tex
cover_names = [f for f in os.listdir(FIG) if f.startswith("titelbild")]
for f in cover_names:
    refs.setdefault(f, ("00_cover", "FAPS-Thesis.tex", 51))
# 2. every referenced file must exist
missing = [n for n in refs if not os.path.exists(os.path.join(FIG, n))]
print("missing:", missing)
# 3. move files
def git_tracked(p):
    return subprocess.run(["git", "ls-files", "--error-unmatch", p],
                          capture_output=True).returncode == 0
moved = []
for f in sorted(os.listdir(FIG)):
    src = os.path.join(FIG, f)
    if os.path.isdir(src) or f.startswith("."):
        continue
    folder = refs[f][0] if f in refs else "_not_used_in_thesis"
    dst_dir = os.path.join(FIG, folder)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f)
    if git_tracked(src):
        subprocess.run(["git", "mv", src, dst], check=True)
    else:
        os.rename(src, dst)
    moved.append((f, folder))
# 4. rewrite paths in the chapter sources
for tex in chap_dir:
    if not os.path.exists(tex):
        continue
    text = open(tex, encoding="utf-8").read()
    def repl(m):
        name = m.group(1)
        if name in refs:
            return "figures/" + refs[name][0] + "/" + name
        return m.group(0)
    new = re.sub(r"(?<=includegraphics)(?:\[[^\]]*\])?\{figures/([^}]+)\}",
                 lambda m: m.group(0).replace("figures/" + m.group(1), "figures/" + refs[m.group(1)][0] + "/" + m.group(1))
                 if m.group(1) in refs else m.group(0), text)
    if new != text:
        open(tex, "w", encoding="utf-8").write(new)
        print("updated", tex)
# 5. graphicspath for cover image
fp = "FAPS-Thesis.tex"
t = open(fp, encoding="utf-8").read()
t2 = t.replace("{./figures/}{./figure/}", "{./figures/00_cover/}")
if t2 != t:
    open(fp, "w", encoding="utf-8").write(t2); print("updated graphicspath")
print("moved", len(moved), "files")
for f, folder in moved:
    print(f"{folder}/{f}")
