"""
Commit latest changes and push to GitHub.
"""
import subprocess, os

REPO = "C:\\Users\\dypag\\Claude\\Projects\\TRADING"
REMOTE = "https://github.com/dypage15/machine-learning-quant.git"
OUT = os.path.join(REPO, "git_log.txt")

def run(cmd):
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (r.stdout + r.stderr).strip()
    print(f"$ {' '.join(cmd)}\n{out}\n{'='*50}")
    return r.returncode, out

lines = []

run(["git", "-C", REPO, "remote", "set-url", "origin", REMOTE])

rc, o = run(["git", "-C", REPO, "add", "-A"])
lines.append(f"add: {o}")

rc, o = run(["git", "-C", REPO, "commit", "-m",
             "Feat: 22-feature model (pca_cs, RTY, GC, grade_score); cyberpunk dashboard; MFE/MAE"])
lines.append(f"commit: {o}")

rc, o = run(["git", "-C", REPO, "push", "origin", "main"])
lines.append(f"push exit={rc}: {o}")

rc, o = run(["git", "-C", REPO, "log", "--oneline", "-3"])
lines.append(f"log:\n{o}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\nDone — git_log.txt written")
input("Press Enter to close...")
