"""
Run git operations and capture output to git_log.txt
"""
import subprocess, os, shutil, sys

REPO = "C:\\Users\\dypag\\Claude\\Projects\\TRADING"
REMOTE = "https://github.com/dypage15/machine-learning-quant.git"
OUT = os.path.join(REPO, "git_log.txt")

def run(cmd, **kw):
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", **kw)
    out = (r.stdout + r.stderr).strip()
    print(f"$ {' '.join(cmd)}\n{out}\n{'='*50}")
    return r.returncode, out

lines = []

# 1. Remove broken .git
git_dir = os.path.join(REPO, ".git")
if os.path.isdir(git_dir):
    # Check if it's a valid repo
    rc, _ = run(["git", "-C", REPO, "rev-parse", "--git-dir"])
    if rc != 0:
        lines.append("Removing broken .git directory...")
        shutil.rmtree(git_dir, ignore_errors=True)
        lines.append("Removed .git")
    else:
        lines.append("Existing valid .git found")

# 2. Init
rc, o = run(["git", "-C", REPO, "init", "-b", "main"])
lines.append(o)

# 3. Configure
run(["git", "-C", REPO, "config", "user.email", "thepage8171@gmail.com"])
run(["git", "-C", REPO, "config", "user.name", "Dylan"])

# 4. Remote
rc, _ = run(["git", "-C", REPO, "remote", "get-url", "origin"])
if rc != 0:
    run(["git", "-C", REPO, "remote", "add", "origin", REMOTE])
else:
    run(["git", "-C", REPO, "remote", "set-url", "origin", REMOTE])

# 5. Add
rc, o = run(["git", "-C", REPO, "add", "-A"])
lines.append(f"add: {o}")

# 6. Commit
rc, o = run(["git", "-C", REPO, "commit", "-m",
             "PCA MNQ ML system: pipeline, Lorentzian+LSTM models, bars import"])
lines.append(f"commit: {o}")

# 7. Push
rc, o = run(["git", "-C", REPO, "push", "-u", "origin", "main"])
lines.append(f"push exit={rc}: {o}")

# 8. Status
rc, o = run(["git", "-C", REPO, "log", "--oneline", "-5"])
lines.append(f"log: {o}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n\nSummary written to git_log.txt")
input("Press Enter to close...")
