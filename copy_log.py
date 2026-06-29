import os, shutil, glob
log_dir = os.path.expanduser("~\\.pca_mnq_ml\\logs")
# Check all files in logs dir
all_files = sorted(glob.glob(os.path.join(log_dir, "nightly_*")), key=os.path.getmtime)
print(f"Log dir: {log_dir}")
print(f"Files: {all_files}")
if all_files:
    latest = all_files[-1]
    dest = os.path.join(os.path.dirname(__file__), "last_run.txt")
    shutil.copy2(latest, dest)
    print(f"\nCopied: {latest}")
    with open(latest, encoding="utf-8", errors="replace") as f:
        print(f.read())
else:
    print("No nightly log files found")

pred_file = os.path.expanduser("~\\.pca_mnq_ml\\next_session_predictions.json")
if os.path.exists(pred_file):
    dest2 = os.path.join(os.path.dirname(__file__), "last_predictions.json")
    shutil.copy2(pred_file, dest2)
    print(f"\nPredictions copied")
    print(open(pred_file, encoding="utf-8").read())
else:
    print(f"\nNo predictions file at {pred_file}")

input("Press Enter...")
