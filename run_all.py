"""Runs the whole pipeline in the correct order. Usage: python run_all.py
Steps: dataset -> baseline models -> advanced experiments -> architecture figure -> web page -> backend tests."""
import os, subprocess, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
for d in ("data", "outputs", "figures"): os.makedirs(d, exist_ok=True)
for step in ["generate_dataset.py", "train_model.py", "experiments2.py", "make_diagram.py", "build_web.py", "test_app.py"]:
    print(f"\n=== {step} ===", flush=True)
    if subprocess.run([sys.executable, step]).returncode != 0: sys.exit(f"Step failed: {step}")
print("\nAll steps finished. Open web/index.html in a browser, or run:  python app.py  (then open http://127.0.0.1:5000)")
