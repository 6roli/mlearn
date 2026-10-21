"""Ejecuta las tareas del enunciado con tu entorno local y guarda texto y figuras en executed/tasks_out.json.

Uso:  "<LAB>/.venv/Scripts/python.exe" scripts/run_tasks.py
(antes: node scripts/export-tasks.ts; después: npm run import)
"""
import base64
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "executed"
RUN = EX / "run"
MANIFEST = json.loads((ROOT / "content" / "materials.json").read_text(encoding="utf-8"))
FIGS = []


def capture():
    for num in plt.get_fignums():
        buf = io.BytesIO()
        plt.figure(num).savefig(buf, format="png", dpi=90, bbox_inches="tight")
        FIGS.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    plt.close("all")


plt.show = lambda *a, **k: capture()


def main():
    src_root = Path(MANIFEST["defaultRoot"])
    RUN.mkdir(parents=True, exist_ok=True)
    for ds in MANIFEST["datasets"]:
        if not (RUN / ds["target"]).exists():
            shutil.copyfile(src_root / ds["path"], RUN / ds["target"])
    os.chdir(RUN)
    tasks = json.loads((EX / "tasks.json").read_text(encoding="utf-8"))
    import numpy, pandas, scipy, sklearn
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"),
           "versions": {m.__name__: m.__version__ for m in (numpy, pandas, scipy, sklearn, matplotlib)}, "tasks": {}}
    for t in tasks:
        FIGS.clear()
        buf = io.StringIO()
        err = None
        t0 = time.time()
        import warnings
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                exec(compile(t["code"], f"<{t['id']}>", "exec"), {"__name__": "__main__"})
            except Exception:
                err = traceback.format_exc()[-2500:]
        capture()
        out["tasks"][t["id"]] = {"stdout": buf.getvalue()[:60000], "figures": list(FIGS), "error": err,
                                 "seconds": round(time.time() - t0, 1),
                                 "codeHash": hashlib.sha256(t["code"].encode()).hexdigest()[:16]}
        print(f"{t['id']}: {round(time.time() - t0, 1)} s, {len(FIGS)} figuras" + (" ERROR" if err else ""))
        if err:
            print(err)
    (EX / "tasks_out.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
