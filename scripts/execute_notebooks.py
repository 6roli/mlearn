"""Ejecuta COPIAS de todos los notebooks con tu entorno local y guarda las salidas.

Uso (con el Python de tu entorno del laboratorio, que tiene Jupyter):
    "<LAB>/.venv/Scripts/python.exe" scripts/execute_notebooks.py ["C:/ruta/a/LAB"]

- Los originales NO se modifican: se copian a executed/run/, se ejecutan y el resultado se guarda en
  executed/<id>.ipynb (carpeta excluida de git).
- Solo se aplican los arreglos mínimos para que la ejecución no se detenga (lista PATCHES); cada uno queda
  anotado en executed/informe.json y la app lo muestra.
- Después, `npm run import` incorpora estas salidas a la app (y a los PDF).
"""
import json
import shutil
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "content" / "materials.json").read_text(encoding="utf-8"))
OUT = ROOT / "executed"
RUN = OUT / "run"

# (notebook, celda, texto original, reemplazo, motivo)
PATCHES = [
    ("lab0_2", 7, None, 'iris = pd.read_csv("iris.csv")',
     "Ruta absoluta antigua (Lab0/Lab0/Lab0) que ya no existe: se lee iris.csv de la carpeta de ejecución."),
    ("lab2_1", 29, "plt.cm.get_cmap('jet', 10)", "plt.get_cmap('jet', 10)",
     "plt.cm.get_cmap se eliminó en Matplotlib 3.11 (AttributeError en tu entorno)."),
    ("lab2_1", 36, "plt.cm.get_cmap('jet', 10)", "plt.get_cmap('jet', 10)",
     "plt.cm.get_cmap se eliminó en Matplotlib 3.11 (AttributeError en tu entorno)."),
]

SKIP = {"lab1_1_resuelto", "lab1_2_resuelto"}  # ya están ejecutados (soluciones)


def main():
    src_root = Path(sys.argv[1] if len(sys.argv) > 1 else MANIFEST["defaultRoot"])
    RUN.mkdir(parents=True, exist_ok=True)
    for ds in MANIFEST["datasets"]:
        shutil.copyfile(src_root / ds["path"], RUN / ds["target"])
    report = {"generated": time.strftime("%Y-%m-%d %H:%M"), "python": sys.version.split()[0], "notebooks": {}}
    try:
        import numpy, pandas, scipy, sklearn, matplotlib, seaborn  # noqa: E401
        report["versions"] = {m.__name__: m.__version__ for m in (numpy, pandas, scipy, sklearn, matplotlib, seaborn)}
    except Exception:
        pass
    only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
    for nb_meta in MANIFEST["notebooks"]:
        nb_id = nb_meta["id"]
        if nb_id in SKIP or (only and nb_id not in only):
            continue
        nb = nbformat.read(src_root / nb_meta["path"], as_version=4)
        applied = []
        for pid, cell, old, new, why in PATCHES:
            if pid != nb_id:
                continue
            src = nb.cells[cell].source
            if old is None:
                lines = [new if "read_csv" in l else l for l in src.split("\n")]
                nb.cells[cell].source = "\n".join(lines)
            else:
                nb.cells[cell].source = src.replace(old, new)
            applied.append({"cell": cell, "why": why, "new": new})
        t0 = time.time()
        client = NotebookClient(nb, timeout=900, kernel_name="python3", allow_errors=True,
                                resources={"metadata": {"path": str(RUN)}})
        client.execute()
        errors = []
        for i, c in enumerate(nb.cells):
            for o in c.get("outputs", []):
                if o.get("output_type") == "error":
                    errors.append({"cell": i, "error": f"{o.get('ename')}: {o.get('evalue')}"[:300]})
        # se restaura el código original en la copia guardada: lo ejecutado se documenta en el informe
        orig = nbformat.read(src_root / nb_meta["path"], as_version=4)
        for i, c in enumerate(nb.cells):
            if c.cell_type == "code":
                c.source = orig.cells[i].source
        nbformat.write(nb, OUT / f"{nb_id}.ipynb")
        secs = round(time.time() - t0, 1)
        report["notebooks"][nb_id] = {"seconds": secs, "patches": applied, "errors": errors,
                                      "cells_with_output": sum(1 for c in nb.cells if c.get("outputs"))}
        print(f"{nb_id}: {secs} s, {len(errors)} errores, {len(applied)} arreglos")
        for e in errors:
            print("   ", e)
    (OUT / "informe.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
