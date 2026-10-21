"""Importa los materiales de la asignatura a public/materials/ sin modificarlos.

Uso:  python scripts/import_materials.py ["C:/ruta/a/LAB"]

- Lee content/materials.json (qué notebooks, datasets y documentos usa la app).
- Extrae de cada notebook sus celdas (fuente literal) y salidas guardadas a JSON.
- Copia datasets y documentos de apoyo (PDF) para poder abrirlos desde la app.
- Calcula el SHA-256 de cada original y avisa si cambió respecto a la última importación,
  para saber qué contenidos de la app conviene revisar.
"""
import base64
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "content" / "materials.json").read_text(encoding="utf-8"))
OUT = ROOT / "public" / "materials"
MAX_IMAGE_BYTES = 900_000  # las figuras guardadas mayores se omiten para no inflar la app


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_outputs(outputs):
    result = []
    for o in outputs:
        kind = o.get("output_type")
        if kind == "stream":
            result.append({"type": "stream", "text": "".join(o.get("text", ""))})
        elif kind in ("execute_result", "display_data"):
            data = o.get("data", {})
            if "image/png" in data:
                png = data["image/png"]
                png = "".join(png) if isinstance(png, list) else png
                if len(png) <= MAX_IMAGE_BYTES:
                    result.append({"type": "image", "png": png.replace("\n", "")})
                else:
                    result.append({"type": "image-omitted"})
            elif "text/plain" in data:
                result.append({"type": "text", "text": "".join(data["text/plain"])})
        elif kind == "error":
            result.append({"type": "error", "text": f"{o.get('ename')}: {o.get('evalue')}"})
    return result


EXECUTED = ROOT / "executed"


def attach_execution(data, raw):
    """Añade las salidas de la ejecución completa (scripts/execute_notebooks.py), si existe."""
    path = EXECUTED / f"{data['id']}.ipynb"
    if not path.exists():
        return
    ex = json.loads(path.read_text(encoding="utf-8"))
    if len(ex["cells"]) != len(raw["cells"]):
        print(f"  AVISO: la ejecución completa de {data['id']} no coincide con el original; vuelve a ejecutarla")
        return
    for cell, exc, orig in zip(data["cells"], ex["cells"], raw["cells"]):
        if cell["type"] == "code" and "".join(exc.get("source", "")) == "".join(orig.get("source", "")):
            cell["execOutputs"] = convert_outputs(exc.get("outputs", []))
    info = {}
    report_path = EXECUTED / "informe.json"
    if report_path.exists():
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        info = {"generated": rep.get("generated"), "versions": rep.get("versions"),
                **rep.get("notebooks", {}).get(data["id"], {})}
    data["execution"] = info


def main():
    src_root = Path(sys.argv[1] if len(sys.argv) > 1 else MANIFEST["defaultRoot"])
    if not src_root.exists():
        sys.exit(f"No encuentro la carpeta de materiales: {src_root}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "notebooks").mkdir(exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    (OUT / "docs").mkdir(exist_ok=True)

    previous = {}
    index_path = OUT / "index.json"
    if index_path.exists():
        previous = {e["id"]: e["sha256"] for e in json.loads(index_path.read_text(encoding="utf-8"))["files"]}

    files, missing, changed = [], [], []

    for nb in MANIFEST["notebooks"]:
        path = src_root / nb["path"]
        if not path.exists():
            missing.append(nb["path"])
            continue
        digest = sha256(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        cells = []
        for i, c in enumerate(raw["cells"]):
            cells.append({
                "index": i,
                "type": c["cell_type"],
                "source": "".join(c.get("source", "")),
                "executionCount": c.get("execution_count"),
                "outputs": convert_outputs(c.get("outputs", [])) if c["cell_type"] == "code" else [],
            })
        data = {"id": nb["id"], "lab": nb["lab"], "path": nb["path"], "origin": nb["origin"],
                "sha256": digest, "cells": cells}
        attach_execution(data, raw)
        (OUT / "notebooks" / f"{nb['id']}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        files.append({"id": nb["id"], "kind": "notebook", "path": nb["path"], "sha256": digest})
        if nb["id"] in previous and previous[nb["id"]] != digest:
            changed.append(nb["path"])

    for ds in MANIFEST["datasets"]:
        path = src_root / ds["path"]
        if not path.exists():
            missing.append(ds["path"])
            continue
        shutil.copyfile(path, OUT / "data" / ds["target"])
        files.append({"id": ds["id"], "kind": "dataset", "path": ds["path"], "sha256": sha256(path)})

    for doc in MANIFEST["documents"]:
        path = src_root / doc["path"]
        if not path.exists():
            missing.append(doc["path"])
            continue
        target = OUT / "docs" / f"{doc['id']}{path.suffix.lower()}"
        shutil.copyfile(path, target)
        files.append({"id": doc["id"], "kind": "document", "path": doc["path"], "sha256": sha256(path),
                      "file": f"docs/{target.name}"})

    tasks_out = EXECUTED / "tasks_out.json"
    if tasks_out.exists():
        shutil.copyfile(tasks_out, OUT / "tasks.json")
        print("  Tareas del enunciado: salidas copiadas (executed/tasks_out.json)")
    index_path.write_text(json.dumps({"root": str(src_root), "files": files}, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    print(f"Importados {len(files)} archivos en {OUT}")
    for m in missing:
        print(f"  FALTA: {m}")
    for c in changed:
        print(f"  CAMBIADO desde la última importación (revisa sus contenidos): {c}")


if __name__ == "__main__":
    main()
