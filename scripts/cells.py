"""Lista las celdas de un notebook importado con números de línea (ayuda para redactar contenidos)."""
import json, sys
nb = json.load(open(f"public/materials/notebooks/{sys.argv[1]}.json", encoding="utf-8"))
only = set(map(int, sys.argv[2].split(","))) if len(sys.argv) > 2 else None
for c in nb["cells"]:
    if only and c["index"] not in only:
        continue
    if c["type"] != "code":
        print(f"## [{c['index']}] md: {c['source'][:90]!r}")
        continue
    print(f"## [{c['index']}] code")
    for i, l in enumerate(c["source"].split("\n"), 1):
        print(f"{i:3d}| {l}")
