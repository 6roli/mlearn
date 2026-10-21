"""Entorno de ejecución de la app (se ejecuta dentro de Pyodide, en un Web Worker).

Responsabilidades:
- Ejecutar código como una celda de Jupyter: stdout, valor de la última expresión y figuras.
- Mantener «kernels» (espacios de nombres) por notebook para ejecutar bloques en orden.
- Aislar el código del usuario: sin módulos de acceso a JavaScript/red y con límite de tiempo.
- Clasificar los errores (sintaxis, dimensiones, API, razonamiento, ejecución, tiempo).
"""
import ast
import base64
import builtins
import contextlib
import io
import os
import sys
import time
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

USER_FILENAME = "<tu código>"
CELL_FILENAME = "<celda>"
FIG_DPI = 90
MAX_REPR = 20_000
MAX_STDOUT = 60_000

_FIGS = []
_KERNELS = {}

# ---------------------------------------------------------------- aislamiento
# Módulos que dan acceso al navegador (JavaScript, red): bloqueados para cualquier import durante
# la ejecución del usuario, también los indirectos (importlib, __import__ de librerías).
BRIDGE_ROOTS = {"js", "pyodide_js", "pyodide", "_pyodide", "micropip", "pyodide_http"}
# Además, el código del usuario no puede importar directamente módulos de sistema o red
# (las librerías científicas sí pueden usarlos internamente).
BLOCKED_ROOTS = BRIDGE_ROOTS | {"ctypes", "subprocess", "socket", "urllib", "urllib3", "requests", "http",
                                "webbrowser", "multiprocessing"}


class _SandboxImportBlocker:
    """Impide importar módulos que dan acceso al navegador durante la ejecución del usuario."""

    active = False

    def find_spec(self, name, path=None, target=None):
        if self.active and name.split(".")[0] in BRIDGE_ROOTS:
            raise ImportError(f"El módulo '{name}' no está disponible en el entorno aislado de la app.")
        return None


_BLOCKER = _SandboxImportBlocker()
sys.meta_path.insert(0, _BLOCKER)
_real_import = builtins.__import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if _BLOCKER.active and level == 0 and name.split(".")[0] in BLOCKED_ROOTS:
        raise ImportError(f"El módulo '{name}' no está disponible en el entorno aislado de la app.")
    return _real_import(name, globals, locals, fromlist, level)


_ALLOWED_READ = ("/data/", "/usr/lib/", "/lib/", "/tmp/")
_real_open = builtins.open


def _guarded_open(file, mode="r", *args, **kwargs):
    path = os.path.abspath(str(file))
    writing = any(m in mode for m in "wax+")
    if writing and not path.startswith("/tmp/"):
        raise PermissionError("En el entorno aislado solo se puede escribir en /tmp.")
    if not writing and not path.startswith(_ALLOWED_READ):
        raise PermissionError(f"Archivo no disponible en el entorno aislado: {file}")
    return _real_open(file, mode, *args, **kwargs)


def _safe_builtins():
    b = dict(builtins.__dict__)
    b["__import__"] = _guarded_import
    b["open"] = _guarded_open
    b["input"] = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("input() no está disponible."))
    b["exit"] = b["quit"] = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("exit() no está disponible."))
    return b


@contextlib.contextmanager
def sandboxed(time_limit):
    """Activa el bloqueo de módulos y un límite de tiempo para el código del usuario."""
    stash = {k: sys.modules.pop(k) for k in list(sys.modules) if k.split(".")[0] in {"js", "pyodide_js"}}
    deadline = time.monotonic() + time_limit

    def tracer(frame, event, arg):
        if frame.f_code.co_filename not in (USER_FILENAME, CELL_FILENAME):
            return None

        def local(frame, event, arg):
            if time.monotonic() > deadline:
                raise TimeoutError(f"Se superó el tiempo máximo de ejecución ({time_limit:.0f} s).")
            return local

        return local

    _BLOCKER.active = True
    sys.settrace(tracer)
    try:
        yield
    finally:
        sys.settrace(None)
        _BLOCKER.active = False
        sys.modules.update(stash)


# ---------------------------------------------------------------- figuras
def _capture_open_figures():
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", dpi=FIG_DPI, bbox_inches="tight")
            _FIGS.append({
                "png": base64.b64encode(buf.getvalue()).decode("ascii"),
                "axes": len(fig.axes),
                "titles": [ax.get_title() for ax in fig.axes if ax.get_title()],
                "suptitle": fig._suptitle.get_text() if getattr(fig, "_suptitle", None) else "",
            })
        except Exception as exc:  # figura incompleta: no debe romper la ejecución
            _FIGS.append({"png": "", "axes": len(fig.axes), "titles": [], "suptitle": "",
                          "error": f"{type(exc).__name__}: {exc}"})
    plt.close("all")


def _show(*args, **kwargs):
    _capture_open_figures()


plt.show = _show


def figure_summaries():
    return [{k: v for k, v in f.items() if k != "png"} for f in _FIGS]


# ---------------------------------------------------------------- errores
ALIASES = {"np", "pd", "plt", "sns", "ss", "sp", "stats", "norm", "default_rng", "PCA", "TSNE",
           "LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis", "accuracy_score",
           "confusion_matrix", "roc_curve", "RocCurveDisplay", "train_test_split", "metrics",
           "load_digits", "Path", "eigvals"}
DIM_WORDS = ("shape", "dimension", "broadcast", "aligned", "size", "axis", "index", "reshape",
             "same length", "same number", "1d", "2d", "3d", "Expected 2D", "too many indices",
             "inconsistent numbers of samples", "matmul", "out of bounds")


def classify_exception(exc):
    name = type(exc).__name__
    msg = str(exc)
    low = msg.lower()
    if isinstance(exc, (SyntaxError, IndentationError)):
        return "sintaxis"
    if isinstance(exc, TimeoutError):
        return "tiempo"
    if isinstance(exc, NameError):
        missing = getattr(exc, "name", None)
        if not missing and "'" in msg:
            missing = msg.split("'")[1]
        return "api" if missing in ALIASES else "ejecucion"
    if isinstance(exc, (AttributeError, ImportError, KeyError)):
        return "api"
    if isinstance(exc, TypeError):
        if any(w in low for w in ("argument", "keyword", "positional", "not callable", "takes", "missing")):
            return "api"
        return "ejecucion"
    if isinstance(exc, (ValueError, IndexError)) or name == "LinAlgError":
        if any(w.lower() in low for w in DIM_WORDS):
            return "dimensiones"
        return "razonamiento" if name == "LinAlgError" else "ejecucion"
    return "ejecucion"


def explain_exception(exc, ns=None):
    """Mensaje en español que parte de la evidencia (tipo y texto del error)."""
    cat = classify_exception(exc)
    msg = str(exc)
    tips = {
        "sintaxis": "Python no puede interpretar el código. Revisa paréntesis, comas, dos puntos y la sangría en la línea indicada.",
        "tiempo": "El código tarda demasiado. Busca bucles innecesarios o tamaños de arrays muy grandes.",
        "dimensiones": "Las dimensiones de los arrays no encajan. Comprueba .shape de cada operando y el argumento axis.",
        "api": "La función, el atributo o el argumento no existe con ese nombre o esa forma de uso. Revisa la librería y la firma de la función.",
        "ejecucion": "El código se ejecuta pero falla en una instrucción concreta. Lee el mensaje y revisa qué variables existen en ese punto.",
        "razonamiento": "El cálculo no tiene sentido con estos datos (por ejemplo, una matriz singular).",
    }
    detail = tips[cat]
    if isinstance(exc, NameError):
        missing = getattr(exc, "name", None)
        if missing in ALIASES:
            detail = f"'{missing}' no está definido: falta importar la librería o el alias (por ejemplo, import numpy as np)."
        elif missing:
            detail = f"La variable '{missing}' no existe en este punto. ¿La has definido antes o tiene otro nombre?"
    if isinstance(exc, KeyError) and ns is not None:
        cols = []
        for v in ns.values():
            if type(v).__name__ == "DataFrame":
                cols = list(map(str, v.columns))[:12]
                break
        if cols:
            detail = f"La clave {msg} no existe. Columnas disponibles en el DataFrame: {', '.join(cols)}."
    return cat, detail


def _user_line(tb, filename):
    line = None
    for frame, lineno in traceback.walk_tb(tb):
        if frame.f_code.co_filename == filename:
            line = lineno
    return line


# ---------------------------------------------------------------- ejecución
def _short_repr(value):
    try:
        if type(value).__name__ in ("DataFrame", "Series"):
            text = value.to_string(max_rows=60, max_cols=20)
        else:
            text = repr(value)
    except Exception as exc:
        text = f"<repr falló: {exc}>"
    return text[:MAX_REPR]


def run_code(code, ns, filename=CELL_FILENAME, time_limit=60.0, sandbox=True, display_last=True):
    """Ejecuta `code` en `ns` como una celda. Devuelve un dict serializable."""
    out, err = io.StringIO(), io.StringIO()
    _FIGS.clear()
    plt.close("all")
    result = {"stdout": "", "stderr": "", "display": None, "figures": [], "error": None}
    try:
        tree = ast.parse(code, filename=filename, mode="exec")
    except SyntaxError as exc:
        result["error"] = {"type": type(exc).__name__, "message": exc.msg, "line": exc.lineno,
                           "category": "sintaxis",
                           "detail": "Python no puede interpretar el código. Revisa la línea indicada (paréntesis, comas, dos puntos, sangría).",
                           "traceback": "".join(traceback.format_exception_only(type(exc), exc))}
        return result
    last_expr = None
    if display_last and tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = ast.Expression(tree.body.pop().value)
    ctx = sandboxed(time_limit) if sandbox else contextlib.nullcontext()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), ctx:
            exec(compile(tree, filename, "exec"), ns)
            if last_expr is not None:
                value = eval(compile(last_expr, filename, "eval"), ns)
                if value is not None and not _is_matplotlib_artist(value):
                    result["display"] = _short_repr(value)
    except BaseException as exc:  # noqa: BLE001 - queremos informar de cualquier fallo del usuario
        cat, detail = explain_exception(exc, ns)
        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000],
                           "line": _user_line(exc.__traceback__, filename), "category": cat,
                           "detail": detail, "traceback": "".join(tb_lines[-6:])[:6000]}
    _capture_open_figures()
    result["figures"] = list(_FIGS)
    result["stdout"] = out.getvalue()[:MAX_STDOUT]
    result["stderr"] = _clean_warnings(err.getvalue())[:MAX_STDOUT]
    return result


def _is_matplotlib_artist(value):
    mod = type(value).__module__ or ""
    if mod.startswith(("matplotlib", "seaborn")):
        return True
    if isinstance(value, (list, tuple)) and value and (type(value[0]).__module__ or "").startswith("matplotlib"):
        return True
    return False


def _clean_warnings(text):
    keep = []
    for line in text.splitlines():
        if "warnings.warn(" in line:
            continue
        keep.append(line)
    return "\n".join(keep)


def new_namespace():
    ns = {"__name__": "__main__", "__builtins__": _safe_builtins()}
    return ns


def kernel(kernel_id, reset=False):
    if reset or kernel_id not in _KERNELS:
        _KERNELS[kernel_id] = new_namespace()
    return _KERNELS[kernel_id]


def run_in_kernel(kernel_id, code, reset=False, time_limit=120.0):
    ns = kernel(kernel_id, reset)
    return run_code(code, ns, CELL_FILENAME, time_limit)


def kernel_variables(kernel_id):
    """Resumen de variables del kernel (nombre, tipo, forma) para el panel de estado."""
    ns = _KERNELS.get(kernel_id, {})
    rows = []
    for k, v in ns.items():
        if k.startswith("_") or k == "__builtins__":
            continue
        t = type(v)
        if t.__name__ in ("module", "function", "type", "builtin_function_or_method") or callable(v) and not hasattr(v, "shape"):
            continue
        shape = getattr(v, "shape", None)
        rows.append({"name": k, "type": f"{t.__module__.split('.')[0]}.{t.__name__}" if t.__module__ != "builtins" else t.__name__,
                     "shape": list(shape) if isinstance(shape, tuple) else None,
                     "len": len(v) if isinstance(v, (list, tuple, dict, str)) else None})
    return rows


os.makedirs("/data", exist_ok=True)
os.makedirs("/tmp/work", exist_ok=True)
os.chdir("/data")
