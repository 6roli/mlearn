"""Corrección de ejercicios basada en ejecución y comprobaciones verificables.

Cada ejercicio define:
- setup: código que prepara los datos disponibles (igual para el usuario y para la referencia).
- solution: solución de referencia (nunca se muestra antes de intentarlo).
- checks: código Python que usa el objeto `c` (Checker) para comparar el resultado del usuario
  con el de la referencia. No se compara el texto del código, sino sus efectos.
"""
import json
import math
import warnings

import numpy as np

import mlearn_runtime as rt

_REF_CACHE = {}


def _to_array(v):
    try:
        if type(v).__name__ in ("DataFrame", "Series"):
            return np.asarray(v.to_numpy(), dtype=float)
        return np.asarray(v, dtype=float)
    except Exception:
        return None


def _fmt(v, limit=300):
    try:
        if isinstance(v, np.ndarray):
            text = np.array2string(v, precision=5, threshold=40)
        elif type(v).__name__ in ("DataFrame", "Series"):
            text = v.to_string(max_rows=12, max_cols=8)
        elif isinstance(v, float):
            text = f"{v:.6g}"
        else:
            text = repr(v)
    except Exception:
        text = "<valor>"
    return text if len(text) <= limit else text[:limit] + " …"


def _shape_of(v):
    shape = getattr(v, "shape", None)
    if isinstance(shape, tuple):
        return shape
    if isinstance(v, (list, tuple)):
        try:
            return np.asarray(v).shape
        except Exception:
            return (len(v),)
    return ()


class Checker:
    def __init__(self, ns, ref, run, ref_run):
        self.ns, self.ref, self.run, self.ref_run = ns, ref, run, ref_run
        self.items = []

    # -- registro --------------------------------------------------------
    def _add(self, ok, message, category="razonamiento", detail=None, reveal=None):
        self.items.append({"ok": bool(ok), "message": message, "category": None if ok else category,
                           "detail": detail, "reveal": reveal})
        return bool(ok)

    def ok(self, message):
        return self._add(True, message)

    def fail(self, message, category="razonamiento", detail=None):
        return self._add(False, message, category, detail)

    # -- utilidades de búsqueda -----------------------------------------
    def user_vars(self):
        out = {}
        for k, v in self.ns.items():
            if k.startswith("_") or type(v).__name__ == "module" or (callable(v) and not hasattr(v, "shape")):
                continue
            out[k] = v
        return out

    def _matches(self, a, b, rtol, atol):
        if isinstance(b, str):
            return isinstance(a, str) and a.strip().lower() == b.strip().lower()
        if isinstance(b, (bool, np.bool_)):
            return a is not None and bool(a) == bool(b) and not isinstance(a, str)
        A, B = _to_array(a), _to_array(b)
        if A is None or B is None:
            try:
                return a == b
            except Exception:
                return False
        if A.shape != B.shape:
            if A.size == B.size and A.size > 0:
                return False
            return False
        if A.size == 0:
            return True
        return bool(np.allclose(A, B, rtol=rtol, atol=atol, equal_nan=True))

    def locate(self, name, ref_name=None, rtol=1e-6, atol=1e-9):
        """Devuelve (nombre_en_usuario, valor). Acepta otro nombre si el valor coincide."""
        ref_value = self.ref.get(ref_name or name)
        if name in self.ns:
            return name, self.ns[name]
        if ref_value is None:
            return None, None
        for k, v in self.user_vars().items():
            if k in self.ref and k != (ref_name or name):
                continue
            if self._matches(v, ref_value, rtol, atol):
                return k, v
        return None, None

    # -- comprobaciones ---------------------------------------------------
    def same(self, name, ref_name=None, rtol=1e-6, atol=1e-9, desc=None, sort_rows=False, show_expected=True):
        """El valor del usuario coincide con el de la referencia (con tolerancia)."""
        label = desc or f"`{name}`"
        if (ref_name or name) not in self.ref:
            return self.fail(f"(Ejercicio mal definido: la referencia no produce `{ref_name or name}`)", "ejecucion")
        expected = self.ref[ref_name or name]
        found, value = self.locate(name, ref_name, rtol, atol)
        if found is None:
            return self.fail(f"No encuentro {label}. Guarda el resultado en una variable llamada `{name}`.",
                             "ejecucion")
        note = "" if found == name else f" (lo he encontrado en `{found}`; el enunciado pide llamarlo `{name}`)"
        if isinstance(expected, str) or isinstance(value, str):
            ok = self._matches(value, expected, rtol, atol)
            return self._add(ok, f"{label} correcto{note}" if ok else f"{label} no coincide{note}",
                             "razonamiento", None, _fmt(expected) if show_expected else None)
        A, B = _to_array(value), _to_array(expected)
        if A is None:
            return self.fail(f"{label} no es numérico (tipo {type(value).__name__}).", "api")
        if B is not None and A.shape != B.shape:
            if A.size == B.size:
                return self.fail(f"{label} tiene forma {A.shape} y se esperaba {B.shape}{note}.", "dimensiones",
                                 "Mismos valores en otra disposición: revisa si hace falta transponer, reshape o ravel.")
            return self.fail(f"{label} tiene forma {A.shape} y se esperaba {B.shape}{note}.", "dimensiones",
                             "Revisa qué filas/columnas seleccionas y el argumento axis.")
        if sort_rows and A.ndim >= 1:
            A, B = np.sort(A, axis=0), np.sort(B, axis=0)
        if np.allclose(A, B, rtol=rtol, atol=atol, equal_nan=True):
            return self.ok(f"{label} correcto{note}")
        if np.isnan(A).any() and not np.isnan(B).any():
            return self.fail(f"{label} contiene NaN{note}.", "razonamiento",
                             "Suele indicar una selección vacía (un filtro que no coincide con ninguna fila) "
                             "o una operación sobre datos que faltan. Comprueba cuántas filas seleccionas.")
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rel = np.nanmax(np.abs(A - B) / np.maximum(np.abs(B), atol)) if B.size else 0
        hint = None
        if B.size and np.allclose(A, -B, rtol=rtol, atol=atol):
            hint = "Tiene el signo contrario al esperado."
        elif B.size and np.all(B != 0) and np.allclose(A / B, (A / B).flat[0], rtol=1e-6):
            hint = f"Difiere por un factor constante ≈ {(A / B).flat[0]:.4g} (¿ddof, una raíz cuadrada o un porcentaje?)."
        return self._add(False, f"{label} no coincide{note}: error relativo máximo {rel:.3g}.", "razonamiento",
                         hint, _fmt(expected) if show_expected else None)

    def value(self, name, expected, rtol=1e-6, atol=1e-9, desc=None):
        label = desc or f"`{name}`"
        if name not in self.ns:
            return self.fail(f"No encuentro la variable `{name}`.", "ejecucion")
        ok = self._matches(self.ns[name], expected, rtol, atol)
        return self._add(ok, f"{label} correcto" if ok else f"{label} no tiene el valor esperado.", "razonamiento",
                         None, _fmt(expected))

    def shape(self, name, expected=None, ref_name=None, desc=None):
        label = desc or f"`{name}`"
        if name not in self.ns:
            return self.fail(f"No encuentro la variable `{name}`.", "ejecucion")
        exp = tuple(expected) if expected is not None else _shape_of(self.ref[ref_name or name])
        got = _shape_of(self.ns[name])
        ok = tuple(got) == tuple(exp)
        return self._add(ok, f"{label} tiene forma {tuple(got)}" + ("" if ok else f"; se esperaba {exp}"),
                         "dimensiones")

    def isinstance_of(self, name, type_name, desc=None):
        label = desc or f"`{name}`"
        v = self.ns.get(name)
        if v is None:
            return self.fail(f"No encuentro la variable `{name}`.", "ejecucion")
        names = [c.__name__ for c in type(v).__mro__]
        ok = type_name in names
        return self._add(ok, f"{label} es {type(v).__name__}" + ("" if ok else f"; se esperaba {type_name}"), "api")

    def fitted(self, name, cls, attrs=(), desc=None):
        """Estimador de scikit-learn de la clase `cls` y entrenado (atributos aprendidos con _)."""
        label = desc or f"`{name}`"
        v = self.ns.get(name)
        if v is None:
            for k, cand in self.user_vars().items():
                if type(cand).__name__ == cls:
                    name, v = k, cand
                    break
        if v is None:
            return self.fail(f"No encuentro ningún objeto {cls}.", "api")
        if type(v).__name__ != cls:
            return self.fail(f"{label} es {type(v).__name__}; se esperaba {cls}.", "api")
        missing = [a for a in attrs if not hasattr(v, a)]
        if missing:
            return self.fail(f"{label} no está entrenado: le falta {', '.join(missing)}. ¿Has llamado a .fit()?",
                             "api")
        return self.ok(f"{label} es un {cls} entrenado")

    def param(self, name, attr, expected, desc=None):
        v = self.ns.get(name)
        got = getattr(v, attr, None) if v is not None else None
        ok = self._matches(got, expected, 1e-9, 1e-12) if not isinstance(expected, str) else got == expected
        return self._add(ok, (desc or f"`{name}.{attr}`") + (" correcto" if ok else f" = {got!r}; se esperaba {expected!r}"),
                         "api")

    def true(self, condition, ok_msg, fail_msg, category="razonamiento", detail=None):
        return self._add(bool(condition), ok_msg if condition else fail_msg, category, detail)

    def figures(self, min_count=1, min_axes=None, desc=None):
        figs = self.run.get("figures", [])
        n = len(figs)
        ok = n >= min_count
        if ok and min_axes is not None:
            ok = any(f.get("axes", 0) >= min_axes for f in figs)
            if not ok:
                return self.fail(desc or f"La figura debería tener al menos {min_axes} ejes (subplots).", "api")
        return self._add(ok, (desc or "Figura generada") if ok else f"Se esperaban {min_count} figura(s) y hay {n}.",
                         "api", "¿Has llamado a la función de dibujo y a plt.show()?")

    def stdout_contains(self, text, desc=None):
        ok = text.lower() in self.run.get("stdout", "").lower()
        return self._add(ok, desc or f"La salida contiene «{text}»" if ok else f"La salida no contiene «{text}».",
                         "razonamiento")

    def code_uses(self, token, desc=None, category="api"):
        """Solo para ejercicios que piden practicar una función concreta (se avisa en el enunciado)."""
        ok = token in self.run.get("_code", "")
        return self._add(ok, desc or (f"Usa {token}" if ok else f"El ejercicio pide practicar {token}."), category)


def _reference(ex_id, setup, solution):
    if ex_id in _REF_CACHE:
        return _REF_CACHE[ex_id]
    ns = rt.new_namespace()
    if setup:
        r = rt.run_code(setup, ns, rt.CELL_FILENAME, 300, sandbox=False, display_last=False)
        if r["error"]:
            raise RuntimeError(f"La preparación del ejercicio falla: {r['error']['message']}")
    run = rt.run_code(solution, ns, rt.CELL_FILENAME, 300, sandbox=False)
    if run["error"]:
        raise RuntimeError(f"La solución de referencia falla: {run['error']['message']}")
    _REF_CACHE[ex_id] = (ns, run)
    return ns, run


def grade(ex_id, setup, solution, checks, user_code, time_limit=30.0):
    ref_ns, ref_run = _reference(ex_id, setup, solution)
    ns = rt.new_namespace()
    if setup:
        prep = rt.run_code(setup, ns, rt.CELL_FILENAME, 300, sandbox=False, display_last=False)
        if prep["error"]:
            return {"passed": False, "items": [], "run": prep, "fatal": prep["error"]["message"]}
    run = rt.run_code(user_code, ns, rt.USER_FILENAME, time_limit, sandbox=True)
    if run["error"]:
        return {"passed": False, "items": [], "run": run}
    run["_code"] = user_code
    c = Checker(ns, ref_ns, run, ref_run)
    try:
        exec(checks, {"c": c, "np": np, "math": math, "ns": ns, "ref": ref_ns})
    except Exception as exc:  # una comprobación que falla por el formato del resultado del usuario
        cat, detail = rt.explain_exception(exc, ns)
        c.fail(f"No he podido comprobar el resultado: {type(exc).__name__}: {exc}", cat, detail)
    run.pop("_code", None)
    passed = bool(c.items) and all(i["ok"] for i in c.items)
    return {"passed": passed, "items": c.items, "run": run}


def evaluate(setup, code, expr):
    """Ejecuta código y devuelve el valor de `expr` (para ejercicios de predicción)."""
    ns = rt.new_namespace()
    if setup:
        rt.run_code(setup, ns, rt.CELL_FILENAME, 300, sandbox=False, display_last=False)
    run = rt.run_code(code, ns, rt.CELL_FILENAME, 120, sandbox=False)
    value = None
    if not run["error"] and expr:
        value = _jsonable(eval(expr, ns))
    return {"run": run, "value": value}


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if type(v).__name__ in ("DataFrame", "Series"):
        return json.loads(v.to_json())
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def run_experiment(code, time_limit=120.0):
    """Ejecuta una plantilla de experimento ya renderizada en un espacio limpio."""
    ns = rt.new_namespace()
    run = rt.run_code(code, ns, rt.CELL_FILENAME, time_limit, sandbox=False)
    metrics = ns.get("resultados")
    run["metrics"] = _jsonable(metrics) if isinstance(metrics, dict) else None
    return run


def run_generator(code, seed):
    """Ejecuta un generador de variantes: debe definir un dict `variante`."""
    ns = rt.new_namespace()
    ns["SEED"] = int(seed)
    run = rt.run_code(code, ns, rt.CELL_FILENAME, 120, sandbox=False, display_last=False)
    if run["error"]:
        return {"error": run["error"], "variant": None, "figures": run["figures"]}
    return {"error": None, "variant": _jsonable(ns.get("variante")), "figures": run["figures"]}


# ---------------------------------------------------------------- puente con JavaScript
def _json_default(o):
    v = _jsonable(o)
    return v if v is not o else str(o)


API = {
    "run": rt.run_in_kernel,
    "vars": rt.kernel_variables,
    "grade": grade,
    "evaluate": evaluate,
    "experiment": run_experiment,
    "generator": run_generator,
}


def call(name, args_json):
    """Punto de entrada único: recibe y devuelve JSON (texto)."""
    try:
        result = API[name](**json.loads(args_json))
        return json.dumps({"ok": True, "result": result}, default=_json_default)
    except Exception as exc:  # error interno de la app, no del usuario
        import traceback
        return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()[-3000:]})
