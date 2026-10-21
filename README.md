# MLEARN · Estudio de los laboratorios de Machine Learning from Data

Aplicación web local para estudiar los notebooks de los laboratorios: explicación bloque a bloque, experimentos con Python real y código sincronizado, interpretación de resultados, ejercicios de código con corrección automática, repaso espaciado y una sección global, **Ponte a prueba**, que conecta todos los labs.

## Puesta en marcha (Windows)

Requisitos: [Node.js](https://nodejs.org) 22 o superior y Python 3 (solo para importar los materiales).

```powershell
cd C:\PORTFOLIO\mlearn
npm install
npm run import        # lee tus materiales de LAB (sin modificarlos) y los copia a public/materials
npm run dev           # abre http://localhost:5175
```

- La primera vez, la app descarga Pyodide (Python para el navegador, ~40 MB) desde el CDN de jsDelivr; después queda en la caché del navegador. Para trabajar **sin conexión**: `npm run setup-offline` (lo descarga en `public/pyodide/`).
- Si tus materiales están en otra carpeta: `python scripts/import_materials.py "D:\ruta\a\LAB"`.
- Al añadir o cambiar materiales, vuelve a ejecutar `npm run import`: avisa de los notebooks cuyo contenido ha cambiado (SHA-256) para revisar sus explicaciones.
- El progreso se guarda en el navegador. En **Progreso** puedes exportarlo e importarlo (JSON).

### Ejecuciones completas y tareas del enunciado

```powershell
npm run refresh   # = npm run execute + npm run tasks + npm run import
```

- `npm run execute` ejecuta **copias** de todos los notebooks con tu entorno del laboratorio (`LAB/.venv`) y guarda las salidas en `executed/` (tus originales no se tocan; los arreglos mínimos —ruta de `iris.csv`, `get_cmap`— quedan anotados en `executed/informe.json` y en la web).
- `npm run tasks` ejecuta las **tareas de código del enunciado** (`content/tasks.ts`): lo que cada guía pide programar, con qué se pedía, qué se ha hecho, el código y sus salidas. Se ven en la pestaña **G · Tareas del enunciado** de cada notebook, en la página de cada lab y al final de los PDF.
- Debajo de cada celda se muestra tu salida guardada; si la celda no la tiene, la de la ejecución completa (etiquetada).

### Estudiar en la tablet

- **PDF para anotar:** en cada notebook, botón **«🖨 PDF para tablet»** → *Imprimir* → *Guardar como PDF*. Incluye contexto, flujo, cada bloque con su código numerado, notas por línea, explicaciones, tablas de funciones, salidas guardadas, espacio para notas y las preguntas con hueco para responder (las soluciones solo si marcas «Anexo de soluciones»). Ábrelo en tu app de notas para subrayar y escribir.
- **La web en la tablet:** con el PC y la tablet en la misma wifi, arranca `npm run dev:tablet` y abre en la tablet la dirección «Network» que muestra la terminal (p. ej. `http://192.168.1.20:5175`). Python funciona también en la tablet, aunque la primera carga es más lenta. El progreso se guarda en cada dispositivo por separado: usa *Progreso → Exportar/Importar* para pasarlo.

### Comprobaciones

```powershell
npm run verify            # ejecuta con el mismo motor Python todos los ejercicios, experimentos y celdas
npm run verify -- lab1_1  # solo un notebook (o «global» para los retos de Ponte a prueba)
npm run coverage-matrix   # regenera docs/cobertura.md
npm run typecheck
```

`npm run verify` comprueba que cada solución de referencia y cada solución alternativa aprueban, que las respuestas erróneas típicas suspenden con la categoría de error esperada, que el código inicial no aprueba solo, que los experimentos reproducen los valores guardados en tus notebooks y que las celdas originales se ejecutan y dan la misma salida de texto. El resultado queda en `docs/verificacion*.txt`.

## Qué hay dentro

| Sección | Contenido |
|---|---|
| **Laboratorios** | Un módulo por lab; dentro, un recorrido por notebook con seis actividades: A Contexto · B Código · C Experimentar · D Interpretar · E Recuperar de memoria · F Práctica. Tres modos: recorrido guiado, estudio libre y práctica de examen. |
| **Ponte a prueba** | Preguntas mezcladas de todos los labs, código sin indicar la función, figuras y métricas, detectar y corregir, comparar métodos, predicción, retos multietapa (contexto original y transferencia con Wine y Breast Cancer), simulacro con tiempo opcional y repaso de errores. Al terminar, cada fallo enlaza con el notebook y el bloque que conviene repasar. |
| **Repaso** | Repetición espaciada (Leitner) e historial de errores por categoría. |
| **Progreso** | Por notebook y por habilidad (concepto, código, librerías, interpretación, escritura autónoma, elección de métodos); distingue leído / con pistas / autónomo. |
| **Glosario** | Español–inglés, con símbolos. |
| **Cobertura** | Matriz celda a celda: bloque, concepto, funciones, explicación, experimento, actividades y estado. |
| **Materiales** | Archivos importados, documentos de apoyo y versiones de las librerías (app frente a tu entorno local). |

Documentos del proyecto: [docs/inventario.md](docs/inventario.md) (inventario de materiales, erratas y lo que falta) y [docs/cobertura.md](docs/cobertura.md).

## Ejecución aislada del código

El código (tuyo y de los experimentos) se ejecuta en **Pyodide** (CPython compilado a WebAssembly, con numpy, pandas, scipy, scikit-learn, matplotlib y seaborn) dentro de un **Web Worker**:

- no tiene acceso a tus archivos: solo ve un sistema de archivos virtual en memoria con los datasets (`/data`) y `/tmp`;
- no puede importar los módulos que dan acceso al navegador o a la red (`js`, `pyodide`, `micropip`, `urllib`, `socket`…) y la política de seguridad (CSP) solo le permite conectar con la propia app y con el CDN de Pyodide;
- cada ejecución tiene un límite de tiempo (30 s en ejercicios) y, si el motor no responde, se reinicia;
- la corrección se basa en ejecutar tu código y compararlo con la solución de referencia ejecutada en el mismo entorno, con tolerancias numéricas y aceptando otros nombres de variable; no compara texto. No se usa IA: el feedback se genera a partir del error real (tipo de excepción, línea, forma de los arrays, diferencia numérica).

## Arquitectura

```
content/                 contenidos educativos (separados de la interfaz)
  materials.json         qué notebooks, datasets y documentos se importan
  labs/<lab>/<nb>.ts     bloques, explicaciones, experimentos, actividades y prácticas de cada notebook
  labs/<lab>/index.ts    datos del lab (resumen, documentos, pendientes)
  global/index.ts        retos de Ponte a prueba (declaran qué labs necesitan)
  glossary.ts            glosario ES–EN
  index.ts               registro de labs
public/py/               runtime Python: ejecución tipo celda, figuras, aislamiento y corrector
src/engine/              arranque de Pyodide y Web Worker
src/content/types.ts     esquema de los contenidos
src/activities/          preguntas, predicciones, ejercicios de código y simulacros
src/pages/               interfaz
src/progress/            progreso, errores, repaso espaciado
scripts/                 importación de materiales, verificación, cobertura, descarga offline
```

### Añadir un laboratorio (p. ej. el siguiente)

1. Copia la carpeta del lab en `LAB/` y añade sus notebooks, datasets y documentos a `content/materials.json`.
2. `npm run import`.
3. Crea `content/labs/lab3/<notebook>.ts` (usa `python -I scripts/cells.py <id>` para ver las celdas con números de línea) y `content/labs/lab3/index.ts`, y regístralo en `content/index.ts`.
4. `npm run verify -- <id>` hasta 0 fallos y `npm run coverage-matrix`.

Sus actividades entran solas en Ponte a prueba, el repaso, el progreso y la cobertura; los retos globales que declaren `requires: ['lab3']` aparecerán cuando el lab esté registrado.

## Materiales de la asignatura

Los notebooks, guías y slides son de la asignatura (UPC) y **no se incluyen en el repositorio**: `public/materials/` está en `.gitignore` y se genera localmente con `npm run import`. El repositorio contiene el código de la app y las explicaciones y ejercicios propios, que citan las celdas por su número; los experimentos y ejercicios incluyen fragmentos de código adaptados de los notebooks.
