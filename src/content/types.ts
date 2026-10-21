// Esquema de los contenidos educativos. Todo el contenido vive en /content y se registra en
// content/index.ts; la interfaz solo lee estas estructuras.

/** Origen de un contenido: permite distinguir material de la asignatura de soluciones y ampliaciones. */
export type Origin = 'original' | 'modificado' | 'solucion' | 'ampliacion' | 'app';

/** Etapas del flujo de un notebook: datos → exploración → preparación → método → resultados → interpretación. */
export type Stage = 'contexto' | 'datos' | 'exploracion' | 'preparacion' | 'metodo' | 'resultados' | 'interpretacion';

/** Habilidades que se registran en el progreso. */
export type Skill = 'concepto' | 'codigo' | 'api' | 'interpretacion' | 'escritura' | 'metodo';

/** Categorías de error que distingue el corrector. */
export type ErrorCategory = 'sintaxis' | 'dimensiones' | 'api' | 'razonamiento' | 'ejecucion' | 'tiempo' | 'concepto';

/** Referencia a la fuente de una explicación o ejercicio. */
export interface SourceRef {
  notebook?: string; // id de materials.json (p. ej. 'lab1_1')
  cells?: number[]; // índices de celda (desde 0, como en el archivo .ipynb)
  doc?: string; // id de documento de materials.json (guía, slides, report…)
  section?: string; // pregunta o apartado: 'Q2', 'p. 3', 'Case 1'…
  origin: Origin;
  note?: string;
}

export interface ApiItem {
  name: string; // p. ej. 'rng.multivariate_normal'
  lib: string; // numpy, pandas, scipy.stats, sklearn, matplotlib, seaborn, python
  kind: 'función' | 'método' | 'clase' | 'atributo' | 'argumento' | 'operador' | 'módulo' | 'palabra clave';
  does: string;
  args?: { name: string; value?: string; meaning: string }[];
  returns?: string;
  learned?: string[]; // atributos aprendidos tras fit (terminan en _)
}

export interface LineNote {
  cell: number;
  lines: [number, number]; // 1-based dentro de la celda, inclusivo
  text: string;
}

export interface ShapeNote {
  expr: string;
  shape: string;
  meaning: string;
}

export interface Correction {
  cell: number;
  kind: 'errata' | 'api-obsoleta' | 'adaptacion' | 'concepto' | 'modificado';
  original: string;
  fixed: string;
  why: string;
}

export interface Block {
  id: string;
  title: string;
  cells: number[];
  stage: Stage;
  /** Origen del bloque si difiere del notebook (p. ej. celdas añadidas o soluciones tuyas). */
  origin?: Origin;
  /** Si el bloque contiene una solución: su código se oculta hasta intentar esta actividad. */
  hiddenUntil?: string;
  problem: string; // qué problema resuelve
  purpose: string; // para qué sirve dentro del notebook
  inputs: string; // qué datos recibe
  outputs: string; // qué produce
  prereqs: string[]; // conceptos previos (ids del glosario o texto)
  apis: ApiItem[];
  lines: LineNote[];
  shapes?: ShapeNote[];
  interpretation: string;
  prev?: string;
  next?: string;
  whatIf: { change: string; effect: string }[];
  corrections?: Correction[];
  /** Diferencias conocidas entre tu entorno local y el de la app (versiones de librerías). */
  versionNotes?: { cell: number; text: string }[];
  /** Código que ejecuta la app si difiere del original (se muestra como «adaptado»). */
  runCode?: string;
  /** Datos que necesita cargar el kernel antes de ejecutar. */
  data?: string[];
  /** Pregunta de control para avanzar en el recorrido guiado. */
  check?: Question;
  noRun?: boolean;
}

// ------------------------------------------------------------------ experimentos
export type ParamKind = 'datos' | 'preprocesado' | 'modelo' | 'evaluacion' | 'visualizacion';

export interface ParamOption {
  value: string; // literal de Python que se inserta en la plantilla
  label: string;
}

export interface Param {
  id: string; // marcador en la plantilla: {{id}}
  label: string;
  kind: ParamKind;
  control: 'select' | 'range' | 'toggle';
  options?: ParamOption[];
  min?: number;
  max?: number;
  step?: number;
  default: string; // literal de Python
  variable: string; // variable o argumento de Python que modifica
  computation: string; // efecto sobre el cálculo
  result: string; // efecto sobre el resultado
  why: string; // por qué ocurre
  origin?: Origin; // 'ampliacion' si el parámetro no aparece en el notebook
}

export interface MetricDef {
  key: string;
  label: string;
  format?: 'pct' | 'num' | 'int' | 'matrix' | 'text';
  digits?: number;
  better?: 'low' | 'high';
}

export interface Experiment {
  id: string;
  title: string;
  goal: string;
  blocks: string[];
  data?: string[];
  template: string;
  params: Param[];
  seedParam?: string;
  metrics: MetricDef[];
  observe: string[];
  source: SourceRef;
  /** Valores guardados en el notebook que la ejecución debe reproducir (verificación automática). */
  fidelity?: { params?: Record<string, string>; metrics: Record<string, number>; tol?: number; source: string }[];
}

// ------------------------------------------------------------------ preguntas y ejercicios
export interface FigureRef {
  experiment?: string;
  params?: Record<string, string>;
  figureIndex?: number;
  notebook?: string; // salida guardada del notebook original
  cell?: number;
  output?: number;
}

interface BaseActivity {
  id: string;
  skills: Skill[];
  source: SourceRef;
  blocks?: string[];
  tags?: string[];
  difficulty?: 1 | 2 | 3;
  transfer?: boolean; // práctica de transferencia (otro escenario o dataset)
  title?: string;
}

export interface McqQuestion extends BaseActivity {
  kind: 'mcq';
  prompt: string;
  code?: string;
  figure?: FigureRef;
  options: { text: string; correct?: boolean; feedback?: string }[];
  multiple?: boolean;
  explanation: string;
}

export interface NumericQuestion extends BaseActivity {
  kind: 'numeric';
  prompt: string;
  code?: string;
  figure?: FigureRef;
  answer: number;
  tolerance: number; // absoluta
  unit?: string;
  explanation: string;
}

export interface TextQuestion extends BaseActivity {
  kind: 'text';
  prompt: string;
  code?: string;
  figure?: FigureRef;
  rubric: string[]; // puntos que debe contener una buena respuesta
  model: string; // respuesta modelo (se muestra después de intentarlo)
}

export interface DynamicQuestion extends BaseActivity {
  kind: 'dynamic';
  /** Código Python que define el dict `variante` con: prompt, answer, tolerance, explanation, (options). Recibe SEED. */
  generator: string;
  data?: string[];
  answerKind: 'numeric' | 'mcq' | 'shape';
}

export type Question = McqQuestion | NumericQuestion | TextQuestion | DynamicQuestion;

/** Nivel 2: predecir y comprobar ejecutando. */
export interface PredictExercise extends BaseActivity {
  kind: 'predict';
  prompt: string;
  setup?: string;
  data?: string[];
  code: string; // código que se ejecuta para comprobar
  expr: string; // expresión Python cuyo valor es la verdad
  answerKind: 'shape' | 'numeric' | 'mcq' | 'text';
  options?: string[]; // para mcq: valores posibles (se compara con str(valor))
  tolerance?: number;
  explanation: string;
}

/** Niveles 3–6: completar, corregir, escribir de memoria y adaptar. */
export interface CodeExercise extends BaseActivity {
  kind: 'code';
  level: 3 | 4 | 5 | 6;
  title: string;
  prompt: string;
  available: string; // datos y variables disponibles
  produce: string; // qué debe producir el código
  practice?: string; // función o librería que se practica explícitamente
  setup: string;
  data?: string[];
  starter: string;
  solution: string;
  alternatives?: string[]; // soluciones alternativas válidas (se verifican automáticamente)
  wrong?: { code: string; expect: ErrorCategory }[]; // respuestas erróneas típicas (verificación)
  checks: string; // código Python con el objeto c (Checker)
  hints: string[];
  explanation?: string;
}

export type Activity = Question | PredictExercise | CodeExercise;

export const LEVEL_LABEL: Record<number, string> = {
  1: 'Leer y explicar',
  2: 'Predecir',
  3: 'Completar',
  4: 'Corregir',
  5: 'Escribir de memoria',
  6: 'Adaptar',
};

export function activityLevel(a: Activity): number {
  if (a.kind === 'code') return a.level;
  if (a.kind === 'predict') return 2;
  return 1;
}

// ------------------------------------------------------------------ notebooks y labs
export interface Practice {
  id: string;
  title: string;
  scenario: string;
  steps: string[]; // ids de actividades, en orden
  source: SourceRef;
}

export interface NotebookContent {
  id: string; // id de materials.json
  lab: string;
  title: string;
  short: string;
  origin: Origin;
  context: {
    goal: string;
    dataset: string;
    variables: { name: string; meaning: string; shape?: string }[];
    classes?: { id: string; label: string }[];
    questions: { id: string; text: string; blocks: string[]; doc?: string }[];
    notation: { symbol: string; meaning: string; code?: string }[];
  };
  flow: { stage: Stage; label: string; blocks: string[]; summary: string }[];
  blocks: Block[];
  experiments: Experiment[];
  activities: Activity[];
  recall: string[]; // ids de actividades en orden de retirada de ayudas
  practice: Practice[];
  notes?: string[];
  /** Si el notebook no existe en los materiales (p. ej. lo escribe el alumno). */
  proposed?: boolean;
}

export interface LabContent {
  id: string;
  title: string;
  folder: string;
  summary: string;
  topics: string[];
  notebooks: NotebookContent[];
  documents: string[];
  status: 'completo' | 'parcial' | 'pendiente';
  missing: string[];
  overview?: string; // markdown adicional (p. ej. selección vs reducción)
}

export interface GlossaryEntry {
  id: string;
  es: string;
  en: string;
  def: string;
  symbol?: string;
  labs: string[];
}

/** Retos globales que combinan varios labs; solo aparecen si sus labs están registrados. */
export interface GlobalChallenge {
  id: string;
  title: string;
  requires: string[];
  mode: 'contexto' | 'transferencia';
  scenario: string;
  steps: Activity[];
  review: Record<string, { notebook: string; block?: string; label: string }>; // por id de paso
}

/** Tarea de código que pide el enunciado (guía/report): qué se pedía, qué se ha hecho y el código. */
export interface LabTask {
  id: string;
  lab: string;
  notebook: string; // notebook en el que se apoya
  question: string; // 'Q6', 'Q2–Q5'…
  title: string;
  asks: string; // qué pide el enunciado (resumido de la guía)
  did: string; // qué se ha hecho y cómo leer el resultado
  code: string; // script autocontenido que se ejecuta
  data?: string[];
  author: 'solucion' | 'app'; // de tus soluciones o escrita por la app
  source: SourceRef;
}
