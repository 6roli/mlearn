import { createEngine } from './lib/node-engine.ts';
const t0 = Date.now();
const e = await createEngine();
console.log('boot', (Date.now() - t0) / 1000);
await e.data(['iris.csv']);
let r: any = e.call('run', { kernel_id: 'k', code: "import pandas as pd\niris = pd.read_csv('iris.csv')\niris.describe()", reset: true });
console.log('display:', r.display?.slice(0, 200), r.error);
r = e.call('run', { kernel_id: 'k', code: "import seaborn as sns\nsns.pairplot(iris, hue='species', height=2);" });
console.log('figs', r.figures.length, r.figures[0]?.axes, r.error);
r = e.call('run', { kernel_id: 'k', code: "import js" });
console.log('block js:', r.error?.category, r.error?.message);
r = e.call('run', { kernel_id: 'k', code: "import sys\nsys.modules.get('js')" });
console.log('sys.modules js:', r.display, r.error?.message);
r = e.call('run', { kernel_id: 'k', code: "open('/home/pyodide/mlearn/grader.py').read()[:10]" });
console.log('open:', r.error?.message);
r = e.call('run', { kernel_id: 'k', code: "x=0\nwhile True:\n    x+=1", time_limit: 2 });
console.log('timeout:', r.error?.category, r.error?.message);
const ex = {
  ex_id: 't1', setup: "import pandas as pd, numpy as np\niris = pd.read_csv('iris.csv')",
  solution: "x = iris[iris['species']=='Iris-setosa']['sepal_length']\nmedia = x.mean()",
  checks: "c.same('media', rtol=1e-9)\nc.shape('x', (50,))",
};
for (const code of [
  "x = iris.loc[iris.species=='Iris-setosa','sepal_length']\nmedia = np.mean(x)",
  "x = iris[iris['species']=='Iris-setosa']['sepal_length']\nm = x.mean()",
  "x = iris[iris['species']=='setosa']['sepal_length']\nmedia = x.mean()",
  "x = iris[iris['species']=='Iris-setosa']['sepal-length']",
  "x = iris[iris['species']=='Iris-setosa'][['sepal_length']]\nmedia = x.mean()",
  "x = iris[iris['species']=='Iris-setosa']['sepal_length'\nmedia=1",
]) {
  r = e.call('grade', { ...ex, user_code: code });
  console.log('---', JSON.stringify(code).slice(0, 70));
  console.log(r.passed, r.run.error ? [r.run.error.category, r.run.error.detail] : r.items.map((i: any) => [i.ok, i.message, i.category]));
}
