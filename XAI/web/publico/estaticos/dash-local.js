// =============================================================================
// dash-local.js: reemplaza las funciones que el dashboard de Claude le da a la
// página (dash.data, dash.draw, dash.calc, dash.params, ...), para que la misma
// página funcione en cualquier servidor. Los datos se leen de /datos/<id>.json.
// =============================================================================
(() => {
  const DATOS = {};      // id -> { status: 'loading' | 'ok' | 'error', data }
  const calcs = {};      // id -> { inputs, fn }
  const dibujos = {};    // nombre -> función (el elemento)
  const oyentes = [];    // dash.onData
  let params = {};
  try { params = JSON.parse(sessionStorage.getItem('ei-params') || '{}') || {}; } catch (e) { params = {}; }

  // Paleta de los gráficos (la misma idea que el dashboard: un color por serie)
  const PALETA = {
    claro: ['#2a6fd6', '#d9572b', '#2f9466', '#7d52d6', '#b8890f', '#cc4785', '#3d97ad', '#6f6f6f'],
    oscuro: ['#5b9bf0', '#ee7646', '#4cba86', '#a07ef0', '#dcae35', '#e66aa3', '#5bb8cf', '#9a9a9a'],
  };
  const raizHtml = document.documentElement;
  const oscuro = () => raizHtml.dataset.theme === 'dark' ||
    (!raizHtml.dataset.theme && window.matchMedia('(prefers-color-scheme: dark)').matches);

  // Se dibuja solo lo que se ve: los gráficos usan el ancho de su caja
  const visible = (el) => el.getClientRects().length > 0;
  const correr = (el) => {
    const fn = dibujos[el.dataset.draw];
    if (!fn) return;
    try { fn(el); } catch (e) { console.error(`dibujo "${el.dataset.draw}":`, e); }
  };
  let pendiente = 0;
  const render = () => {
    pendiente = 0;
    oyentes.forEach((f) => { try { f(); } catch (e) { console.error('onData:', e); } });
    document.querySelectorAll('[data-draw]').forEach((el) => { if (visible(el)) correr(el); });
  };
  // setTimeout y no requestAnimationFrame: este último no corre con la pestaña en segundo plano
  const programar = () => { if (!pendiente) pendiente = setTimeout(render, 0); };

  window.dash = {
    get colors() { return oscuro() ? PALETA.oscuro : PALETA.claro; },
    data(id) {
      const c = calcs[id];
      if (c) {
        const ins = c.inputs.map((i) => window.dash.data(i));
        if (ins.some((x) => x.status === 'error')) return { status: 'error', data: [] };
        if (ins.some((x) => x.status !== 'ok')) return { status: 'loading', data: [] };
        return { status: 'ok', data: c.fn(...ins.map((x) => x.data)) };
      }
      return DATOS[id] || { status: 'loading', data: [] };
    },
    draw(nombre, fn) {
      if (fn) { dibujos[nombre] = fn; return; }
      document.querySelectorAll(`[data-draw="${nombre}"]`).forEach(correr);
    },
    calc(id, c) { calcs[id] = c; },
    onData(fn) { oyentes.push(fn); },
    params() { return params; },
    setParams(p) {
      params = { ...params, ...p };
      try { sessionStorage.setItem('ei-params', JSON.stringify(params)); } catch (e) { /* sin almacenamiento */ }
      programar();
    },
    setLink(tab) { try { history.replaceState(null, '', `#${tab}`); } catch (e) { /* nada */ } programar(); },
  };

  // Carga la página del dashboard y sus datos
  window.cargarDashboard = async (raiz, urlPagina) => {
    const html = await (await fetch(urlPagina, { cache: 'no-store' })).text();
    raiz.innerHTML = html;
    const ids = [...new Set([...raiz.querySelectorAll('[data-source]')].flatMap((e) => e.dataset.source.split(/\s+/)).filter(Boolean))];
    ids.forEach((id) => { DATOS[id] = { status: 'loading', data: [] }; });
    // Los scripts insertados con innerHTML no corren: se recrean en orden
    for (const viejo of [...raiz.querySelectorAll('script')]) {
      const nuevo = document.createElement('script');
      [...viejo.attributes].forEach((a) => nuevo.setAttribute(a.name, a.value));
      nuevo.textContent = viejo.textContent;
      viejo.replaceWith(nuevo);
    }
    render();
    await Promise.all(ids.map(async (id) => {
      if (calcs[id]) { delete DATOS[id]; return; } // es un cálculo de la página, no un archivo
      try {
        const r = await fetch(`datos/${encodeURIComponent(id)}.json`);
        if (!r.ok) throw new Error(r.status);
        DATOS[id] = { status: 'ok', data: await r.json() };
      } catch (e) {
        console.error(`datos "${id}":`, e);
        DATOS[id] = { status: 'error', data: [] };
      }
      programar();
    }));
  };

  // Al cambiar el tamaño de la ventana o el tema, se vuelve a dibujar
  let espera;
  let ancho = window.innerWidth;
  window.addEventListener('resize', () => {
    if (window.innerWidth === ancho) return; // en el celular, la barra del navegador cambia solo el alto
    ancho = window.innerWidth;
    clearTimeout(espera); espera = setTimeout(programar, 150);
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', programar);
  window.redibujarDashboard = programar;
})();
