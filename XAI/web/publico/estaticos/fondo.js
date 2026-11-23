// =============================================================================
// fondo.js: animaciones interactivas en los bordes vacíos de la página.
//   - Modelo inmunológico: células de partículas (linfocitos, macrófagos, neutrófilos).
//   - Modelo epigenético: dos hélices de ADN de partículas, una en cada borde,
//     con una hebra azul y la otra naranja.
// Basado en las animaciones originales celulas.html y Adn_animado_azul_naranja.html
// (copias en dashboard/Media/animaciones/). Cambios para usarlas de fondo:
//   - se dibujan solo en los bordes, a los lados de la columna del contenido,
//     y se desvanecen al acercarse a ella para no tapar el texto;
//   - el lienzo es transparente: en tema oscuro brillan (mezcla aditiva) y en
//     tema claro se dibujan con colores más oscuros;
//   - se elige la animación del modelo que está abierto y cambian con un fundido;
//   - el mouse se escucha en toda la ventana (el lienzo no recibe clics).
// =============================================================================
(() => {
  const capas = {
    inmuno: document.querySelector('.fondo-capa[data-fondo="inmuno"]'),
    epi: document.querySelector('.fondo-capa[data-fondo="epi"]'),
  };
  if (!capas.inmuno || !capas.epi) return;
  const quieto = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const rnd = Math.random;
  const gauss = () => (rnd() + rnd() + rnd() - 1.5) / 1.5;
  const MIN_BORDE = 90; // px: con bordes más angostos no se dibuja nada

  // Estado común a las dos animaciones
  const S = {
    W: 0, H: 0, dpr: 1, L: 0, R: 0, oscuro: true, modelo: null, mvx: 0, mvy: 0,
    mouse: { x: -9999, y: -9999, px: -9999, py: -9999, activo: false },
  };

  // Columna del contenido (todo lo que esté fuera son "bordes")
  const leerColumna = () => {
    const el = document.querySelector('#dash-root .ei') || document.getElementById('dash-root');
    const r = el ? el.getBoundingClientRect() : { left: S.W / 2, right: S.W / 2 };
    const L = Math.round(r.left), R = Math.round(r.right);
    const cambio = Math.abs(L - S.L) > 2 || Math.abs(R - S.R) > 2;
    S.L = L; S.R = R;
    return cambio;
  };
  const leerTema = () => {
    const c = (getComputedStyle(document.body).backgroundColor.match(/[\d.]+/g) || [0, 0, 0]).map(Number);
    S.oscuro = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] < 128;
  };
  const leerModelo = () => {
    const epi = document.querySelector('[data-modelo="epi"]');
    return epi && !epi.hidden ? 'epi' : 'inmuno';
  };
  const bandas = () => [[0, S.L], [S.R, S.W]].filter(([a, b]) => b - a >= MIN_BORDE);
  // Opacidad según la posición: 1 en los bordes, 0 dentro de la columna del contenido
  const fundido = (x) => {
    const fuera = x < S.L ? S.L - x : x > S.R ? x - S.R : -Math.min(x - S.L, S.R - x);
    const a = (fuera + 16) / 90;
    return a <= 0 ? 0 : a >= 1 ? 1 : a;
  };

  // ---------------------------------------------------------------------------
  // Células
  // ---------------------------------------------------------------------------
  function crearCelulas(canvas) {
    const ctx = canvas.getContext('2d');
    const HOME_SPRING = 0.0016, DRAG = 0.935, SPIN_BASE = 0.004, MOUSE_R = 170, MOUSE_PUSH = 2.6;
    const TIPOS = {
      oscuro: [{ shell: '120,215,235', nuc: '60,130,225' }, { shell: '200,150,255', nuc: '135,75,225' }, { shell: '255,190,120', nuc: '235,115,70' }],
      claro: [{ shell: '30,130,165', nuc: '30,80,185' }, { shell: '115,65,200', nuc: '85,35,170' }, { shell: '205,115,45', nuc: '185,70,30' }],
    };
    let cells = [], dust = [], SC = 1;
    const sphere = () => { const u = rnd() * 2 - 1, th = rnd() * 6.2832, s = Math.sqrt(1 - u * u); return [s * Math.cos(th), s * Math.sin(th), u]; };
    const inSphere = (rad) => { const p = sphere(), r = rad * Math.cbrt(rnd()); return [p[0] * r, p[1] * r, p[2] * r]; };

    function makeCell(type, x, y, z) {
      const base = type === 1 ? 62 + rnd() * 26 : (type === 2 ? 34 + rnd() * 10 : 28 + rnd() * 12);
      const r = base * SC * (0.6 + 0.8 * z);
      const pts = []; // [x, y, z, parte]  parte 0 = membrana, 1 = núcleo
      let i, p, k;
      const shellN = type === 1 ? 300 : 230;
      const ph = [rnd() * 6, rnd() * 6, rnd() * 6];
      for (i = 0; i < shellN; i++) {
        p = sphere();
        let rr = 1 + gauss() * 0.03;
        if (type === 1) rr += 0.14 * (Math.sin(2.3 * p[0] + ph[0]) + Math.sin(2.1 * p[1] + ph[1]) + Math.sin(2.7 * p[2] + ph[2])) / 3 * 2;
        else if (type === 2) rr += 0.05 * Math.sin(9 * p[0] + ph[0]) * Math.sin(9 * p[1] + ph[1]);
        pts.push([p[0] * rr, p[1] * rr, p[2] * rr, 0]);
      }
      if (type === 1) { // seudópodos del macrófago
        const tent = 6 + (rnd() * 3 | 0);
        for (k = 0; k < tent; k++) {
          const d = sphere(), len = 0.5 + rnd() * 0.6, bend = [gauss() * 0.4, gauss() * 0.4, gauss() * 0.4];
          for (i = 0; i < 14; i++) {
            const t = i / 13, rr2 = 1 + t * len;
            pts.push([d[0] * rr2 + bend[0] * t * t + gauss() * 0.02, d[1] * rr2 + bend[1] * t * t + gauss() * 0.02, d[2] * rr2 + bend[2] * t * t + gauss() * 0.02, 0]);
          }
        }
      }
      if (type === 0) { // linfocito: núcleo grande y redondo
        for (i = 0; i < 110; i++) { p = inSphere(0.58); pts.push([p[0] + 0.10, p[1] - 0.05, p[2], 1]); }
      } else if (type === 1) { // macrófago: núcleo en forma de riñón
        let n = 0;
        while (n < 85) {
          p = inSphere(0.46);
          const dx = p[0] - 0.22;
          if (dx * dx + p[1] * p[1] + p[2] * p[2] < 0.07) continue;
          pts.push([p[0] - 0.08, p[1], p[2], 1]); n++;
        }
      } else { // neutrófilo: núcleo con lóbulos
        const lobes = [[0.36, 0.02, 0.05], [-0.22, 0.32, 0.1], [-0.16, -0.34, -0.1]];
        for (k = 0; k < lobes.length; k++) for (i = 0; i < 36; i++) { p = inSphere(0.27); pts.push([p[0] + lobes[k][0], p[1] + lobes[k][1], p[2] + lobes[k][2], 1]); }
        for (i = 0; i < 16; i++) {
          const a = lobes[i % 2], b = lobes[(i % 2) + 1], t2 = rnd();
          pts.push([a[0] + (b[0] - a[0]) * t2 + gauss() * 0.03, a[1] + (b[1] - a[1]) * t2 + gauss() * 0.03, a[2] + (b[2] - a[2]) * t2, 1]);
        }
      }
      const n2 = pts.length;
      const c = {
        type, z, r, sc: SC, hitR: r * (type === 1 ? 1.35 : 1.1), x, y, hx: x, hy: y, vx: 0, vy: 0,
        wph: [rnd() * 6.28, rnd() * 6.28], wf: [0.25 + rnd() * 0.3, 0.2 + rnd() * 0.3], wa: (10 + rnd() * 18) * SC,
        ax: rnd() * 6.28, ay: rnd() * 6.28,
        sx: (rnd() - 0.5) * 2 * SPIN_BASE * 1.6, sy: (rnd() - 0.5) * 2 * SPIN_BASE * 1.6,
        n: n2, lx: new Float32Array(n2), ly: new Float32Array(n2), lz: new Float32Array(n2),
        part: new Uint8Array(n2), lag: new Float32Array(n2), tw: new Float32Array(n2), sz: new Float32Array(n2),
      };
      for (i = 0; i < n2; i++) {
        c.lx[i] = pts[i][0]; c.ly[i] = pts[i][1]; c.lz[i] = pts[i][2]; c.part[i] = pts[i][3];
        c.lag[i] = 0.3 + rnd() * 1.1; c.tw[i] = rnd() * 6.28;
        c.sz[i] = pts[i][3] ? 1.1 + rnd() * 1.0 : 0.9 + rnd() * rnd() * 1.5;
      }
      return c;
    }

    function init() {
      cells = []; dust = [];
      for (const [a, b] of bandas()) {
        const w = b - a;
        // Tamaño de las células según el ancho del borde
        SC = Math.max(0.45, Math.min(1, w / 330)) * Math.max(0.75, Math.min(1.15, S.H / 850));
        const count = Math.max(4, Math.min(24, Math.round(w * S.H / 16000)));
        let tries = 0, n = 0;
        while (n < count && tries < count * 40) {
          tries++;
          const z = 0.25 + rnd() * 0.75, t = rnd(), type = t < 0.5 ? 0 : (t < 0.78 ? 2 : 1);
          const c = makeCell(type, a + rnd() * w, rnd() * S.H, z);
          let ok = true;
          for (const o of cells) {
            const dx = c.x - o.x, dy = c.y - o.y, min = (c.hitR + o.hitR) * 0.9;
            if (dx * dx + dy * dy < min * min) { ok = false; break; }
          }
          if (ok) { cells.push(c); n++; }
        }
        for (let i = 0; i < Math.round(w * S.H / 2600); i++) {
          dust.push({ x: a + rnd() * w, a0: a, w, y: rnd() * S.H, v: 0.05 + rnd() * 0.25, s: 0.5 + rnd() * 1.1, al: 0.1 + rnd() * 0.3, ph: rnd() * 6.28 });
        }
      }
      cells.sort((p, q) => p.z - q.z); // las lejanas primero
    }

    function step(time) {
      const m = S.mouse;
      for (const c of cells) {
        const tx = c.hx + Math.sin(time * c.wf[0] + c.wph[0]) * c.wa;
        const ty = c.hy + Math.cos(time * c.wf[1] + c.wph[1]) * c.wa;
        c.vx += (tx - c.x) * HOME_SPRING;
        c.vy += (ty - c.y) * HOME_SPRING;
        if (m.activo) {
          const dx = c.x - m.x, dy = c.y - m.y, dist = Math.sqrt(dx * dx + dy * dy) || 0.001, reach = MOUSE_R * c.sc + c.hitR;
          if (dist < reach) {
            const f = 1 - dist / reach, push = f * f * MOUSE_PUSH;
            c.vx += (dx / dist) * push + S.mvx * 0.012 * f;
            c.vy += (dy / dist) * push + S.mvy * 0.012 * f;
            c.sy += (dx / dist) * f * 0.0006;
            c.sx += (dy / dist) * f * 0.0006;
          }
        }
      }
      for (let i = 0; i < cells.length; i++) {
        for (let j = i + 1; j < cells.length; j++) {
          const a = cells[i], b = cells[j], ddx = b.x - a.x, ddy = b.y - a.y, min = (a.hitR + b.hitR) * 0.92, d2 = ddx * ddx + ddy * ddy;
          if (d2 < min * min && d2 > 0.01) {
            const d = Math.sqrt(d2), o = (min - d) / min * 0.12, nx = ddx / d, ny = ddy / d;
            a.vx -= nx * o; a.vy -= ny * o; b.vx += nx * o; b.vy += ny * o;
          }
        }
      }
      for (const c of cells) {
        c.vx *= DRAG; c.vy *= DRAG; c.x += c.vx; c.y += c.vy;
        c.sx *= 0.995; c.sy *= 0.995;
        c.ax += c.sx + (c.sx >= 0 ? SPIN_BASE : -SPIN_BASE) * 0.6;
        c.ay += c.sy + (c.sy >= 0 ? SPIN_BASE : -SPIN_BASE) * 0.6;
      }
    }

    function frame(time, mover) {
      if (mover) step(time);
      ctx.globalCompositeOperation = 'source-over';
      ctx.clearRect(0, 0, S.W, S.H);
      ctx.globalCompositeOperation = S.oscuro ? 'lighter' : 'source-over';
      const tipos = S.oscuro ? TIPOS.oscuro : TIPOS.claro, tono = S.oscuro ? 1.1 : 0.95;
      ctx.fillStyle = S.oscuro ? 'rgb(150,130,210)' : 'rgb(110,90,170)';
      for (const q of dust) {
        if (mover) {
          q.y -= q.v; q.x += Math.sin(time * 0.5 + q.ph) * 0.12;
          if (q.y < -2) { q.y = S.H + 2; q.x = q.a0 + rnd() * q.w; }
        }
        const f = fundido(q.x); if (!f) continue;
        ctx.globalAlpha = q.al * (0.6 + 0.4 * Math.sin(time * 1.5 + q.ph)) * f * tono;
        ctx.fillRect(q.x, q.y, q.s, q.s);
      }
      const m = S.mouse;
      for (const c of cells) {
        if (!fundido(c.x - c.r * 1.6) && !fundido(c.x + c.r * 1.6)) continue; // toda dentro del contenido
        const T = tipos[c.type];
        const cax = Math.cos(c.ax), sax = Math.sin(c.ax), cay = Math.cos(c.ay), say = Math.sin(c.ay);
        const zA = 0.5 + 0.5 * c.z, lagx = c.vx * 5, lagy = c.vy * 5;
        let glow = 1;
        if (m.activo) {
          const gx = c.x - m.x, gy = c.y - m.y, gd = Math.sqrt(gx * gx + gy * gy), lim = MOUSE_R * c.sc * 1.6;
          if (gd < lim) glow = 1 + 0.35 * (1 - gd / lim);
        }
        let lastPart = -1;
        for (let k = 0; k < c.n; k++) {
          const x = c.lx[k], y = c.ly[k], z = c.lz[k];
          const x1 = x * cay + z * say, z1 = -x * say + z * cay;
          const y2 = y * cax - z1 * sax, z2 = y * sax + z1 * cax;
          const persp = 1 / (1 - z2 * 0.12);
          const px = c.x + x1 * c.r * persp - lagx * c.lag[k], py = c.y + y2 * c.r * persp - lagy * c.lag[k];
          if (px < -6 || px > S.W + 6 || py < -6 || py > S.H + 6) continue;
          const f = fundido(px); if (!f) continue;
          let depth = (z2 + 1) * 0.5; depth = depth < 0 ? 0 : depth > 1 ? 1 : depth;
          const part = c.part[k];
          if (part !== lastPart) { ctx.fillStyle = `rgb(${part ? T.nuc : T.shell})`; lastPart = part; }
          const tw = 0.8 + 0.2 * Math.sin(time * 2 + c.tw[k]);
          let alpha = (part ? 0.5 + 0.5 * depth : 0.38 + 0.62 * depth) * zA * tw * glow * f * tono;
          if (alpha > 1) alpha = 1;
          const s = c.sz[k] * (0.6 + 0.8 * c.z) * (0.75 + 0.5 * depth);
          ctx.globalAlpha = alpha;
          ctx.fillRect(px - s * 0.5, py - s * 0.5, s, s);
        }
      }
      ctx.globalAlpha = 1;
    }
    return { canvas, ctx, init, frame };
  }

  // ---------------------------------------------------------------------------
  // ADN: una hélice en cada borde, inclinadas en espejo y girando en sentido contrario
  // ---------------------------------------------------------------------------
  function crearAdn(canvas) {
    const ctx = canvas.getContext('2d');
    const TURNS = 4.2, ROT_SPEED = 0.0075, MOUSE_R = 110, SPRING = 0.022, DAMPING = 0.915;
    // Una hebra azul y la otra naranja (3 tonos de cada una); en tema claro, tonos más oscuros
    const COLORES = {
      oscuro: ['105,150,225', '40,95,200', '18,50,140', '245,170,100', '225,110,45', '160,60,20'],
      claro: ['60,100,190', '30,70,170', '20,45,125', '215,130,55', '195,90,30', '140,50,15'],
    };
    const smooth = (a, b, x) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
    let helices = [], P = [], N = 0, ox, oy, vx, vy, rot = 0;

    function init() {
      helices = bandas().map(([a, b]) => {
        const w = b - a, half = Math.min(S.H * 0.42, 520), izquierda = a === 0;
        const rad = Math.min(half * 0.2, w * 0.24);
        return { cx: a + w / 2, cy: S.H * 0.5, half, rad, tilt: izquierda ? 0.2 : -0.2, giro: izquierda ? 1 : -1 };
      });
      P = [];
      helices.forEach((h, hi) => {
        const k = Math.max(0.35, Math.min(1, h.rad / 86)); // 86 px: el radio de la animación original
        const add = (o) => { o.h = hi; P.push(o); };
        for (let s = 0; s < 2; s++) {
          for (let i = 0, n = Math.round(4200 * k); i < n; i++) {
            add({ kind: 0, side: s, u: rnd(), ph: s * Math.PI, rr: 1 + gauss() * 0.14, jx: gauss() * 9 * k, jy: gauss() * 9 * k, jz: gauss() * 9 * k });
          }
        }
        for (let r = 0; r < 46; r++) {
          const ur = (r + 0.5) / 46;
          for (let i = 0, n = Math.round(10 + 30 * k); i < n; i++) add({ kind: 1, u: ur, f: rnd() * 2 - 1, rr: 1, jx: gauss() * 1.2, jy: gauss() * 1.2, jz: gauss() * 1.2 });
        }
        for (let i = 0, n = Math.round(1700 * k); i < n; i++) {
          add({ kind: 2, u: -0.14 + rnd() * 0.40, ph: rnd() * 6.2832, rr: 0.3 + rnd() * rnd() * 2.6, jx: gauss() * 6 * k, jy: gauss() * 6 * k, jz: gauss() * 6 * k });
        }
      });
      N = P.length;
      for (const o of P) {
        o.rx = gauss(); o.ry = rnd(); o.rz = gauss(); o.tw = rnd() * 6.2832;
        o.sz = 0.55 + rnd() * rnd() * 1.6; o.b = rnd();
        // Tono (claro, medio, profundo) + color: cada hebra el suyo; los peldaños, mitad azul y mitad naranja; el polvo, al azar
        const c = rnd(), tono = c < 0.45 ? 0 : (c < 0.8 ? 1 : 2);
        const naranja = o.kind === 1 ? (o.f > 0 ? 0 : 1) : (o.kind === 2 ? (rnd() < 0.5 ? 0 : 1) : o.side);
        o.col = tono + 3 * naranja;
      }
      P.sort((a, b) => a.col - b.col); // pocas veces se cambia el color de relleno
      ox = new Float32Array(N); oy = new Float32Array(N); vx = new Float32Array(N); vy = new Float32Array(N);
    }

    function frame(time, mover) {
      ctx.globalCompositeOperation = 'source-over';
      ctx.clearRect(0, 0, S.W, S.H);
      ctx.globalCompositeOperation = S.oscuro ? 'lighter' : 'source-over';
      // El fondo de la página es gris oscuro (no negro como en el original): un poco más de brillo
      const colores = S.oscuro ? COLORES.oscuro : COLORES.claro, tono = S.oscuro ? 1.35 : 1.15;
      const m = S.mouse, R2 = MOUSE_R * MOUSE_R, twoPi = 6.2832;
      let lastCol = -1;
      for (let i = 0; i < N; i++) {
        const p = P[i], h = helices[p.h], u = p.u;
        const cosT = Math.cos(h.tilt), sinT = Math.sin(h.tilt);
        let y = (1 - 2 * u) * h.half + p.jy;
        const a = u * TURNS * twoPi + (p.ph || 0) + rot * h.giro;
        const ff = p.kind === 1 ? p.f : p.rr;
        let xr = Math.cos(a) * h.rad * ff + p.jx;
        let z = Math.sin(a) * h.rad * ff + p.jz;
        // Disolución: un extremo de la hélice se deshace en nube
        let d = 1 - smooth(-0.04, 0.28, u);
        if (p.kind === 2) d = Math.max(d, 0.6);
        let al = 1;
        if (d > 0) {
          const sh = 0.75 + 0.25 * Math.sin(time * 0.8 + p.tw);
          xr += p.rx * d * h.rad * 2.8 * sh;
          z += p.rz * d * h.rad * 1.6;
          y += d * d * (0.10 + p.ry * 0.55) * h.half * sh;
          al = 1 - 0.45 * d;
        }
        const persp = 1 / (1 - z / (h.rad * 7));
        const bx = xr * persp, by = y * persp;
        const sx = h.cx + bx * cosT - by * sinT, sy = h.cy + bx * sinT + by * cosT;
        let depth = (z / (h.rad * 1.4) + 1) * 0.5; depth = depth < 0 ? 0 : depth > 1 ? 1 : depth;
        if (mover) {
          const px = sx + ox[i], py = sy + oy[i];
          if (m.activo) {
            const dx = px - m.x, dy = py - m.y, d2 = dx * dx + dy * dy;
            if (d2 < R2) {
              const dist = Math.sqrt(d2) || 0.001, f = 1 - dist / MOUSE_R, push = f * f * 7 + f * 1.5;
              vx[i] += (dx / dist) * push + (rnd() - 0.5) * 4 * f + S.mvx * 0.05 * f;
              vy[i] += (dy / dist) * push + (rnd() - 0.5) * 4 * f + S.mvy * 0.05 * f;
            }
          }
          vx[i] = (vx[i] - ox[i] * SPRING) * DAMPING;
          vy[i] = (vy[i] - oy[i] * SPRING) * DAMPING;
          ox[i] += vx[i]; oy[i] += vy[i];
        }
        const fx = sx + ox[i], fy = sy + oy[i];
        if (fx < -4 || fx > S.W + 4 || fy < -4 || fy > S.H + 4) continue;
        const fd = fundido(fx); if (!fd) continue;
        const tw = 0.75 + 0.25 * Math.sin(time * 2.2 + p.tw);
        const disp = Math.min(1, Math.hypot(ox[i], oy[i]) / 70);
        let alpha = 0.7 * (0.30 + 0.70 * depth) * (0.55 + 0.45 * p.b) * tw * al * (1 + disp * 0.5) * fd * tono;
        if (alpha > 1) alpha = 1;
        if (p.col !== lastCol) { ctx.fillStyle = `rgb(${colores[p.col]})`; lastCol = p.col; }
        ctx.globalAlpha = alpha;
        const sz = p.sz * (0.7 + depth * 0.8);
        ctx.fillRect(fx - sz * 0.5, fy - sz * 0.5, sz, sz);
      }
      ctx.globalAlpha = 1;
      if (mover) rot += ROT_SPEED;
    }
    return { canvas, ctx, init, frame };
  }

  // ---------------------------------------------------------------------------
  // Bucle común
  // ---------------------------------------------------------------------------
  const anims = { inmuno: crearCelulas(capas.inmuno), epi: crearAdn(capas.epi) };
  let time = 0, cuadro = 0;
  const hasta = { inmuno: 0, epi: 0 }; // la capa que se apaga sigue animándose durante el fundido

  function redimensionar() {
    S.dpr = Math.min(window.devicePixelRatio || 1, 2);
    S.W = window.innerWidth; S.H = window.innerHeight;
    for (const a of Object.values(anims)) {
      a.canvas.width = Math.round(S.W * S.dpr);
      a.canvas.height = Math.round(S.H * S.dpr);
      a.ctx.setTransform(S.dpr, 0, 0, S.dpr, 0, 0);
    }
    leerColumna();
    for (const a of Object.values(anims)) a.init();
  }

  function revisar() {
    if (leerColumna()) Object.values(anims).forEach((a) => a.init());
    leerTema();
    const m = leerModelo();
    if (m !== S.modelo) {
      if (S.modelo) hasta[S.modelo] = performance.now() + 1000;
      S.modelo = m;
      Object.entries(capas).forEach(([k, c]) => c.classList.toggle('activo', k === m));
      return true;
    }
    return false;
  }

  function bucle() {
    if (cuadro++ % 20 === 0) revisar();
    const m = S.mouse;
    S.mvx = m.activo ? m.x - m.px : 0; S.mvy = m.activo ? m.y - m.py : 0;
    m.px = m.x; m.py = m.y;
    const ahora = performance.now();
    for (const [k, a] of Object.entries(anims)) {
      if (k === S.modelo || ahora < hasta[k]) a.frame(time, true);
    }
    time += 0.016;
    requestAnimationFrame(bucle);
  }

  // Mouse en toda la ventana: el lienzo está detrás del contenido y no recibe eventos
  const poner = (e) => { S.mouse.x = e.clientX; S.mouse.y = e.clientY; S.mouse.activo = true; };
  const quitar = () => { S.mouse.activo = false; S.mouse.x = S.mouse.y = -9999; };
  window.addEventListener('pointermove', poner, { passive: true });
  window.addEventListener('pointerdown', poner, { passive: true });
  window.addEventListener('pointerup', (e) => { if (e.pointerType !== 'mouse') quitar(); });
  document.documentElement.addEventListener('pointerleave', quitar);
  window.addEventListener('blur', quitar);

  let espera;
  window.addEventListener('resize', () => { clearTimeout(espera); espera = setTimeout(() => { redimensionar(); if (quieto) pintarQuieto(); }, 150); });

  // Movimiento reducido: una imagen fija, que se vuelve a dibujar si cambia el modelo, el tema o el tamaño
  function pintarQuieto() { leerTema(); anims[S.modelo || leerModelo()].frame(0, false); }

  redimensionar();
  if (quieto) {
    revisar(); pintarQuieto();
    setInterval(() => {
      const tema = S.oscuro, cambio = revisar(); // revisar() también vuelve a leer el tema
      if (cambio || tema !== S.oscuro) pintarQuieto();
    }, 1000);
  } else {
    revisar(); // elige la capa de inmediato, sin esperar el primer cuadro
    requestAnimationFrame(bucle);
  }
  window.fondoRevisar = () => { revisar(); if (quieto) pintarQuieto(); };
})();
