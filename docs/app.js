import { Detector } from './forest.js';

/* ---------------- seeded RNG ---------------- */
let _s = 20261023;
function rnd() {
  _s |= 0; _s = (_s + 0x6D2B79F5) | 0;
  let t = Math.imul(_s ^ (_s >>> 15), 1 | _s);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
function gauss(sd = 1) {
  let u = 0, v = 0;
  while (u === 0) u = rnd();
  while (v === 0) v = rnd();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * sd;
}

/* ---------------- channels ---------------- */
const CHANNELS = [
  { key: 'battery_v',     label: 'BATTERY',  unit: 'V',  color: '#22d3ee', min: 11.4, max: 13.0, fmt: v => v.toFixed(2) },
  { key: 'motor_current',  label: 'MOTOR I',  unit: 'A',  color: '#a78bfa', min: -0.5, max: 7.0,  fmt: v => v.toFixed(2) },
  { key: 'chassis_temp',   label: 'CHASSIS',  unit: '°C', color: '#f59e0b', min: 24,   max: 36,   fmt: v => v.toFixed(1) },
  { key: 'ph',             label: 'pH PROBE', unit: 'pH', color: '#34d399', min: 6.0,  max: 7.8,  fmt: v => v.toFixed(2) },
  { key: 'soil_moisture',  label: 'SOIL θ',   unit: '%',  color: '#60a5fa', min: 0,    max: 60,   fmt: v => (v * 100).toFixed(1), scale: v => v * 100 },
  { key: 'gas_mq',         label: 'GAS MQ',   unit: '',   color: '#f472b6', min: 0,    max: 0.55, fmt: v => v.toFixed(3) },
];

/* ---------------- simulator (mirrors train.py, streaming) ---------------- */
const sim = {
  t: 0, drive: 0, driveTarget: 0, driveLeft: 200,
  phWalk: 6.9, temp: 28, fault: null, spikeLeft: 0, spikeVal: 0,
};
function stepDrive() {
  if (--sim.driveLeft <= 0) {
    sim.driveTarget = sim.driveTarget > 0.5 ? 0 : 1;
    sim.driveLeft = sim.driveTarget > 0.5
      ? 90 + Math.floor(rnd() * 310)
      : 120 + Math.floor(rnd() * 300);
  }
  sim.drive += (sim.driveTarget - sim.drive) * 0.12;
}
function armFault(kind) {
  const F = {
    freeze:  { kind, ch: 3, dur: 120, val: null },
    drift:   { kind, ch: 5, dur: 150, rate: 0.004 },
    spikes:  { kind, ch: 1, dur: 200 },
    dropout: { kind, ch: 4, dur: 90 },
    gasleak: { kind, ch: 5, dur: 180 },
  }[kind];
  F.left = F.dur;
  sim.fault = F;
  document.querySelectorAll('#faultBtns button').forEach(b => {
    b.disabled = true;
    b.classList.toggle('armed', b.dataset.fault === kind);
  });
  const note = {
    freeze: 'pH probe fouled — value frozen. Stuck-sensor guardrail should trip in ~12 s.',
    drift: 'Gas sensor aging — slow upward bias. Watch the ML score climb before the guardrail.',
    spikes: 'EMI bursts on the motor-current line — 1–3 s transients.',
    dropout: 'Loose connector — 90 s of missing samples on soil moisture.',
    gasleak: 'A REAL leak, not a fault: fast rise, hold, decay on the gas channel.',
  }[kind];
  document.getElementById('faultNote').textContent = 'ACTIVE: ' + note;
}
function endFault() {
  sim.fault = null; sim.spikeLeft = 0;
  document.querySelectorAll('#faultBtns button').forEach(b => { b.disabled = false; b.classList.remove('armed'); });
  document.getElementById('faultNote').textContent =
    'Nominal telemetry streaming at 5× real time. Inject a fault and watch the detector catch it.';
}
function tickSim() {
  stepDrive();
  const t = sim.t++, d = sim.drive;
  const battery = 12.6 - 0.00022 * t - 0.25 * d + gauss(0.015);
  const current = 0.4 + 3.1 * d + 0.6 * Math.sin(2 * Math.PI * t / 47) * d + gauss(0.12);
  sim.temp += ((28 + 0.4 * Math.sin(2 * Math.PI * t / 1800) + 2.2 * d - sim.temp) * 0.02) + gauss(0.08);
  sim.phWalk += gauss(0.0016);
  sim.phWalk = Math.min(7.5, Math.max(6.3, sim.phWalk));
  const ph = sim.phWalk + gauss(0.02);
  const moist = Math.min(0.6, Math.max(0.06, 0.34 - 0.00002 * t - 0.03 * d)) + gauss(0.004);
  let gas = 0.040 + gauss(0.006);
  const v = [battery, current, sim.temp, ph, moist, gas];

  const f = sim.fault;
  if (f) {
    if (f.kind === 'freeze') {
      if (f.val === null) f.val = v[f.ch];
      v[f.ch] = f.val;
    } else if (f.kind === 'drift') {
      v[f.ch] += f.rate * (f.dur - f.left);
    } else if (f.kind === 'spikes') {
      if (sim.spikeLeft > 0) { v[f.ch] += sim.spikeVal; sim.spikeLeft--; }
      else if (rnd() < 0.06) {
        sim.spikeLeft = 1 + Math.floor(rnd() * 3);
        sim.spikeVal = (rnd() < 0.5 ? -1 : 1) * (1.5 + rnd() * 1.3);
      }
    } else if (f.kind === 'dropout') {
      v[f.ch] = NaN;
    } else if (f.kind === 'gasleak') {
      const e = f.dur - f.left;
      let add = 0;
      if (e < 30) add = 0.42 * (e / 30);
      else if (e < 120) add = 0.42;
      else add = 0.42 * Math.exp(-(e - 120) / 22);
      v[f.ch] += add;
    }
    if (--f.left <= 0) endFault();
  }
  return v;
}

/* ---------------- UI ---------------- */
const HIST = 150;
const histories = CHANNELS.map(() => []);
const cards = [];
function buildCards() {
  const wrap = document.getElementById('channels');
  CHANNELS.forEach((ch, i) => {
    const el = document.createElement('div');
    el.className = 'chan';
    el.innerHTML = `<div class="chan-head"><span class="chan-name">${ch.label}</span>
      <span class="chan-val" style="color:${ch.color}"><span class="vv">–</span><small>${ch.unit}</small></span></div>
      <canvas width="300" height="74"></canvas>
      <div class="chan-state">LINK OK</div>`;
    wrap.appendChild(el);
    cards.push({ el, cv: el.querySelector('canvas'), vv: el.querySelector('.vv'), st: el.querySelector('.chan-state') });
  });
}
function drawSpark(i, incident) {
  const ch = CHANNELS[i], { cv } = cards[i];
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const h = histories[i];
  const sc = v => H - 6 - ((ch.scale ? ch.scale(v) : v) - ch.min) / (ch.max - ch.min) * (H - 12);
  ctx.strokeStyle = 'rgba(148,163,184,0.12)';
  ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
  ctx.strokeStyle = ch.color; ctx.lineWidth = 1.6;
  ctx.shadowColor = ch.color; ctx.shadowBlur = incident ? 8 : 0;
  ctx.beginPath();
  let pen = false;
  for (let k = 0; k < h.length; k++) {
    const v = h[k];
    if (!Number.isFinite(v)) { pen = false; continue; }
    const x = k / (HIST - 1) * W, y = Math.max(2, Math.min(H - 2, sc(v)));
    if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;
}
const gauge = document.getElementById('gauge');
function drawGauge(score, thr) {
  const ctx = gauge.getContext('2d');
  const W = gauge.width, H = gauge.height;
  ctx.clearRect(0, 0, W, H);
  const cx = W / 2, cy = H - 16, R = 118;
  const a0 = Math.PI, a1 = 2 * Math.PI;
  const ang = s => a0 + s * (a1 - a0);
  ctx.lineWidth = 14; ctx.lineCap = 'round';
  const bg = ctx.createLinearGradient(0, 0, W, 0);
  bg.addColorStop(0, 'rgba(52,211,153,0.25)'); bg.addColorStop(0.6, 'rgba(245,158,11,0.25)'); bg.addColorStop(1, 'rgba(239,68,68,0.25)');
  ctx.strokeStyle = bg;
  ctx.beginPath(); ctx.arc(cx, cy, R, a0, a1); ctx.stroke();
  const col = score > thr ? '#ef4444' : score > thr * 0.8 ? '#f59e0b' : '#34d399';
  ctx.strokeStyle = col; ctx.shadowColor = col; ctx.shadowBlur = 12;
  ctx.beginPath(); ctx.arc(cx, cy, R, a0, ang(Math.min(score, 1))); ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.strokeStyle = '#e6edf6'; ctx.lineWidth = 2;
  const ta = ang(thr);
  ctx.beginPath();
  ctx.moveTo(cx + Math.cos(ta) * (R - 12), cy + Math.sin(ta) * (R - 12));
  ctx.lineTo(cx + Math.cos(ta) * (R + 12), cy + Math.sin(ta) * (R + 12));
  ctx.stroke();
}
function fmtClock(s) {
  const p = n => String(n).padStart(2, '0');
  return `${p(Math.floor(s / 3600))}:${p(Math.floor(s / 60) % 60)}:${p(s % 60)}`;
}
function classify(layers, ch, rising) {
  if (layers.includes('NAN')) return ['Sensor dropout', 'signal lost — connector or link'];
  if (layers.includes('STUCK')) return ['Sensor freeze', 'value stuck — probe fouling suspected'];
  if (layers.includes('SPIKE')) return ['Transient spikes', 'EMI or contact bounce on the line'];
  if (layers.includes('Z') && ch === 5 && rising) return ['Gas leak suspected', 'fast rise on gas channel — not a fault, a real event'];
  if (layers.includes('Z') && ch === 5) return ['Sensor drift', 'slow bias growth on gas channel'];
  if (layers.includes('Z')) return ['Level excursion', 'channel far outside its nominal envelope'];
  return ['Anomalous pattern', 'isolation forest flagged an unusual combination'];
}

/* ---------------- main loop ---------------- */
let detector, model, thr = 0.5;
let incident = null, gasPrev = 0.04;
const logEl = document.getElementById('eventLog');

function pushLog(entry, open) {
  if (logEl.querySelector('.log-empty')) logEl.innerHTML = '';
  const div = document.createElement('div');
  div.className = 'entry' + (open ? '' : ' closed');
  div.innerHTML = `<div class="et">${entry.t0}${entry.t1 ? ' → ' + entry.t1 : ' → …'} · ${entry.chName}</div>
    <div class="el">${entry.label}</div><div class="ed">${entry.detail}</div>
    <div class="tags">${entry.layers.map(l => `<span class="tag">${l}</span>`).join('')}
    <span class="tag">peak ${entry.peak.toFixed(3)}</span>${entry.dur ? `<span class="tag">${entry.dur}s</span>` : ''}</div>`;
  logEl.prepend(div);
  while (logEl.children.length > 12) logEl.lastChild.remove();
}

async function init() {
  buildCards();
  document.querySelectorAll('#faultBtns button').forEach(b =>
    b.addEventListener('click', () => armFault(b.dataset.fault)));
  model = await (await fetch('model.json')).json();
  detector = new Detector(model);
  thr = model.meta.threshold;
  document.getElementById('thrLabel').textContent = thr.toFixed(3);
  const metrics = await (await fetch('metrics.json')).json().catch(() => null);
  if (metrics) renderMetrics(metrics);
  setInterval(tick, 200);
}

function tick() {
  const raw = tickSim();
  const r = detector.push(raw);
  document.getElementById('clock').textContent = fmtClock(sim.t);
  document.getElementById('scoreNum').textContent = Math.round(r.score * 100);
  drawGauge(r.ewma, thr);

  const chips = document.getElementById('layerChips');
  chips.innerHTML = r.layers.map(l =>
    `<span class="lchip${l === 'ML' ? ' ml' : ''}">${l}</span>`).join('');

  const pill = document.getElementById('statusPill');
  const ptxt = document.getElementById('statusText');
  if (r.calibrating) { pill.className = 'pill warn'; ptxt.textContent = 'CALIBRATING'; }
  else if (r.incident) { pill.className = 'pill bad'; ptxt.textContent = 'ANOMALY'; }
  else { pill.className = 'pill ok'; ptxt.textContent = 'NOMINAL'; }

  const rising = Number.isFinite(raw[5]) && raw[5] > gasPrev + 0.002;
  gasPrev = Number.isFinite(raw[5]) ? raw[5] : gasPrev;

  CHANNELS.forEach((ch, i) => {
    const h = histories[i];
    h.push(raw[i]);
    if (h.length > HIST) h.shift();
    const c = cards[i];
    const lost = !Number.isFinite(raw[i]);
    c.vv.textContent = lost ? 'LOST' : ch.fmt(raw[i]);
    c.st.textContent = lost ? 'SIGNAL LOST' : 'LINK OK';
    c.st.className = 'chan-state' + (lost ? ' lost' : '');
    c.el.classList.toggle('alarm', r.incident && r.worstCh === i);
    drawSpark(i, r.incident && r.worstCh === i);
  });

  if (r.incident && !incident) {
    const chName = r.worstCh >= 0 ? CHANNELS[r.worstCh].label : '—';
    const [label, detail] = classify(r.layers, r.worstCh, rising);
    incident = { t0: fmtClock(sim.t), chName, label, detail, layers: [...r.layers], peak: r.score };
    pushLog({ ...incident, t1: null }, true);
  } else if (incident) {
    incident.peak = Math.max(incident.peak, r.score);
    for (const l of r.layers) if (!incident.layers.includes(l)) incident.layers.push(l);
    if (!r.incident) {
      incident.t1 = fmtClock(sim.t);
      incident.dur = null;
      const [label, detail] = classify(incident.layers, r.worstCh, rising);
      incident.label = label; incident.detail = detail;
      pushLog(incident, false);
      incident = null;
    }
  }
}

function renderMetrics(m) {
  const o = m.overall;
  document.getElementById('metrics').innerHTML = `
    <div class="metric"><div class="mv">${o.precision.toFixed(3)}</div><div class="mk">PRECISION</div></div>
    <div class="metric"><div class="mv">${o.recall.toFixed(3)}</div><div class="mk">RECALL</div></div>
    <div class="metric"><div class="mv">${o.f1.toFixed(3)}</div><div class="mk">F1</div></div>
    <div class="metric"><div class="mv">${o.roc_auc.toFixed(3)}</div><div class="mk">ROC-AUC</div></div>`;
  document.getElementById('perFault').innerHTML = Object.entries(m.per_fault).map(([name, p]) => `
    <div class="pf"><div class="pn">${name}</div>
      <div class="pr">recall ${(p.recall * 100).toFixed(0)}%</div>
      <div class="pd">detected in ${p.detect_delay_s === null ? '—' : p.detect_delay_s + 's'} · peak ${p.peak_score.toFixed(3)}</div>
      <div class="pl">via ${p.first_layer || '—'}</div></div>`).join('');
}

init();
