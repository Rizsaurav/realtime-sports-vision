// Sideline viewer: receives one binary websocket message per frame
// (4-byte little-endian JSON length, JSON state, JPEG preview) and draws the
// broadcast with boxes next to a top-down minimap.

const $ = (id) => document.getElementById(id);
const SRC_W = 1920, SRC_H = 1080;              // pipeline working size
const PITCH = { L: 105, W: 68, pad: 4 };       // metres; SoccerNet origin at the centre spot
const FALLBACK = { left: "#e4572e", right: "#3e8ed0" };
const DELAY_LABEL = (f) => (f === 0 ? "LIVE" : `${f / 25} S`);

const feed = $("feed"), fctx = feed.getContext("2d");
const pitch = $("pitch"), pctx = pitch.getContext("2d");
let ws = null, delays = [0], delay = 0, colors = { ...FALLBACK }, running = false;
const trails = new Map();                       // id -> [[x, y], ...]
const rosters = { left: new Map(), right: new Map() };   // jersey -> last frame seen
let lastFrame = -1;

// ---------------------------------------------------------------- controls
function buildDelays() {
  const g = $("delayGroup");
  g.innerHTML = "";
  for (const d of delays) {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(d === delay));
    b.textContent = DELAY_LABEL(d);
    b.title = d === 0 ? "Frames as they happen" : `Decisions may use ${d / 25} s of evidence after each frame`;
    b.onclick = () => setDelay(d);
    g.appendChild(b);
  }
}

function setDelay(d) {
  delay = d;
  try { localStorage.setItem("sideline.delay", String(d)); } catch (e) {}
  for (const b of $("delayGroup").children) b.setAttribute("aria-checked", String(b.textContent === DELAY_LABEL(d)));
  $("delayMeta").textContent = d === 0 ? "live" : `${d / 25} s delay`;
  trails.clear();
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ delay: d }));
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = !msg;
  clearTimeout(toast.timer);
  if (msg) toast.timer = setTimeout(() => (t.hidden = true), 6000);
}

async function loadSources() {
  try {
    const list = await (await fetch("/api/sources")).json();
    $("sourceList").innerHTML = list.map((s) => `<option value="${s.path}">${s.label}</option>`).join("");
    if (!$("sourceInput").value && list.length) $("sourceInput").value = list[0].path;
  } catch (e) {}
}

$("sourceForm").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const source = $("sourceInput").value.trim();
  if (!source) return;
  $("startBtn").disabled = true;
  rosters.left.clear(); rosters.right.clear(); trails.clear(); lastFrame = -1;
  try {
    const r = await fetch("/api/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, loop: $("loop").checked }),
    });
    const j = await r.json();
    if (!r.ok) toast(j.error || "Could not start");
  } catch (e) { toast("Server not reachable"); }
  $("startBtn").disabled = false;
});
$("stopBtn").onclick = () => fetch("/api/stop", { method: "POST" });

// ---------------------------------------------------------------- status
async function poll() {
  try {
    const s = await (await fetch("/api/status")).json();
    const air = $("onair");
    running = s.running;
    air.className = "onair" + (s.loading ? " loading" : s.running ? " live" : "");
    air.textContent = s.loading ? "LOADING MODELS" : s.running ? "ON AIR" : "OFF AIR";
    if (JSON.stringify(s.delays) !== JSON.stringify(delays)) {
      delays = s.delays;
      if (!delays.includes(delay)) delay = delays[0];
      buildDelays();
      setDelay(delay);
    }
    if (s.error) toast(s.error);
    if (s.stats && !s.running) showStats(s.stats);
  } catch (e) {
    $("onair").className = "onair"; $("onair").textContent = "NO SERVER";
  }
  setTimeout(poll, 1000);
}

function showStats(st) {
  $("fps").textContent = st.fps.toFixed(1);
  $("lat").textContent = Math.round(st.latency_ms_p50);
  $("dropped").textContent = st.dropped;
  $("behind").textContent = (delay / 25 + st.latency_ms_p50 / 1000).toFixed(1);
  $("calib").textContent = st.calibrated ? `calibrated · ${st.keyframes} keyframes` : "uncalibrated";
  const order = ["gpu", "cpu", "jersey", "decide", "calib"];
  $("stages").innerHTML = order.filter((k) => st.stage_ms[k] != null).map((k) => {
    const v = st.stage_ms[k], budget = k === "calib" ? 400 : 40;
    return `<div class="stage-row"><span>${k}</span><span class="track"><span class="fill${v > budget ? " hot" : ""}"
      style="width:${Math.min(100, (100 * v) / budget)}%"></span></span><span class="v">${v.toFixed(1)} ms</span></div>`;
  }).join("");
}

// ---------------------------------------------------------------- websocket
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => ws.send(JSON.stringify({ delay }));
  ws.onclose = () => setTimeout(connect, 1000);
  ws.onmessage = async (ev) => {
    if (typeof ev.data === "string") {
      const m = JSON.parse(ev.data);
      if (m.event === "end") $("frameMeta").textContent += " · ended";
      return;
    }
    const n = new DataView(ev.data).getUint32(0, true);
    const state = JSON.parse(new TextDecoder().decode(new Uint8Array(ev.data, 4, n)));
    const img = await createImageBitmap(new Blob([new Uint8Array(ev.data, 4 + n)], { type: "image/jpeg" }));
    render(state, img);
  };
}

// ---------------------------------------------------------------- drawing
const sideColor = (p) => (p.role === "referee" ? getVar("--ref") : p.team ? colors[p.team] : "#9aa59f");
function getVar(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }

function textOn(hex) {
  const c = hex.replace("#", "");
  const r = parseInt(c.slice(0, 2), 16), g = parseInt(c.slice(2, 4), 16), b = parseInt(c.slice(4, 6), 16);
  return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? "#0b0f0d" : "#ffffff";
}

function render(st, img) {
  if (st.colors) colors = { left: st.colors.left || FALLBACK.left, right: st.colors.right || FALLBACK.right };
  for (const side of ["left", "right"]) {
    document.querySelectorAll(`[data-side="${side}"]`).forEach((el) => el.style.setProperty("--swatch", colors[side]));
    document.querySelectorAll(`.dot[data-side="${side}"]`).forEach((el) => (el.style.background = colors[side]));
  }
  $("feedEmpty").hidden = true;
  $("frameMeta").textContent = `frame ${st.frame} · ${(st.frame / 25).toFixed(1)} s`;
  if (st.stats) showStats(st.stats);
  drawFeed(st, img);
  drawPitch(st);
  updateRosters(st);
  lastFrame = st.frame;
}

function drawFeed(st, img) {
  if (feed.width !== img.width) { feed.width = img.width; feed.height = img.height; }
  const s = feed.width / SRC_W;
  fctx.drawImage(img, 0, 0);
  fctx.lineWidth = 2;
  const fs = Math.round(Math.max(13, 22 * s));
  fctx.font = `700 ${fs}px "IBM Plex Mono", monospace`;
  fctx.textAlign = "center";
  fctx.textBaseline = "bottom";
  for (const p of st.people) {
    const [x, y, w, h] = p.box.map((v) => v * s);
    const c = sideColor(p);
    fctx.strokeStyle = c;
    fctx.strokeRect(x, y, w, h);
    const label = p.jersey || (p.role === "goalkeeper" ? "GK" : p.role === "referee" ? "REF" : "");
    if (label) {
      const tw = fctx.measureText(label).width + 8, th = fs + 4;
      fctx.fillStyle = c;
      fctx.fillRect(x + w / 2 - tw / 2, y - th - 2, tw, th);
      fctx.fillStyle = textOn(c.startsWith("#") ? c : "#9aa59f");
      fctx.fillText(label, x + w / 2, y - 3);
    }
  }
  if (st.ball) {
    const [x, y, w, h] = st.ball.box.map((v) => v * s);
    fctx.strokeStyle = "#ffffff";
    fctx.beginPath();
    fctx.arc(x + w / 2, y + h / 2, Math.max(w, h) * 0.9 + 3, 0, Math.PI * 2);
    fctx.stroke();
  }
}

function pitchTransform() {
  const W = pitch.width, H = pitch.height;
  const k = Math.min(W / (PITCH.L + 2 * PITCH.pad), H / (PITCH.W + 2 * PITCH.pad));
  return { k, ox: W / 2, oy: H / 2 };
}

function drawPitchLines(t) {
  const { k, ox, oy } = t;
  const X = (x) => ox + x * k, Y = (y) => oy + y * k;
  const L = PITCH.L / 2, Wd = PITCH.W / 2;
  // mowing stripes, 11 bands across the length
  const band = PITCH.L / 11;
  for (let i = 0; i < 11; i++) {
    pctx.fillStyle = i % 2 ? getVar("--turf") : getVar("--turf-2");
    pctx.fillRect(X(-L + i * band), Y(-Wd), band * k + 1, PITCH.W * k);
  }
  pctx.strokeStyle = getVar("--chalk");
  pctx.lineWidth = Math.max(1.5, k * 0.14);
  pctx.strokeRect(X(-L), Y(-Wd), PITCH.L * k, PITCH.W * k);
  pctx.beginPath(); pctx.moveTo(X(0), Y(-Wd)); pctx.lineTo(X(0), Y(Wd)); pctx.stroke();
  pctx.beginPath(); pctx.arc(X(0), Y(0), 9.15 * k, 0, Math.PI * 2); pctx.stroke();
  pctx.fillStyle = getVar("--chalk");
  pctx.beginPath(); pctx.arc(X(0), Y(0), k * 0.35, 0, Math.PI * 2); pctx.fill();
  for (const s of [-1, 1]) {
    const gx = s * L;
    pctx.strokeRect(Math.min(X(gx), X(gx - s * 16.5)), Y(-20.16), 16.5 * k, 40.32 * k);
    pctx.strokeRect(Math.min(X(gx), X(gx - s * 5.5)), Y(-9.16), 5.5 * k, 18.32 * k);
    pctx.strokeRect(Math.min(X(gx), X(gx + s * 2)), Y(-3.66), 2 * k, 7.32 * k);
    pctx.beginPath(); pctx.arc(X(gx - s * 11), Y(0), k * 0.3, 0, Math.PI * 2); pctx.fill();
    const a = Math.acos(5.5 / 9.15);
    pctx.beginPath();
    if (s < 0) pctx.arc(X(gx + 11), Y(0), 9.15 * k, -a, a);
    else pctx.arc(X(gx - 11), Y(0), 9.15 * k, Math.PI - a, Math.PI + a);
    pctx.stroke();
  }
}

function drawPitch(st) {
  const t = pitchTransform();
  const { k, ox, oy } = t;
  pctx.fillStyle = "#123826";
  pctx.fillRect(0, 0, pitch.width, pitch.height);
  drawPitchLines(t);
  const X = (x) => ox + x * k, Y = (y) => oy + y * k;
  const seen = new Set();
  for (const p of st.people) {
    seen.add(p.id);
    const tr = trails.get(p.id) || [];
    tr.push(p.xy);
    if (tr.length > 20) tr.shift();
    trails.set(p.id, tr);
  }
  for (const id of [...trails.keys()]) if (!seen.has(id)) trails.delete(id);
  // trails
  for (const p of st.people) {
    const tr = trails.get(p.id);
    if (!tr || tr.length < 2) continue;
    pctx.strokeStyle = sideColor(p) + "66";
    pctx.lineWidth = Math.max(1, k * 0.25);
    pctx.beginPath();
    tr.forEach(([x, y], i) => (i ? pctx.lineTo(X(x), Y(y)) : pctx.moveTo(X(x), Y(y))));
    pctx.stroke();
  }
  const r = Math.max(9, k * 1.9);
  pctx.textAlign = "center";
  pctx.textBaseline = "middle";
  pctx.font = `700 ${Math.round(r * 1.1)}px "IBM Plex Mono", monospace`;
  for (const p of st.people) {
    const [x, y] = p.xy;
    if (Math.abs(x) > 60 || Math.abs(y) > 42) continue;
    const c = sideColor(p);
    pctx.fillStyle = c;
    pctx.strokeStyle = p.role === "goalkeeper" ? "#ffffff" : "rgba(0,0,0,0.55)";
    pctx.lineWidth = p.role === "goalkeeper" ? 2.5 : 1.5;
    pctx.beginPath(); pctx.arc(X(x), Y(y), r, 0, Math.PI * 2); pctx.fill(); pctx.stroke();
    if (p.jersey) {
      pctx.fillStyle = textOn(c.startsWith("#") ? c : "#9aa59f");
      pctx.fillText(p.jersey, X(x), Y(y) + 0.5);
    }
  }
  if (st.ball && st.ball.xy) {
    const [x, y] = st.ball.xy;
    pctx.fillStyle = "#ffffff";
    pctx.strokeStyle = "#000000";
    pctx.lineWidth = 1.5;
    pctx.beginPath(); pctx.arc(X(x), Y(y), r * 0.5, 0, Math.PI * 2); pctx.fill(); pctx.stroke();
  }
}

function updateRosters(st) {
  const now = { left: new Set(), right: new Set() };
  for (const p of st.people) {
    if (p.jersey && p.team && p.role !== "referee") {
      rosters[p.team].set(p.jersey, st.frame);
      now[p.team].add(p.jersey);
    }
  }
  for (const side of ["left", "right"]) {
    const items = [...rosters[side].keys()].sort((a, b) => Number(a) - Number(b));
    $(`${side}Roster`).innerHTML = items.map((j) => `<li class="${now[side].has(j) ? "on" : "off"}">${j}</li>`).join("");
  }
  $("onPitch").textContent = `${st.people.length} on screen`;
}

// ---------------------------------------------------------------- boot
try { const d = Number(localStorage.getItem("sideline.delay")); if (!Number.isNaN(d)) delay = d; } catch (e) {}
buildDelays();
drawPitch({ people: [], ball: null });
loadSources();
poll();
connect();
