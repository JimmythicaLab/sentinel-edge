// ===================== Tab switching =====================
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelector(`.page[data-page="${tab}"]`).classList.add('active');
  });
});

// ===================== Camera switching =====================
const mainVideo = document.getElementById('mainVideo');
const mainCamTitle = document.getElementById('mainCamTitle');

// Each entry can optionally point at a tracks JSON produced by
// scripts/export_webdemo_tracks.py — if present, the live overlay
// animates real detections for that camera; otherwise the overlay
// stays empty (video is still shown).
const CAM_INFO = {
  cam01: { src: 'assets/videos/cam01_small.mp4', title: 'CAM-01', zone: 'Operations Room · Server Rack',    tracks: null },
  cam07: { src: 'assets/videos/cam07.mp4',       title: 'CAM-07', zone: 'Warehouse Floor · Inventory Area', tracks: null },
  cam23: { src: 'assets/videos/cam23.mp4',       title: 'CAM-23', zone: 'Entrance Gate · Perimeter',        tracks: null },
  cam42: { src: 'assets/videos/cam42.mp4',       title: 'CAM-42', zone: 'Loading Bay 3 · Restricted',       tracks: 'CAM42_TRACKS' },
};

document.querySelectorAll('.cam-card').forEach(card => {
  card.addEventListener('click', () => {
    const cam = card.dataset.cam;
    document.querySelectorAll('.cam-card').forEach(c => c.classList.remove('active'));
    card.classList.add('active');
    const info = CAM_INFO[cam];
    if (!info) return;
    switchCamera(cam, info);
  });
});

function switchCamera(cam, info) {
  mainVideo.src = info.src;
  mainVideo.play();
  mainCamTitle.innerHTML = `${info.title} <span>· ${info.zone}</span>`;
  loadTracks(info.tracks);
  resetRuleState();
  document.getElementById('alertRibbon').classList.remove('show');
}

// ===================== Live clock =====================
function pad(n) { return String(n).padStart(2, '0'); }
function nowStr() {
  const n = new Date();
  return `${pad(n.getHours())}:${pad(n.getMinutes())}:${pad(n.getSeconds())}`;
}
function updateClock() {
  const now = new Date();
  const dt = `2026-10-14 ${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
  document.getElementById('feedTime').textContent = dt;
}
setInterval(updateClock, 1000);
updateClock();

// ===================== Live detection overlay =====================
const zoneLayer = document.getElementById('zoneLayer');
const boxLayer  = document.getElementById('boxLayer');
const feedStatus = document.getElementById('feedStatus');

let trackData = null;      // {fps, frames:[{t, boxes:[]}], zone:{polygon:[[x,y]...], name, rule}}
let zonePolyPx = [];       // zone polygon in viewBox 0..100 units for SVG

function loadTracks(globalKey) {
  // Clear existing overlay + any stale zone label.
  zoneLayer.innerHTML = '';
  boxLayer.innerHTML = '';
  const container = zoneLayer.parentElement;
  container.querySelectorAll('.zone-label').forEach(el => el.remove());
  trackData = null;
  zonePolyPx = [];
  if (!globalKey) return;
  const d = window[globalKey];
  if (!d) {
    console.warn('tracks data missing for', globalKey);
    return;
  }
  trackData = d;
  const pts = d.zone.polygon.map(([x,y]) => `${(x*100).toFixed(2)},${(y*100).toFixed(2)}`).join(' ');
  zoneLayer.innerHTML = `<polygon points="${pts}"></polygon>`;
  zonePolyPx = d.zone.polygon;
  const [zx, zy] = d.zone.polygon[0];
  const label = document.createElement('div');
  label.className = 'zone-label';
  label.textContent = `ZONE · ${d.zone.name.toUpperCase()} · ${d.zone.rule}`;
  label.style.left = `calc(${(zx*100).toFixed(1)}% + 8px)`;
  label.style.top  = `calc(${(zy*100).toFixed(1)}% + 8px)`;
  container.appendChild(label);
  feedStatus.textContent = `${d.frames.length} frames @ ${d.fps}fps · track replay · ${d.zone.rule}`;
}

// ===================== JS rule engine (mirrors src/rule_engine.py) =====================
const RULE = {
  name:           'restricted_access',
  dwellFrames:    12,     // ~0.5s @ 24fps
  cooldownFrames: 240,    // 10s @ 24fps
};
let dwellCounters = new Map();  // track_id -> consecutive frames in zone
let lastFiredFrame = new Map(); // track_id -> frame_idx of last ALERT

function resetRuleState() {
  dwellCounters.clear();
  lastFiredFrame.clear();
}

function pointInPolygon(px, py, poly) {
  // ray-casting
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    const intersect = ((yi > py) !== (yj > py)) &&
                      (px < (xj - xi) * (py - yi) / ((yj - yi) || 1e-9) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function evaluateFrame(frameIdx, boxes) {
  const violations = [];
  for (const b of boxes) {
    if (b.cls !== 'person') continue;
    const cx = b.x + b.w / 2;
    const cy = b.y + b.h / 2;  // use centroid for rule check
    const inZone = zonePolyPx.length ? pointInPolygon(cx, cy, zonePolyPx) : false;
    b._inZone = inZone;
    if (!inZone) {
      dwellCounters.delete(b.id);
      continue;
    }
    const dwell = (dwellCounters.get(b.id) || 0) + 1;
    dwellCounters.set(b.id, dwell);
    b._dwell = dwell;
    if (dwell < RULE.dwellFrames) continue;
    const last = lastFiredFrame.get(b.id) ?? -Infinity;
    if (frameIdx - last < RULE.cooldownFrames) {
      b._violating = true;  // still mark red while in cooldown
      continue;
    }
    lastFiredFrame.set(b.id, frameIdx);
    b._violating = true;
    violations.push({ track: b.id, dwell });
  }
  return violations;
}

// ===================== Overlay animation loop =====================
function renderFrame() {
  if (trackData && !mainVideo.paused && !mainVideo.ended) {
    const idx = Math.min(
      trackData.frames.length - 1,
      Math.max(0, Math.round(mainVideo.currentTime * trackData.fps))
    );
    const boxes = trackData.frames[idx].boxes;
    const violations = evaluateFrame(idx, boxes);

    // Reconcile DOM: keep one .bbox div per visible track id.
    const seen = new Set();
    for (const b of boxes) {
      if (b.cls !== 'person') continue;
      const key = `b${b.id}`;
      seen.add(key);
      let el = boxLayer.querySelector(`[data-id="${key}"]`);
      if (!el) {
        el = document.createElement('div');
        el.className = 'bbox';
        el.dataset.id = key;
        el.innerHTML = '<div class="bbox-label"></div>';
        boxLayer.appendChild(el);
      }
      el.style.left   = `${(b.x * 100).toFixed(2)}%`;
      el.style.top    = `${(b.y * 100).toFixed(2)}%`;
      el.style.width  = `${(b.w * 100).toFixed(2)}%`;
      el.style.height = `${(b.h * 100).toFixed(2)}%`;
      const tid = b.id === -1 ? '?' : b.id;
      const dwellLbl = (b._violating || b._dwell) ? ` · dwell ${b._dwell}` : '';
      el.querySelector('.bbox-label').textContent =
        `${b.cls} #${tid} · ${b.conf}${dwellLbl}`;
      el.classList.toggle('in-zone',   !!b._inZone && !b._violating);
      el.classList.toggle('violating', !!b._violating);
    }
    // Remove dead boxes.
    boxLayer.querySelectorAll('.bbox').forEach(el => {
      if (!seen.has(el.dataset.id)) el.remove();
    });

    // Fire ALERTs when the rule engine emits violations.
    for (const v of violations) emitAlert(v, idx);
  }
  requestAnimationFrame(renderFrame);
}
requestAnimationFrame(renderFrame);

// ===================== Alert emission =====================
const alertRibbon  = document.getElementById('alertRibbon');
const alertKindEl  = document.getElementById('alertKind');
const alertTextEl  = document.getElementById('alertText');
const linePopup    = document.getElementById('linePopup');

function emitAlert(v, frameIdx) {
  const kind = RULE.name.toUpperCase();
  const text = `Person #${v.track} in restricted_zone for ${v.dwell} frames`;

  alertKindEl.textContent = `ALERT · ${kind} · CAM-42`;
  alertTextEl.textContent = text;
  alertRibbon.classList.add('show');
  clearTimeout(emitAlert._hideTimer);
  emitAlert._hideTimer = setTimeout(() => alertRibbon.classList.remove('show'), 6000);

  const t = nowStr();
  prependEvent(makeEvent(
    'ALERT', t,
    `<strong>CAM-42</strong> · ${kind}: person #${v.track} dwell ${v.dwell} frames`,
    'alert'
  ));
  // Downstream dispatch simulation (LINE popup + PUSH event).
  setTimeout(() => {
    prependEvent(makeEvent(
      'REPORT', nowStr(),
      'On-device incident summary generated · bilingual <em>(click to view)</em>',
      'report'
    ));
  }, 1200);
  setTimeout(() => {
    prependEvent(makeEvent(
      'PUSH', nowStr(),
      'Dispatched: LINE Notify · MQTT bus · on-site strobe · local JSONL log',
      'push'
    ));
    linePopup.classList.add('show');
    clearTimeout(emitAlert._lineHide);
    emitAlert._lineHide = setTimeout(() => linePopup.classList.remove('show'), 8000);
  }, 2400);
}

// ===================== Event stream helpers =====================
const eventsList = document.getElementById('eventsList');

function makeEvent(kind, time, text, type) {
  const div = document.createElement('div');
  div.className = `event ${type} new`;
  div.innerHTML = `
    <div class="ev-head">
      <span class="ev-time">${time}</span>
      <span class="ev-tag">${kind}</span>
    </div>
    <div class="ev-body">${text}</div>
  `;
  div.addEventListener('click', () => {
    if (type === 'alert' || type === 'report') openModal();
  });
  return div;
}

function prependEvent(div) {
  eventsList.insertBefore(div, eventsList.firstChild);
  Array.from(eventsList.children).forEach((el, i) => {
    if (i > 4) el.classList.add('muted');
  });
  // Cap list length so demo stays snappy.
  while (eventsList.children.length > 20) {
    eventsList.removeChild(eventsList.lastChild);
  }
}

// ===================== Trigger demo (manual replay from t=0) =====================
const triggerBtn = document.getElementById('trigger');
triggerBtn.classList.add('pulsing');
triggerBtn.addEventListener('click', () => {
  // Rewind CAM-42 so the reviewer can watch the rule fire from the top.
  document.querySelectorAll('.cam-card').forEach(c => c.classList.remove('active'));
  document.querySelector('.cam-card[data-cam="cam42"]').classList.add('active');
  switchCamera('cam42', CAM_INFO.cam42);
  mainVideo.currentTime = 0;
  mainVideo.play();
});

// ===================== Modal =====================
const modalOverlay = document.getElementById('modalOverlay');
const modalClose   = document.getElementById('modalClose');
function openModal()  { modalOverlay.classList.add('show'); }
function closeModal() { modalOverlay.classList.remove('show'); }
modalClose.addEventListener('click', closeModal);
modalOverlay.addEventListener('click', (e) => { if (e.target === modalOverlay) closeModal(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

document.querySelectorAll('#historyTbody tr').forEach(tr => {
  tr.style.cursor = 'pointer';
  tr.addEventListener('click', openModal);
});
document.querySelectorAll('.rule-item').forEach(item => {
  item.addEventListener('click', () => {
    document.querySelectorAll('.rule-item').forEach(i => i.classList.remove('active'));
    item.classList.add('active');
  });
});

// ===================== Boot =====================
// Default to CAM-42 so the live overlay is visible on load.
loadTracks(CAM_INFO.cam42.tracks);
