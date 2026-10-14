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

const CAM_INFO = {
  cam01: { src: 'assets/videos/cam01_small.mp4', title: 'CAM-01', zone: 'Operations Room · Server Rack' },
  cam07: { src: 'assets/videos/cam07.mp4',       title: 'CAM-07', zone: 'Warehouse Floor · Inventory Area' },
  cam23: { src: 'assets/videos/cam23.mp4',       title: 'CAM-23', zone: 'Entrance Gate · Perimeter' },
  cam42: { src: 'assets/videos/cam42.mp4',       title: 'CAM-42', zone: 'Loading Bay 3 · Helmet Mandatory' },
};

document.querySelectorAll('.cam-card').forEach(card => {
  card.addEventListener('click', () => {
    const cam = card.dataset.cam;
    document.querySelectorAll('.cam-card').forEach(c => c.classList.remove('active'));
    card.classList.add('active');

    const info = CAM_INFO[cam];
    if (!info) return;
    // Reset detection overlay when switching cameras
    document.getElementById('detectionOverlay').classList.remove('show');
    document.getElementById('alertRibbon').classList.remove('show');

    mainVideo.src = info.src;
    mainVideo.play();
    mainCamTitle.innerHTML = `${info.title} <span>· ${info.zone}</span>`;
  });
});

// ===================== Live clock =====================
function pad(n) { return String(n).padStart(2, '0'); }
function updateClock() {
  const now = new Date();
  // Use a static-ish date to maintain "staging" feel
  const dt = `2026-10-14 ${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
  document.getElementById('feedTime').textContent = dt;
}
setInterval(updateClock, 1000);
updateClock();

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
  // Soften older events
  Array.from(eventsList.children).forEach((el, i) => {
    if (i > 4) el.classList.add('muted');
  });
}

// ===================== Trigger demo =====================
const triggerBtn = document.getElementById('trigger');
const detOverlay = document.getElementById('detectionOverlay');
const alertRibbon = document.getElementById('alertRibbon');
const linePopup = document.getElementById('linePopup');

triggerBtn.classList.add('pulsing');

let demoRunning = false;
function runDemo() {
  if (demoRunning) return;
  demoRunning = true;
  triggerBtn.disabled = true;
  triggerBtn.classList.remove('pulsing');
  triggerBtn.textContent = 'Running simulation…';

  // Make sure we're on CAM-42
  document.querySelectorAll('.cam-card').forEach(c => c.classList.remove('active'));
  document.querySelector('.cam-card[data-cam="cam42"]').classList.add('active');
  mainVideo.src = CAM_INFO.cam42.src;
  mainVideo.play();
  mainCamTitle.innerHTML = `CAM-42 <span>· Loading Bay 3 · Helmet Mandatory</span>`;

  const now = new Date();
  const t0 = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;

  // Reset overlays
  detOverlay.classList.remove('show');
  alertRibbon.classList.remove('show');
  linePopup.classList.remove('show');

  // t+1s: show detection overlay
  setTimeout(() => {
    detOverlay.classList.add('show');
  }, 1000);

  // t+1.5s: show alert ribbon
  setTimeout(() => {
    alertRibbon.classList.add('show');
  }, 1500);

  // t+2s: ALERT event
  setTimeout(() => {
    prependEvent(makeEvent(
      'ALERT', t0,
      '<strong>CAM-42</strong> · Loading Bay 3: unprotected worker detected (0.94 confidence)',
      'alert'
    ));
  }, 2000);

  // t+3.5s: REPORT event
  setTimeout(() => {
    const n = new Date();
    const t = `${pad(n.getHours())}:${pad(n.getMinutes())}:${pad(n.getSeconds())}`;
    prependEvent(makeEvent(
      'REPORT', t,
      'LLM incident summary generated · bilingual · ready for audit log <em>(click to view)</em>',
      'report'
    ));
  }, 3500);

  // t+4.5s: PUSH event + LINE popup
  setTimeout(() => {
    const n = new Date();
    const t = `${pad(n.getHours())}:${pad(n.getMinutes())}:${pad(n.getSeconds())}`;
    prependEvent(makeEvent(
      'PUSH', t,
      'Dispatched: LINE Notify · MQTT bus · on-site strobe · local JSONL log',
      'push'
    ));
    linePopup.classList.add('show');
  }, 4500);

  // t+13s: hide LINE popup
  setTimeout(() => {
    linePopup.classList.remove('show');
  }, 13000);

  // t+16s: reset trigger
  setTimeout(() => {
    triggerBtn.disabled = false;
    triggerBtn.classList.add('pulsing');
    triggerBtn.textContent = '▶ Simulate Violation Event';
    demoRunning = false;
  }, 16000);
}

triggerBtn.addEventListener('click', runDemo);

// ===================== Modal =====================
const modalOverlay = document.getElementById('modalOverlay');
const modalClose = document.getElementById('modalClose');

function openModal() { modalOverlay.classList.add('show'); }
function closeModal() { modalOverlay.classList.remove('show'); }

modalClose.addEventListener('click', closeModal);
modalOverlay.addEventListener('click', (e) => {
  if (e.target === modalOverlay) closeModal();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeModal();
});

// Make history table rows clickable too
document.querySelectorAll('#historyTbody tr').forEach(tr => {
  tr.style.cursor = 'pointer';
  tr.addEventListener('click', openModal);
});

// Make rule items clickable (visual only)
document.querySelectorAll('.rule-item').forEach(item => {
  item.addEventListener('click', () => {
    document.querySelectorAll('.rule-item').forEach(i => i.classList.remove('active'));
    item.classList.add('active');
  });
});
