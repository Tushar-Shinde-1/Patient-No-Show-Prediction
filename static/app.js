/* ═══════════════════════════════════════════════════════════════════
   Patient No-Show Predictive System — Client-Side Logic
   Chart.js (loaded via CDN) + AJAX Prediction
   ═══════════════════════════════════════════════════════════════════ */

// ── Colour palette for charts ───────────────────────────────────────
const PALETTE = {
    blue:    'rgba(67, 97, 238, %a)',
    purple:  'rgba(114, 9, 183, %a)',
    green:   'rgba(16, 185, 129, %a)',
    orange:  'rgba(245, 158, 11, %a)',
};

function solid(c)  { return c.replace('%a', '1');   }
function alpha(c)  { return c.replace('%a', '0.18'); }

// ── Tabs ────────────────────────────────────────────────────────────
function initTabs() {
    const btns = document.querySelectorAll('.tab-btn');
    const panes = document.querySelectorAll('.tab-content');
    if (!btns.length) return;

    btns.forEach(btn => {
        btn.addEventListener('click', () => {
            btns.forEach(b => b.classList.remove('active'));
            panes.forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            const target = document.getElementById(btn.dataset.tab);
            if (target) target.classList.add('active');
        });
    });
}

// ── Dashboard Charts ────────────────────────────────────────────────
async function loadDashboard() {
    try {
        const res = await fetch('/api/metrics');
        if (!res.ok) return;
        const data = await res.json();

        renderBestMetrics(data);
        renderModelComparison(data);
        renderGeneralization(data);
        renderFeatureImportance(data);
    } catch (err) {
        console.error('Failed to load metrics:', err);
    }
}

function renderBestMetrics(data) {
    const best = data.best_model;
    const m = data.models[best];
    if (!m) return;

    const setVal = (id, v) => {
        const el = document.getElementById(id);
        if (el) el.textContent = (v * 100).toFixed(1) + '%';
    };
    setVal('best-accuracy',  m.accuracy);
    setVal('best-precision', m.precision);
    setVal('best-recall',    m.recall);
    setVal('best-f1',        m.f1_score);

    const el = document.getElementById('best-model-name');
    if (el) el.textContent = best;
}

function renderModelComparison(data) {
    const ctx = document.getElementById('chartComparison');
    if (!ctx) return;

    const names = Object.keys(data.models);
    const colors = [PALETTE.blue, PALETTE.purple, PALETTE.green, PALETTE.orange];

    const datasets = ['accuracy', 'precision', 'recall', 'f1_score'].map((metric, i) => ({
        label: metric === 'f1_score' ? 'F1-Score' : metric.charAt(0).toUpperCase() + metric.slice(1),
        data: names.map(n => +(data.models[n][metric] * 100).toFixed(1)),
        backgroundColor: alpha(colors[i]),
        borderColor: solid(colors[i]),
        borderWidth: 2,
        borderRadius: 6,
    }));

    new Chart(ctx, {
        type: 'bar',
        data: { labels: names, datasets },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top', labels: { font: { family: "'Inter'" } } },
                title:  { display: false },
            },
            scales: {
                y: { beginAtZero: true, max: 100,
                     ticks: { callback: v => v + '%', font: { family: "'Inter'" } },
                     grid: { color: 'rgba(0,0,0,.04)' } },
                x: { ticks: { font: { family: "'Inter'" } },
                     grid: { display: false } },
            },
        },
    });
}

function renderGeneralization(data) {
    const tbody = document.getElementById('gen-table-body');
    if (!tbody) return;

    tbody.innerHTML = '';
    for (const [name, m] of Object.entries(data.models)) {
        const trainAcc = m.train_accuracy !== undefined ? (m.train_accuracy * 100).toFixed(1) + '%' : '—';
        const testAcc  = (m.accuracy * 100).toFixed(1) + '%';
        const gap      = m.train_accuracy !== undefined
            ? ((m.train_accuracy - m.accuracy) * 100).toFixed(1) + ' pp'
            : '—';
        const verdict  = m.train_accuracy !== undefined
            ? (m.train_accuracy - m.accuracy > 5 ? '⚠️ Possible overfit' : '✅ Good')
            : '—';

        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${name}</td><td>${trainAcc}</td><td>${testAcc}</td><td>${gap}</td><td>${verdict}</td>`;
        tbody.appendChild(tr);
    }
}

function renderFeatureImportance(data) {
    const ctx = document.getElementById('chartFeatures');
    if (!ctx || !data.feature_importances) return;

    const feats = data.feature_importances.slice(0, 12);
    const labels = feats.map(f => f.Feature);
    const values = feats.map(f => +(f.Importance * 100).toFixed(2));

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Importance (%)',
                data: values,
                backgroundColor: labels.map((_, i) => {
                    const t = i / labels.length;
                    return `rgba(${Math.round(67 + t * 47)}, ${Math.round(97 - t * 88)}, ${Math.round(238 - t * 55)}, 0.78)`;
                }),
                borderRadius: 6,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { callback: v => v + '%', font: { family: "'Inter'" } },
                     grid: { color: 'rgba(0,0,0,.04)' } },
                y: { ticks: { font: { family: "'Inter'", size: 12 } },
                     grid: { display: false } },
            },
        },
    });
}

// ── Prediction Form ─────────────────────────────────────────────────
function initPredictionForm() {
    const form = document.getElementById('predict-form');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const btn = form.querySelector('button[type="submit"]');
        const original = btn.innerHTML;
        btn.innerHTML = '<span class="spinner"></span> Analysing…';
        btn.disabled = true;

        const dayMap = { Monday: 0, Tuesday: 1, Wednesday: 2, Thursday: 3, Friday: 4, Saturday: 5, Sunday: 6 };

        const payload = {
            Age:                  +form.age.value,
            Gender:              form.gender.value,
            Neighbourhood:       form.neighbourhood.value,
            Scholarship:         form.scholarship.value === 'Yes' ? 1 : 0,
            Hipertension:        form.hipertension.value === 'Yes' ? 1 : 0,
            Diabetes:            form.diabetes.value === 'Yes' ? 1 : 0,
            Alcoholism:          form.alcoholism.value === 'Yes' ? 1 : 0,
            Handcap:             +form.handcap.value,
            SMS_received:        form.sms_received.value === 'Yes' ? 1 : 0,
            WaitingTime:         +form.waiting_time.value,
            AppointmentDayOfWeek: dayMap[form.day_of_week.value] ?? 0,
            AppointmentMonth:    +form.month.value,
        };

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const result = await res.json();

            if (result.error) {
                alert('Error: ' + result.error);
                return;
            }

            showResult(result);
        } catch (err) {
            alert('Network error: ' + err.message);
        } finally {
            btn.innerHTML = original;
            btn.disabled = false;
        }
    });
}

function showResult(r) {
    const panel = document.getElementById('result-panel');
    if (!panel) return;

    document.getElementById('res-probability').textContent = (r.probability * 100).toFixed(1) + '%';
    document.getElementById('res-label').textContent = r.label;

    const riskEl = document.getElementById('res-risk');
    riskEl.textContent = r.risk + ' Risk';

    const riskCard = riskEl.closest('.result-card');
    riskCard.className = 'result-card risk-' + r.risk.toLowerCase();

    const action = document.getElementById('res-action');
    if (r.prediction === 1) {
        action.className = 'action-box noshow';
        action.textContent = '🚨 Action: Consider contacting the patient for an extra reminder or double-booking this slot.';
    } else {
        action.className = 'action-box show';
        action.textContent = '✅ Action: Standard protocol — patient is expected to attend.';
    }

    panel.style.display = 'block';
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Boot ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    loadDashboard();
    initPredictionForm();
});
