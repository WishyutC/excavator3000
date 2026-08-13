const csrfToken = document.querySelector('meta[name="csrf-token"]').content;
let configFields = [];
let noticeTimer = null;

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function showNotice(message, isError = false) {
  const notice = document.getElementById('notice');
  notice.textContent = message;
  notice.classList.toggle('error', isError);
  notice.hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { notice.hidden = true; }, 7000);
}

async function api(path, options = {}) {
  const settings = { ...options, headers: { ...(options.headers || {}) } };
  if (settings.method && settings.method !== 'GET') settings.headers['X-CSRF-Token'] = csrfToken;
  if (settings.body) settings.headers['Content-Type'] = 'application/json';
  const response = await fetch(path, settings);
  if (response.status === 401) { window.location.href = '/login'; throw new Error('Authentication required'); }
  const payload = await response.json();
  if (!response.ok || !payload.ok) throw new Error(payload.error || 'Request failed');
  return payload;
}

function updateProcess(process) {
  const state = process.state || 'stopped';
  const dot = document.getElementById('process-dot');
  dot.className = `status-dot ${state}`;
  document.getElementById('process-label').textContent = state[0].toUpperCase() + state.slice(1);
  document.getElementById('process-detail').textContent = process.pid ? `PID ${process.pid}` : 'No managed process';
  document.getElementById('start-button').disabled = state !== 'stopped';
  document.getElementById('pause-button').disabled = state !== 'running';
  document.getElementById('resume-button').disabled = state !== 'paused';
  document.getElementById('stop-button').disabled = state === 'stopped';
}

function updateMetrics(training) {
  const latest = training.latest;
  if (!latest) return;
  const recent = training.windows['20'];
  const last100 = training.windows['100'];
  document.getElementById('metric-episode').textContent = formatNumber(latest.episode, 0);
  document.getElementById('metric-reward').textContent = formatNumber(latest.reward);
  document.getElementById('metric-average').textContent = formatNumber(recent.average_reward);
  document.getElementById('metric-success').textContent = `${formatNumber(last100.success_rate, 1)}%`;
  document.getElementById('metric-loss').textContent = formatNumber(latest.loss, 4);
  document.getElementById('metric-buffer').textContent = `Replay buffer ${formatNumber(latest.buffer_size, 0)}`;
  document.getElementById('metric-termination').textContent = `${formatNumber(last100.collision_rate, 0)}% collision · ${formatNumber(last100.timeout_rate, 0)}% timeout`;
  const trend = training.reward_trend_20;
  document.getElementById('metric-trend').textContent = trend === null ? 'Latest episode' : `${trend >= 0 ? '+' : ''}${formatNumber(trend)} vs previous 20`;
  const stale = training.csv_stale_seconds;
  document.getElementById('metric-updated').textContent = stale === null ? 'CSV unavailable' : `CSV updated ${formatNumber(stale, 0)}s ago`;

  const goal = last100.success_rate;
  const timeout = last100.timeout_rate;
  const collision = last100.collision_rate;
  const other = Math.max(0, 100 - goal - timeout - collision);
  document.getElementById('bar-goal').style.width = `${goal}%`;
  document.getElementById('bar-timeout').style.width = `${timeout}%`;
  document.getElementById('bar-collision').style.width = `${collision}%`;
  document.getElementById('bar-other').style.width = `${other}%`;
  document.getElementById('goal-rate').textContent = `${formatNumber(goal, 1)}%`;
  document.getElementById('timeout-rate').textContent = `${formatNumber(timeout, 1)}%`;
  document.getElementById('collision-rate').textContent = `${formatNumber(collision, 1)}%`;
}

function cssColor(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function drawLineChart(canvas, rows, series, options = {}) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = Number(canvas.getAttribute('height'));
  canvas.width = Math.max(1, width * ratio);
  canvas.height = height * ratio;
  const context = canvas.getContext('2d');
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  if (!rows.length) return;

  const padding = { left: 58, right: 14, top: 12, bottom: 30 };
  const plotWidth = Math.max(1, width - padding.left - padding.right);
  const plotHeight = height - padding.top - padding.bottom;
  const transformed = series.flatMap(item => rows.map(row => item.transform ? item.transform(row[item.key]) : row[item.key])).filter(Number.isFinite);
  if (!transformed.length) return;
  let minimum = Math.min(...transformed);
  let maximum = Math.max(...transformed);
  if (minimum === maximum) { minimum -= 1; maximum += 1; }
  const margin = (maximum - minimum) * 0.08;
  minimum -= margin; maximum += margin;

  context.strokeStyle = cssColor('--grid'); context.fillStyle = cssColor('--muted');
  context.font = '11px system-ui'; context.lineWidth = 1;
  for (let index = 0; index <= 4; index += 1) {
    const y = padding.top + plotHeight * index / 4;
    context.beginPath(); context.moveTo(padding.left, y); context.lineTo(width - padding.right, y); context.stroke();
    const value = maximum - (maximum - minimum) * index / 4;
    const label = options.log ? `10^${value.toFixed(1)}` : formatNumber(value, 1);
    context.fillText(label, 4, y + 4);
  }
  context.fillText(String(rows[0].episode), padding.left, height - 8);
  const lastLabel = String(rows[rows.length - 1].episode);
  context.fillText(lastLabel, width - padding.right - context.measureText(lastLabel).width, height - 8);

  for (const item of series) {
    context.strokeStyle = item.color; context.lineWidth = item.width || 1.5; context.beginPath();
    let started = false;
    rows.forEach((row, index) => {
      let value = item.transform ? item.transform(row[item.key]) : row[item.key];
      if (!Number.isFinite(value)) return;
      const x = padding.left + plotWidth * index / Math.max(1, rows.length - 1);
      const y = padding.top + plotHeight * (maximum - value) / (maximum - minimum);
      if (!started) { context.moveTo(x, y); started = true; } else context.lineTo(x, y);
    });
    context.stroke();
  }
}

function updateCharts(history) {
  drawLineChart(document.getElementById('reward-chart'), history, [
    { key: 'reward', color: cssColor('--cyan'), width: 1 },
    { key: 'rolling_reward_20', color: cssColor('--green'), width: 2.4 }
  ]);
  const lossRows = history.filter(row => Number.isFinite(row.loss) && row.loss >= 0);
  drawLineChart(document.getElementById('loss-chart'), lossRows, [
    { key: 'loss', color: cssColor('--yellow'), width: 1.6, transform: value => Math.log10(Number(value) + 1) }
  ], { log: true });
}

function updateTable(rows) {
  const body = document.getElementById('episode-table');
  if (!rows.length) { body.innerHTML = '<tr><td colspan="6">Waiting for training data…</td></tr>'; return; }
  body.replaceChildren(...rows.map(row => {
    const tr = document.createElement('tr');
    const values = [
      formatNumber(row.episode, 0), formatNumber(row.reward), formatNumber(row.steps, 0),
      row.reason.replaceAll('_', ' '), formatNumber(row.epsilon, 3), formatNumber(row.loss, 4)
    ];
    values.forEach((value, index) => {
      const cell = document.createElement('td');
      if (index === 3) { const badge = document.createElement('span'); badge.className = `result ${row.reason}`; badge.textContent = value; cell.appendChild(badge); }
      else cell.textContent = value;
      tr.appendChild(cell);
    });
    return tr;
  }));
}

async function refreshDashboard() {
  try {
    const payload = await api('/api/dashboard');
    updateProcess(payload.process);
    if (payload.training.has_data) {
      updateMetrics(payload.training);
      updateCharts(payload.training.history);
      updateTable(payload.training.recent_rows);
    }
    document.getElementById('connection-state').textContent = 'Live · refreshes every 3 seconds';
  } catch (error) {
    document.getElementById('connection-state').textContent = 'Connection error';
    showNotice(error.message, true);
  }
}

function makeField(field) {
  const wrapper = document.createElement('div');
  wrapper.className = field.type === 'bool' ? 'checkbox-field' : 'field';
  const id = `config-${field.path.replaceAll('.', '-')}`;
  const label = document.createElement('label'); label.htmlFor = id; label.textContent = field.label;
  let input;
  if (field.type === 'select') {
    input = document.createElement('select');
    field.choices.forEach(choice => { const option = document.createElement('option'); option.value = choice; option.textContent = choice; input.appendChild(option); });
    input.value = field.value;
  } else {
    input = document.createElement('input');
    input.type = field.type === 'bool' ? 'checkbox' : 'number';
    if (field.type === 'bool') input.checked = field.value;
    else {
      input.value = field.value;
      if (field.minimum !== undefined) input.min = field.minimum;
      if (field.maximum !== undefined) input.max = field.maximum;
      input.step = field.step || 1;
    }
  }
  input.id = id; input.dataset.path = field.path; input.dataset.type = field.type;
  if (field.type === 'bool') wrapper.append(input, label); else wrapper.append(label, input);
  return wrapper;
}

function renderConfig(fields) {
  configFields = fields;
  const target = document.getElementById('config-fields');
  target.replaceChildren();
  const groups = new Map();
  fields.forEach(field => {
    if (!groups.has(field.group)) groups.set(field.group, []);
    groups.get(field.group).push(field);
  });
  groups.forEach((items, name) => {
    const fieldset = document.createElement('fieldset'); fieldset.className = 'config-group';
    const legend = document.createElement('legend'); legend.textContent = name; fieldset.appendChild(legend);
    items.forEach(item => fieldset.appendChild(makeField(item)));
    target.appendChild(fieldset);
  });
}

async function loadConfig() {
  try { const payload = await api('/api/config'); renderConfig(payload.fields); }
  catch (error) { showNotice(error.message, true); }
}

async function saveConfig(event) {
  event.preventDefault();
  const values = {};
  document.querySelectorAll('#config-fields [data-path]').forEach(input => {
    if (input.dataset.type === 'bool') values[input.dataset.path] = input.checked;
    else if (input.dataset.type === 'int') values[input.dataset.path] = Number.parseInt(input.value, 10);
    else if (input.dataset.type === 'float') values[input.dataset.path] = Number.parseFloat(input.value);
    else values[input.dataset.path] = input.value;
  });
  try {
    const payload = await api('/api/config', { method: 'POST', body: JSON.stringify({ values }) });
    renderConfig(payload.fields); showNotice(payload.message);
  } catch (error) { showNotice(error.message, true); }
}

async function processAction(action) {
  if (action === 'stop' && !window.confirm('Stop Webots now? Progress since the last checkpoint may be lost.')) return;
  try {
    const payload = await api(`/api/process/${action}`, { method: 'POST' });
    updateProcess(payload.process); showNotice(`Training ${action} request completed.`); setTimeout(refreshDashboard, 600);
  } catch (error) { showNotice(error.message, true); }
}

document.getElementById('start-button').addEventListener('click', () => processAction('start'));
document.getElementById('pause-button').addEventListener('click', () => processAction('pause'));
document.getElementById('resume-button').addEventListener('click', () => processAction('resume'));
document.getElementById('stop-button').addEventListener('click', () => processAction('stop'));
document.getElementById('reload-config').addEventListener('click', loadConfig);
document.getElementById('config-form').addEventListener('submit', saveConfig);
window.addEventListener('resize', () => refreshDashboard());

loadConfig();
refreshDashboard();
setInterval(refreshDashboard, 3000);
