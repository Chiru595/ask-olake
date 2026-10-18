/*! Ask OLake chat widget - vanilla web component, no dependencies. */
(() => {
'use strict';
const W = window, D = document;
if (W.AskOLake && W.AskOLake.__loaded) return;

// == Config from our own <script> tag (read at load time)
const script = D.currentScript;
const sattr = n => (script && script.hasAttribute(n) ? script.getAttribute(n) : null);
const scriptOrigin = () => { try { return new URL(script.src, location.href).origin; } catch (e) { return ''; } };
const apiAttr = sattr('data-api-url'), themeAttr = sattr('data-theme') || '';
const CONFIG = {
  api: (apiAttr === null ? scriptOrigin() : apiAttr.trim()).replace(/\/+$/, ''), // "" => same origin
  position: sattr('data-position') === 'bottom-left' ? 'bottom-left' : 'bottom-right',
  theme: /^(light|dark)$/.test(themeAttr) ? themeAttr : 'auto'
};
const MAX = 1000, TIMEOUT = 60000, DOCS = 'https://olake.io/docs/';
const REFUSAL = "i don't know based on the olake documentation";
const STARTERS = ['How do I set up Postgres CDC?', 'Which Iceberg catalogs does OLake support?',
  'What does OLake Fusion compaction do?', 'How do I deploy OLake with Helm?'];

function h(tag, props, kids) {
  const e = D.createElement(tag);
  for (const k in props || {}) {
    const v = props[k];
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.slice(0, 2) === 'on') e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids || []) if (c != null) e.appendChild(typeof c === 'string' ? D.createTextNode(c) : c);
  return e;
}
const ext = (href, kids, x) => h('a', Object.assign({ href, target: '_blank', rel: 'noopener noreferrer' }, x), kids);
const txt = s => D.createTextNode(s);

function safeUrl(u) {
  try { const x = new URL(String(u).trim()); return /^https?:$/.test(x.protocol) ? x.href : null; } catch (e) { return null; }
}
const ACR = { cdc: 'CDC', aws: 'AWS', s3: 'S3', sql: 'SQL', api: 'API', jdbc: 'JDBC', mysql: 'MySQL', mongodb: 'MongoDB', olake: 'OLake' };
const GENERIC = /^(index|overview|setup|intro|introduction|readme|config|configuration)$/i;
function pretty(s) {
  try { s = decodeURIComponent(s); } catch (e) { /* keep raw */ }
  return s.replace(/\.(mdx?|html?)$/i, '').replace(/[-_+]+/g, ' ').trim()
    .replace(/\w+/g, w => { const l = w.toLowerCase(); return ACR[l] || l[0].toUpperCase() + l.slice(1); });
}
// "https://olake.io/docs/connectors/postgres/setup#cdc-setup" -> "Postgres Setup > CDC Setup"
function titleFromUrl(u) {
  let x;
  try { x = new URL(u); } catch (e) { return String(u || 'Source'); }
  const segs = x.pathname.split('/').filter(Boolean), last = segs.pop() || '', parent = segs.pop();
  let t = last ? pretty(last) : '';
  if (last && parent && GENERIC.test(last.replace(/\.\w+$/, '')) && !/^docs$/i.test(parent)) t = pretty(parent) + ' ' + t;
  if (x.hash.length > 1) t = (t ? t + ' \u203a ' : '') + pretty(x.hash.slice(1));
  return t || x.hostname;
}
const normalizeSources = list => (Array.isArray(list) ? list : []).map(s => {
  const url = typeof s === 'string' ? s : (s && s.url) || '';
  return { url, title: String((s && typeof s === 'object' && s.title) || titleFromUrl(url)) };
}).filter(s => s.url);
const isRefusal = (answer, flag) => flag === true ||
  String(answer || '').trim().replace(/\u2019/g, "'").toLowerCase().startsWith(REFUSAL);
function parseRetryAfter(v) {
  if (v == null || v === '') return null; // delta-seconds or HTTP date
  const n = isFinite(v) ? +v : (Date.parse(v) - Date.now()) / 1000;
  return isNaN(n) ? null : Math.max(0, Math.ceil(n));
}

// == SSE parser, fed with decoded text chunks
function createSSEParser(onEvent) {
  let buf = '', ev = '', data = [];
  const line = l => {
    if (l === '') { if (data.length) onEvent(ev || 'message', data.join('\n')); ev = ''; data = []; return; }
    if (l[0] === ':') return;
    const i = l.indexOf(':'), f = i < 0 ? l : l.slice(0, i);
    let v = i < 0 ? '' : l.slice(i + 1);
    if (v[0] === ' ') v = v.slice(1);
    if (f === 'event') ev = v; else if (f === 'data') data.push(v);
  };
  return {
    push(chunk) {
      buf += chunk;
      const end = buf.endsWith('\r') ? buf.length - 1 : buf.length; // "\n" may follow in next chunk
      const lines = buf.slice(0, end).split(/\r\n|\r|\n/);
      buf = lines.pop() + buf.slice(end);
      lines.forEach(line);
    },
    end() { if (buf) line(buf.replace(/\r$/, '')); buf = ''; line(''); }
  };
}

// == Safe markdown -> DOM nodes (model text never touches innerHTML)
const LIST_RE = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/, FENCE_RE = /^(\s*)(`{3,}|~{3,})\s*([\w+#.-]*)/,
  HEAD_RE = /^\s{0,3}#{1,6}\s+(.*?)(?:\s+#+)?\s*$/, TSEP_RE = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;
// code | [text](url) | [1] or [1, 2] | **b** | __b__ | *i* | _i_ | bare URL
let SRC = null; // sources of the answer being rendered
const INL = /`([^`\n]+)`|\[([^\]\n]+)\]\(\s*(\S+?)(?:\s+"[^"]*")?\s*\)|\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]|\*\*(.+?)\*\*|__(.+?)__|\*(?![\s*])(.+?)\*|\b_(?![\s_])(.+?)_\b|(https?:\/\/[^\s<>"'`]*[^\s<>"'`.,;:!?)\]])/.source;

function cite(k) {
  const s = SRC && SRC[k - 1], u = s && safeUrl(s.url);
  return h('sup', { class: 'cite' }, [u ? ext(u, [String(k)], { title: s.title, 'aria-label': `Source ${k}: ${s.title}` })
    : h('span', { text: String(k) })]);
}
function inline(text, parent) {
  const re = new RegExp(INL, 'g');
  let last = 0, m, u;
  while ((m = re.exec(text))) {
    if (m.index > last) parent.appendChild(txt(text.slice(last, m.index)));
    last = re.lastIndex;
    if (m[1]) parent.appendChild(h('code', { text: m[1] }));
    else if (m[2]) { u = safeUrl(m[3]); u ? parent.appendChild(inline(m[2], ext(u))) : inline(m[2], parent); }
    else if (m[4]) m[4].split(/\s*,\s*/).forEach(k => parent.appendChild(cite(+k)));
    else if (m[5] || m[6]) parent.appendChild(inline(m[5] || m[6], h('strong')));
    else if (m[7] || m[8]) parent.appendChild(inline(m[7] || m[8], h('em')));
    else { u = safeUrl(m[9]); parent.appendChild(u ? ext(u, [m[9]]) : txt(m[9])); }
  }
  if (last < text.length) parent.appendChild(txt(text.slice(last)));
  return parent;
}
const isTable = (L, i) => L[i].includes('|') && /\|/.test(L[i + 1]) && TSEP_RE.test(L[i + 1]);
const splitRow = l => l.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(c => c.trim());
const blockStart = (L, i) => [FENCE_RE, HEAD_RE, LIST_RE].some(r => r.test(L[i])) || isTable(L, i);

function readFence(L, i) {
  const m = L[i].match(FENCE_RE), strip = new RegExp('^ {0,' + m[1].length + '}'), code = [];
  for (i++; i < L.length; i++) {
    const t = L[i].trim();
    if (t.startsWith(m[2]) && /^[`~]+$/.test(t)) { i++; break; }
    code.push(L[i].replace(strip, ''));
  }
  return { i, el: h('div', { class: 'code' }, [
    h('div', { class: 'bar' }, [h('span', { text: m[3] || 'code' }), h('button', { type: 'button', class: 'copy', 'aria-label': 'Copy code', text: 'Copy' })]),
    h('pre', null, [h('code', { text: code.join('\n') })])]) };
}
function parseList(L, i) {
  const first = L[i].match(LIST_RE), base = first[1].length, ordered = /\d/.test(first[2]), list = h(ordered ? 'ol' : 'ul');
  if (ordered && parseInt(first[2], 10) !== 1) list.setAttribute('start', parseInt(first[2], 10));
  let li = null, r;
  while (i < L.length) {
    const l = L[i], m = l.match(LIST_RE);
    if (!m) {
      if (!l.trim()) { // blank: continue if a sibling item or indented text follows
        const nx = L[i + 1] || '', nm = nx.match(LIST_RE), ni = nx.match(/^\s*/)[0].length;
        if ((nm && ni >= base && (ni > base || /\d/.test(nm[2]) === ordered)) || (li && nx.trim() && ni >= base + 2)) { i++; continue; }
        break;
      }
      if (!li || !/^\s/.test(l)) break;
      if (FENCE_RE.test(l)) { r = readFence(L, i); li.appendChild(r.el); i = r.i; continue; }
      li.appendChild(h('br')); inline(l.trim(), li); i++; continue;
    }
    if (m[1].length < base) break;
    if (m[1].length >= base + 2 && li) { r = parseList(L, i); li.appendChild(r.el); i = r.i; continue; }
    if (/\d/.test(m[2]) !== ordered) break;
    list.appendChild(li = inline(m[3], h('li'))); i++;
  }
  return { el: list, i };
}
function renderTable(L, i, out) {
  const head = splitRow(L[i]), cell = (tag, c) => inline(c || '', h(tag));
  const tb = h('tbody');
  for (i += 2; i < L.length && L[i].includes('|') && L[i].trim(); i++) {
    const cells = splitRow(L[i]);
    tb.appendChild(h('tr', null, head.map((_, j) => cell('td', cells[j]))));
  }
  out.appendChild(h('div', { class: 'tw' }, [h('table', null, [h('thead', null, [h('tr', null, head.map(c => cell('th', c)))]), tb])]));
  return i;
}
function renderBlocks(L, out) {
  let i = 0, m, r;
  while (i < L.length) {
    const l = L[i];
    if (!l.trim()) i++;
    else if (FENCE_RE.test(l)) { r = readFence(L, i); out.appendChild(r.el); i = r.i; }
    else if ((m = l.match(HEAD_RE))) { out.appendChild(h('p', { class: 'hd' }, [inline(m[1], h('strong'))])); i++; }
    else if (isTable(L, i)) i = renderTable(L, i, out);
    else if (LIST_RE.test(l)) { r = parseList(L, i); out.appendChild(r.el); i = r.i; }
    else {
      const p = inline(l.trim(), h('p'));
      for (i++; i < L.length && L[i].trim() && !blockStart(L, i); i++) { p.appendChild(h('br')); inline(L[i].trim(), p); }
      out.appendChild(p);
    }
  }
  return out;
}
// [k] links to sources[k-1]; sources=null (streaming) => unlinked markers
function renderMarkdown(src, sources) {
  SRC = sources || null;
  return renderBlocks(String(src || '').replace(/\r\n?/g, '\n').split('\n'), D.createDocumentFragment());
}

// == Styles & static shell
const icon = d => `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${d}"/></svg>`;
const I_CHAT = icon('M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z');
const CSS = `:host{all:initial}
.w{--ac:#2563eb;--ach:#1d4ed8;--onac:#fff;--bg:#fff;--bg2:#f8fafc;--fg:#0f172a;--mu:#64748b;--bd:#e2e8f0;--bot:#f1f5f9;--ubg:#2563eb;--ufg:#fff;--cbg:#0f172a;--er:#b91c1c;--erbg:#fef2f2;--wa:#92400e;--wabg:#fffbeb;--sh:0 16px 48px rgba(15,23,42,.18),0 2px 6px rgba(15,23,42,.08);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--fg)}
.w.dark{--ac:#60a5fa;--ach:#93c5fd;--onac:#0b1220;--bg:#0f172a;--bg2:#111c31;--fg:#e2e8f0;--mu:#94a3b8;--bd:#24324a;--bot:#1a2436;--ubg:#1d4ed8;--ufg:#eff6ff;--cbg:#020617;--er:#fca5a5;--erbg:#2a1418;--wa:#fcd34d;--wabg:#2a2010;--sh:0 16px 48px rgba(0,0,0,.55)}
*{box-sizing:border-box}svg{fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}button{font:inherit;color:inherit;cursor:pointer;border:0;background:none}:focus-visible{outline:2px solid var(--ac);outline-offset:2px}[hidden]{display:none!important}
.launcher,.logo,.send{display:grid;place-items:center;background:var(--ac);color:var(--onac)}button:disabled{opacity:.45;cursor:not-allowed}
.launcher,.panel{position:fixed;bottom:20px;right:20px;z-index:2147483000;box-shadow:var(--sh)}.left .launcher,.left .panel{right:auto;left:20px}
.launcher{width:56px;height:56px;border-radius:50%;transition:transform .15s}.launcher:hover{transform:translateY(-2px);background:var(--ach)}.launcher svg{width:26px}
.panel{width:400px;max-width:calc(100vw - 40px);height:min(660px,calc(100vh - 40px));background:var(--bg);border:1px solid var(--bd);border-radius:16px;display:flex;flex-direction:column;overflow:hidden;animation:pop .18s ease-out}
@keyframes pop{from{opacity:0;transform:translateY(10px) scale(.98)}}
@media (max-width:640px){.panel,.left .panel{inset:0;width:100%;max-width:none;height:100%;border-radius:0;border:0}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
header{display:flex;align-items:center;gap:6px;padding:10px 10px 10px 14px;border-bottom:1px solid var(--bd);background:var(--bg2)}
h2{flex:1;display:flex;align-items:center;gap:10px;margin:0;font-size:15px;font-weight:650}.logo{width:28px;height:28px;border-radius:8px}.logo svg{width:16px}
.hb{border-radius:8px;padding:5px 9px;color:var(--mu);font-size:13px}.hb:hover{background:var(--bot);color:var(--fg)}.x{display:grid;padding:6px}.x svg{width:20px}
.log{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:14px;overscroll-behavior:contain}
.empty{margin:auto 0;text-align:center;color:var(--mu)}.empty p{margin:0}.big{font-size:16px;font-weight:650;color:var(--fg)}
.starters{display:grid;gap:8px;margin-top:16px}.starters button{text-align:left;border:1px solid var(--bd);border-radius:10px;padding:10px 12px}.starters button:hover{border-color:var(--ac);color:var(--ac)}
.msg{display:flex;min-width:0}.user{justify-content:flex-end}.bubble{max-width:86%;padding:10px 13px;border-radius:14px;overflow-wrap:anywhere;min-width:0}
.user .bubble{background:var(--ubg);color:var(--ufg);border-bottom-right-radius:4px;white-space:pre-wrap}.bot .bubble{max-width:100%;background:var(--bot);border-bottom-left-radius:4px}
.ct>:first-child{margin-top:0}.ct>:last-child{margin-bottom:0}.ct a{color:var(--ac)}.hd{margin-top:12px}
.ct p,.ct ul,.ct ol,.tw,.code{margin:0 0 9px}.ct ul,.ct ol{padding-left:20px}.ct li{margin:2px 0}
code{font:12.5px/1.5 ui-monospace,Menlo,Consolas,monospace}.ct :not(pre)>code{background:rgba(127,127,127,.16);padding:1px 5px;border-radius:5px}
.code{border-radius:9px;overflow:hidden;background:var(--cbg);color:#e2e8f0}li>.code{margin:6px 0}
.bar{display:flex;justify-content:space-between;align-items:center;padding:4px 6px 4px 12px;font-size:11.5px;color:#94a3b8;border-bottom:1px solid #334155}
.copy{border:1px solid #475569;border-radius:6px;padding:1px 8px;font-size:11.5px;color:#cbd5e1}.copy:hover{color:#fff}
pre{margin:0;padding:10px 12px;overflow-x:auto}pre code{white-space:pre}
.tw{overflow-x:auto}table{border-collapse:collapse;font-size:13px;min-width:100%}th,td{border:1px solid var(--bd);padding:5px 8px;text-align:left;vertical-align:top}th{background:var(--bg2)}
.cite{font-size:10px;line-height:0;margin:0 1px}.ct .cite>*{display:inline-block;min-width:16px;padding:0 4px;border-radius:8px;background:var(--ac);color:var(--onac);text-decoration:none;text-align:center;line-height:16px;font-weight:650}
.ct .cite span{background:var(--bd);color:var(--mu)}
.src{margin-top:10px;padding-top:8px;border-top:1px solid var(--bd);font-size:12.5px}.src p{margin:0 0 4px;font-size:11px;font-weight:650;letter-spacing:.05em;text-transform:uppercase;color:var(--mu)}
.src ol{margin:0;padding-left:20px}.src a,.dl{color:var(--ac);text-decoration:none}.src a:hover,.dl:hover{text-decoration:underline}
.refusal .bubble{background:none;border:1px dashed var(--bd);color:var(--mu)}.refusal .ct{font-style:italic}.dl{display:inline-block;margin-top:6px;font-size:13px}
.err .bubble{background:var(--erbg);color:var(--er)}.rl .bubble{background:var(--wabg);color:var(--wa)}.err p,.rl p{margin:0 0 4px}
.retry{margin-top:6px;border:1px solid currentColor;border-radius:8px;padding:3px 12px;font-size:13px}
.streaming .ct>:last-child::after{content:"";display:inline-block;width:7px;height:14px;margin-left:2px;vertical-align:-2px;background:var(--ac);animation:bl 1s steps(2) infinite}@keyframes bl{50%{opacity:0}}
.dots{display:inline-flex;gap:4px;padding:6px 2px}.dots i{width:7px;height:7px;border-radius:50%;background:var(--mu);animation:bo 1.2s infinite}
.dots i+i{animation-delay:.15s}.dots i+i+i{animation-delay:.3s}@keyframes bo{0%,80%,100%{opacity:.3}40%{opacity:1;transform:translateY(-3px)}}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
form{border-top:1px solid var(--bd);padding:10px 12px 8px}
.box{display:flex;align-items:flex-end;gap:8px;border:1px solid var(--bd);border-radius:12px;padding:6px 6px 6px 12px;background:var(--bg2)}.box:focus-within{border-color:var(--ac);box-shadow:0 0 0 1px var(--ac)}
.box textarea{flex:1;min-width:0;border:0;background:none;resize:none;font:inherit;color:inherit;max-height:120px;padding:5px 0;outline:none}textarea::placeholder{color:var(--mu)}
.send{width:34px;height:34px;flex:none;border-radius:9px}.send svg{width:18px;stroke-width:2.4}
.meta{display:flex;justify-content:space-between;gap:10px;padding:6px 2px 0;font-size:11.5px;color:var(--mu)}.cnt{flex:none;font-variant-numeric:tabular-nums}.hi{color:var(--wa)}`;

const SHELL = `<style>${CSS}</style><div class="w">
<button class="launcher" type="button" aria-label="Open Ask OLake assistant" aria-haspopup="dialog">${I_CHAT}</button>
<div class="panel" id="p" role="dialog" aria-modal="true" aria-label="Ask OLake" hidden>
<header><h2><span class="logo">${I_CHAT}</span>Ask OLake</h2>
<button class="hb clear" type="button" aria-label="Clear conversation">Clear</button>
<button class="hb x close" type="button" aria-label="Close Ask OLake">${icon('M18 6 6 18M6 6l12 12')}</button></header>
<div class="log" tabindex="0" aria-label="Conversation" aria-live="polite"></div>
<form novalidate><div class="box"><textarea rows="1" maxlength="${MAX}" aria-label="Ask a question about OLake" aria-describedby="c" placeholder="Ask a question about OLake\u2026"></textarea>
<button class="send" type="submit" aria-label="Send question" disabled>${icon('M12 19V5M5 12l7-7 7 7')}</button></div>
<div class="meta"><span>Answers are generated from OLake docs and may be imperfect.</span><span class="cnt" id="c">0/${MAX}</span></div>
</form></div></div>`;

function friendly(status, timedOut) {
  if (timedOut || status === 504) return 'The answer took too long. Please try again.';
  if (status === 503) return 'Ask OLake is temporarily unavailable. Please try again shortly.';
  return status ? `Something went wrong (error ${status}). Please try again.` : "Couldn't reach Ask OLake. Check your connection and try again.";
}
const httpErr = status => Object.assign(new Error('HTTP ' + status), { status });

// == One assistant message bubble and its states
class Bot {
  constructor(host) { this.host = host; this.b = h('div', { class: 'bubble' }); this.el = h('div', { class: 'msg bot' }, [this.b]); this.el._bot = this; }
  reset(cls) {
    cancelAnimationFrame(this.raf); this.raf = 0; clearInterval(this.tick);
    this.el.className = 'msg bot' + (cls ? ' ' + cls : '');
    this.b.replaceChildren(); this.ct = null;
  }
  loading() {
    this.reset('loading');
    this.b.appendChild(h('span', { class: 'dots' }, [h('i'), h('i'), h('i'), h('span', { class: 'sr', text: 'Ask OLake is thinking\u2026' })]));
  }
  stream(text) { // re-render the partial answer at most once per frame
    this.text = text;
    if (!this.ct) { this.reset('streaming'); this.b.appendChild(this.ct = h('div', { class: 'ct' })); }
    this.raf = this.raf || requestAnimationFrame(() => {
      this.raf = 0;
      const near = this.host._near();
      this.ct.replaceChildren(renderMarkdown(this.text, null));
      this.host._scroll(near);
    });
  }
  final(answer, sources, refusal) {
    const near = this.host._near();
    this.reset(refusal ? 'refusal' : '');
    this.b.appendChild(h('div', { class: 'ct' }, [renderMarkdown(answer, sources)]));
    if (refusal) this.b.appendChild(ext(DOCS, ['Browse the OLake documentation \u2197'], { class: 'dl' }));
    if (sources.length) {
      const ol = h('ol', null, sources.map(s => { const u = safeUrl(s.url); return h('li', null, [u ? ext(u, [s.title]) : s.title]); }));
      this.b.appendChild(h('div', { class: 'src' }, [h('p', { text: 'Sources' }), ol]));
    }
    this.host._scroll(near);
  }
  error(msg, retry) {
    this.reset('err');
    this.b.append(h('p', { role: 'alert', text: msg }), h('button', { type: 'button', class: 'retry', text: 'Retry', onclick: retry }));
    this.host._scroll(true);
  }
  rateLimited(secs, msg, retry) {
    const until = Date.now() + secs * 1000, wait = h('p');
    const btn = h('button', { type: 'button', class: 'retry', text: 'Retry', disabled: true, onclick: retry });
    this.reset('rl');
    this.b.append(h('p', { role: 'alert', text: msg || "You're asking questions a little too quickly." }), wait, btn);
    this.host._coolUntil = until;
    const upd = () => {
      const left = Math.ceil((until - Date.now()) / 1000);
      if (left > 0) wait.textContent = `Please wait ${left}s before trying again.`;
      else { clearInterval(this.tick); wait.textContent = 'You can try again now.'; btn.disabled = false; this.host._sync(); }
    };
    upd(); this.tick = setInterval(upd, 1000);
    this.host._scroll(true);
  }
}

// == The <ask-olake> element
const Base = typeof HTMLElement === 'undefined' ? class {} : HTMLElement;
class AskOLakeElement extends Base {
  static get observedAttributes() { return ['theme', 'position']; }
  constructor() {
    super();
    this._busy = this._isOpen = false; this._ctl = null; this._coolUntil = this._count = 0;
    this._onKey = this._onKey.bind(this); this._onMq = () => this._applyTheme();
  }
  connectedCallback() {
    if (!this._root) this._build();
    D.addEventListener('keydown', this._onKey, true);
    this._mq.addEventListener('change', this._onMq);
    this._applyTheme(); this._state('idle');
  }
  disconnectedCallback() {
    D.removeEventListener('keydown', this._onKey, true);
    this._mq.removeEventListener('change', this._onMq);
    if (this._ctl) { this._ctl.cleared = true; this._ctl.abort(); }
  }
  attributeChangedCallback() { if (this._root) this._applyTheme(); }
  _cfg(n) { return this.getAttribute(n) || CONFIG[n]; }
  get api() { const v = this.getAttribute('api-url'); return v == null ? CONFIG.api : v.trim().replace(/\/+$/, ''); }

  _build() {
    const r = this._root = this.attachShadow({ mode: 'open' }), q = s => r.querySelector(s);
    r.innerHTML = SHELL; // static markup only; dynamic content uses DOM APIs
    this._w = q('.w'); this._launch = q('.launcher'); this._panel = q('.panel'); this._log = q('.log');
    this._ta = q('textarea'); this._send = q('.send'); this._cnt = q('.cnt');
    this._mq = W.matchMedia('(prefers-color-scheme: dark)');
    this._launch.onclick = () => this.open();
    q('.close').onclick = () => this.close();
    q('.clear').onclick = () => this.clear();
    q('form').onsubmit = e => { e.preventDefault(); this.ask(this._ta.value); };
    this._ta.oninput = () => this._sync();
    this._ta.onkeydown = e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); this.ask(this._ta.value); } };
    this._log.onclick = e => { const b = e.target.closest && e.target.closest('.copy'); if (b) this._copy(b); };
    this._empty();
  }
  _applyTheme() {
    const t = this._cfg('theme');
    this._w.classList.toggle('dark', t === 'dark' || (t !== 'light' && this._mq.matches));
    this._w.classList.toggle('left', this._cfg('position') === 'bottom-left');
  }
  _empty() {
    this._log.replaceChildren(h('div', { class: 'empty' }, [
      h('p', { class: 'big', text: 'Hi! Ask me anything about OLake.' }),
      h('p', { text: 'Answers come from the OLake documentation, with sources.' }),
      h('div', { class: 'starters' }, STARTERS.map(s => h('button', { type: 'button', text: s, onclick: () => this.ask(s) })))]));
  }
  _sync() {
    const ta = this._ta, n = ta.value.length;
    this._cnt.textContent = n + '/' + MAX;
    this._cnt.classList.toggle('hi', n > MAX * 0.9);
    ta.style.height = 'auto'; ta.style.height = Math.min(ta.scrollHeight, 120) + 'px';
    this._send.disabled = this._busy || !ta.value.trim() || Date.now() < this._coolUntil;
  }
  _setBusy(b) {
    const hadFocus = this._root.activeElement === this._ta;
    this._busy = b; this._ta.disabled = b;
    this._log.setAttribute('aria-busy', String(b));
    this._sync();
    if (!b && this._isOpen && (hadFocus || !this._root.activeElement)) this._ta.focus();
  }
  _state(s) { this.setAttribute('data-state', s); } // idle | loading | streaming | error | rate-limited
  _near() { const l = this._log; return l.scrollHeight - l.scrollTop - l.clientHeight < 80; }
  _scroll(force) { if (force) this._log.scrollTop = this._log.scrollHeight; }
  _copy(btn) {
    const code = btn.closest('.code').querySelector('code').textContent;
    const done = () => { btn.textContent = 'Copied'; setTimeout(() => { btn.textContent = 'Copy'; }, 1500); };
    if (navigator.clipboard) navigator.clipboard.writeText(code).then(done, () => {});
  }
  _onKey(e) { // Esc closes; Tab cycles inside the open dialog
    if (!this._isOpen) return;
    if (e.key === 'Escape') { e.preventDefault(); return this.close(); }
    if (e.key !== 'Tab') return;
    const f = [...this._panel.querySelectorAll('button,textarea,a[href],[tabindex]:not([tabindex="-1"])')]
      .filter(x => !x.disabled && x.getClientRects().length);
    if (!f.length) return;
    const a = this._root.activeElement, inside = a && this._panel.contains(a), first = f[0], last = f[f.length - 1];
    if (e.shiftKey && (!inside || a === first)) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && (!inside || a === last)) { e.preventDefault(); first.focus(); }
  }

  open() {
    if (this._isOpen) return;
    this._isOpen = true; this._panel.hidden = false; this._launch.hidden = true;
    this._sync();
    (this._ta.disabled ? this._panel.querySelector('.close') : this._ta).focus();
  }
  close() {
    if (!this._isOpen) return;
    this._isOpen = false; this._panel.hidden = true; this._launch.hidden = false;
    this._launch.focus();
  }
  clear() {
    if (this._ctl) { this._ctl.cleared = true; this._ctl.abort(); }
    this._log.querySelectorAll('.bot').forEach(m => m._bot && m._bot.reset());
    this._count = 0; this._empty(); this._state('idle');
    if (this._isOpen) this._ta.focus();
  }
  ask(question) {
    const q = String(question || '').trim().slice(0, MAX);
    if (!q || this._busy || Date.now() < this._coolUntil) return;
    if (!this._isOpen) this.open();
    if (!this._count++) this._log.replaceChildren();
    const bot = new Bot(this);
    this._log.append(h('div', { class: 'msg user' }, [h('div', { class: 'bubble', text: q })]), bot.el);
    this._ta.value = '';
    this._scroll(true);
    this._run(q, bot);
  }

  async _run(q, bot) {
    const ctl = new AbortController(), api = this.api;
    let timer, text = '', gotToken = false, fin = null, err = null;
    const arm = () => { clearTimeout(timer); timer = setTimeout(() => { ctl.timedOut = true; ctl.abort(); }, TIMEOUT); };
    const retry = () => { if (!this._busy && Date.now() >= this._coolUntil) this._run(q, bot); };
    const post = (path, accept) => fetch(api + path, { method: 'POST', signal: ctl.signal,
      headers: { 'Content-Type': 'application/json', Accept: accept }, body: JSON.stringify({ question: q }) });
    const limited = (secs, msg) => { this._state('rate-limited'); bot.rateLimited(secs == null ? 10 : secs, msg, retry); };
    const rate = async res => {
      let body = null;
      try { body = await res.json(); } catch (e) { /* ignore */ }
      limited(parseRetryAfter(res.headers.get('Retry-After')), body && body.answer);
    };
    this._ctl = ctl; arm();
    this._setBusy(true); this._state('loading'); bot.loading();
    try {
      let res = null;
      try { res = await post('/ask/stream', 'text/event-stream'); } catch (e) { if (ctl.signal.aborted) throw e; }
      if (res && res.status === 429) return await rate(res);
      if (res && res.ok && res.body) {
        const reader = res.body.getReader(), dec = new TextDecoder();
        let done = false;
        const parser = createSSEParser((ev, data) => {
          let d; try { d = JSON.parse(data) || {}; } catch (e) { d = {}; }
          if (ev === 'token' && typeof d.t === 'string') {
            if (!gotToken) this._state('streaming');
            gotToken = true; text += d.t; bot.stream(text);
          } else if (ev === 'final') fin = d;
          else if (ev === 'error') err = d;
          else if (ev === 'done') done = true;
        });
        try {
          while (!done && !err) {
            const c = await reader.read();
            if (c.done) break;
            parser.push(dec.decode(c.value, { stream: true }));
          }
          parser.push(dec.decode()); parser.end();
          if (done || err) reader.cancel().catch(() => {});
        } catch (e) {
          if (ctl.signal.aborted) throw e;
          if (gotToken && !fin) throw httpErr(0); // connection dropped mid-answer
        }
      }
      if (err) {
        if (err.status === 429) return limited(parseRetryAfter(err.retry_after), err.message);
        throw httpErr(err.status || 500);
      }
      if (!fin && gotToken) fin = { answer: text, sources: [] };
      if (!fin) { // stream failed before any token: fall back to POST /ask
        arm();
        const r2 = await post('/ask', 'application/json');
        if (r2.status === 429) return await rate(r2);
        if (!r2.ok) throw httpErr(r2.status);
        fin = await r2.json();
      }
      bot.final(String(fin.answer || ''), normalizeSources(fin.sources), isRefusal(fin.answer, fin.refusal));
      this._state('idle');
    } catch (e) {
      if (ctl.cleared) return;
      this._state('error');
      bot.error(friendly(e.status || 0, ctl.timedOut), retry);
    } finally {
      clearTimeout(timer);
      if (this._ctl === ctl) { this._ctl = null; this._setBusy(false); }
    }
  }
}

// == Public namespace & auto-mount
const el = () => D.querySelector('ask-olake');
const call = m => (...a) => { const e = el(); if (e) e[m](...a); };
W.AskOLake = { __loaded: true, config: CONFIG, open: call('open'), close: call('close'), ask: call('ask'),
  _internals: { renderMarkdown, createSSEParser, titleFromUrl, normalizeSources, isRefusal, safeUrl, parseRetryAfter } };
if (W.customElements && typeof HTMLElement !== 'undefined') {
  if (!customElements.get('ask-olake')) customElements.define('ask-olake', AskOLakeElement);
  const boot = () => { if (!el()) D.body.appendChild(D.createElement('ask-olake')); };
  D.readyState === 'loading' ? D.addEventListener('DOMContentLoaded', boot) : boot();
}
})();
