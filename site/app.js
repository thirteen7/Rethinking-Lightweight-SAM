"use strict";
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const names = {tinysam: "TinySAM", mobilesam: "MobileSAM", vith: "SAM ViT-H"};
const state = {example: null, model: "tinysam", prompt: "point", round: 1, opacity: .5, dataset: "coco", benchmarkPrompt: "point", benchmarkStage: 0, everyView: "masks", pairShown: false};
const pictures = new Map();
let records, benchmarks, manifest, config, everything, everyComparison, paperTiming, renderVersion = 0, toastTimer;
const number = value => Number(value).toLocaleString("en-US");
const svg = id => `<svg class="icon" aria-hidden="true"><use href="#i-${id}"/></svg>`;

async function json(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path}`);
  return response.json();
}
function toast(message) {
  $("#toast").textContent = message;
  $("#toast").classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("#toast").classList.remove("visible"), 2600);
}
async function copy(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast("Copied to clipboard");
  } catch {
    const input = document.createElement("textarea");
    input.value = text;
    input.style.cssText = "position:fixed;left:-9999px";
    document.body.append(input);
    input.select();
    document.execCommand("copy");
    input.remove();
    toast("Copied to clipboard");
  }
}
function active(buttons, selected) {
  buttons.forEach(button => {
    const isActive = button === selected;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  });
}
function image(path) {
  if (!pictures.has(path)) pictures.set(path, new Promise((resolve, reject) => {
    const picture = new Image();
    picture.onload = () => resolve(picture);
    picture.onerror = () => reject(new Error("Could not load an example image"));
    picture.src = path;
  }));
  return pictures.get(path);
}
async function opaqueMask(path) {
  const key = "opaque:" + path;
  if (!pictures.has(key)) pictures.set(key, (async () => {
    const picture = await image(path);
    const canvas = document.createElement("canvas");
    canvas.width = picture.naturalWidth; canvas.height = picture.naturalHeight;
    const context = canvas.getContext("2d", {willReadFrequently: true});
    context.drawImage(picture, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
    for (let index = 3; index < pixels.data.length; index += 4) pixels.data[index] = pixels.data[index] ? 255 : 0;
    context.putImageData(pixels, 0, 0);
    return canvas;
  })());
  return pictures.get(key);
}
function drawPrompts(context, points, box, scale = 1) {
  if (box) {
    context.save();
    context.strokeStyle = "#fff"; context.lineWidth = 3 * scale;
    context.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
    context.strokeStyle = "#a889f4"; context.lineWidth = 2 * scale;
    context.setLineDash([8 * scale, 5 * scale]);
    context.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
    context.restore();
  }
  for (const point of points) {
    if (point.label < 0) continue;
    const radius = 8 * scale;
    context.save();
    context.fillStyle = point.label ? "#7960e1" : "#d87865";
    context.strokeStyle = "#fff"; context.lineWidth = 2 * scale;
    context.beginPath(); context.arc(point.x, point.y, radius, 0, Math.PI * 2); context.fill(); context.stroke();
    context.beginPath(); context.moveTo(point.x - radius * .4, point.y); context.lineTo(point.x + radius * .4, point.y);
    if (point.label) { context.moveTo(point.x, point.y - radius * .4); context.lineTo(point.x, point.y + radius * .4); }
    context.lineWidth = 1.7 * scale; context.stroke(); context.restore();
  }
}
function refreshExampleOptions() {
  const list = $("#example-list");
  const options = records.examples.filter(row => row.modes.includes(state.prompt) && (!row.visible_for || row.visible_for[state.model]?.includes(state.prompt)));
  const preferred = records.default_examples?.[state.model]?.[state.prompt];
  if (!options.some(row => row.id === state.example)) state.example = preferred || options[0].id;
  const key = state.prompt + ":" + state.model;
  if (list.dataset.mode !== key) {
    options.sort((a,b) => Number(b.id === preferred) - Number(a.id === preferred));
    list.innerHTML = options.map(row => {
      const first = row.trajectories?.[state.model]?.[state.prompt]?.[0];
      const gain = first ? first.refined.iou_percent - first.reference.iou_percent : null;
      const label = gain !== null && gain > 1 ? `<em class="gain-tag">+${gain.toFixed(1)} pp · one prompt</em>` : "";
      return `<button class="example-option" data-example="${row.id}" aria-pressed="false"><img src="${row.image}" alt="" width="60" height="60"><span><b>${row.title}</b><small>${row.dataset} · Image ${row.image_id}</small>${label}</span>${svg("arrow")}</button>`;
    }).join("");
    list.dataset.mode = key;
  }
  active($$("[data-example]"),$(`[data-example="${state.example}"]`));
}
async function renderExample() {
  const version = ++renderVersion;
  setEveryControls(state.prompt === "everything");
  if (state.prompt === "everything") return renderEverything(version);
  refreshExampleOptions();
  $("#reference-heading").textContent = "Original " + names[state.model];
  $("#refined-heading").textContent = "Prompt-adaptive " + names[state.model];
  $("#refined-canvas").hidden = false;
  $("#prediction-grid").classList.remove("is-flow");
  $("#reference-canvas").setAttribute("aria-label","Original lightweight model prediction with the same prompt");
  $("#refined-canvas").setAttribute("aria-label","Refined model mask and prompts");
  $("#example-metric-note").textContent = "Measured legacy IoU · curated example";
  $("#trajectory-heading").textContent = "SAME PROMPT / MEASURED IoU";
  $(".explorer-note").textContent = "Curated foreground-only prompts. Original and refined models receive identical clicks and keep independent mask feedback. IoU is measured against the annotated target; these examples illustrate successful cases.";
  const sample = records.examples.find(item => item.id === state.example);
  const trajectory = sample.trajectories[state.model][state.prompt];
  const round = trajectory[state.round - 1];
  $("#example-caption").textContent = `${sample.dataset} · image ${sample.image_id} · object ${sample.annotation_id} · ${state.round === 1 ? state.prompt === "point" ? "one click" : "one box" : `+${state.round-1} corrective click${state.round>2?"s":""}`}`;
  $("#reference-score").textContent = round.reference.iou_percent.toFixed(2) + "%";
  $("#refined-score").textContent = round.refined.iou_percent.toFixed(2) + "%";
  $("#example-trajectory").innerHTML = trajectory.map(row => `<div class="trajectory-value${row.round === state.round ? " active" : ""}"><span>${row.round === 1 ? state.prompt === "point" ? "ONE CLICK" : "ONE BOX" : `+${row.round-1} CORRECTIVE CLICK${row.round>2?"S":""}`}</span><b>${row.refined.iou_percent.toFixed(2)}%</b><small>Original ${row.reference.iou_percent.toFixed(2)}%</small></div>`).join("");
  const source = $("#image-source");
  if (sample.source_url && /^https?:\/\//.test(sample.source_url)) {
    source.href = "assets/credits.md"; source.target = "_blank"; source.rel = "noopener"; source.hidden = false;
  } else source.hidden = true;
  const photo = await image(sample.image);
  await Promise.all(["reference", "refined"].map(async field => {
    const mask = await opaqueMask(round[field].mask);
    if (version !== renderVersion) return;
    const canvas = $(`#${field}-canvas`);
    canvas.width = sample.width; canvas.height = sample.height;
    const context = canvas.getContext("2d");
    context.drawImage(photo, 0, 0);
    context.globalAlpha = state.opacity; context.drawImage(mask, 0, 0); context.globalAlpha = 1;
    drawPrompts(context, round.points, round.box, Math.max(sample.width, sample.height) / 640);
  }));
}
function wireExamples() {
  $("#example-model").addEventListener("change", event => {
    state.model = event.target.value;
    state.example = state.prompt === "everything" ? "fruit" : records.default_examples?.[state.model]?.[state.prompt] || state.example;
    state.round = 1; active($$("[data-round]"), $("[data-round='1']"));
    renderExample().catch(error => toast(error.message));
  });
  $("#example-list").addEventListener("click", event => {
    const button = event.target.closest("[data-example]");
    if (!button) return;
    state.example = button.dataset.example; renderExample().catch(error => toast(error.message));
  });
  $$("[data-prompt]").forEach(button => button.addEventListener("click", () => {
    state.prompt = button.dataset.prompt;
    state.example = state.prompt === "everything" ? "fruit" : records.default_examples?.[state.model]?.[state.prompt] || state.example;
    state.round = 1; active($$("[data-round]"), $("[data-round='1']"));
    active($$("[data-prompt]"), button); renderExample().catch(error => toast(error.message));
  }));
  $$("[data-round]").forEach(button => button.addEventListener("click", () => {
    state.round = Number(button.dataset.round); active($$("[data-round]"), button); renderExample().catch(error => toast(error.message));
  }));
  $("#mask-opacity").addEventListener("input", event => {
    state.opacity = Number(event.target.value) / 100; $("#opacity-value").textContent = event.target.value + "%";
    renderExample().catch(error => toast(error.message));
  });
  const hero = records.examples.find(row => row.id === records.default_examples?.tinysam?.point) || records.examples[0];
  const first = hero.trajectories.tinysam.point[0];
  $("#hero-reference").textContent = first.reference.iou_percent.toFixed(2) + "%";
  $("#hero-refined").textContent = first.refined.iou_percent.toFixed(2) + "%";
  $$(".hero-photo>img:first-child").forEach(photo => {photo.src=hero.image;photo.alt=hero.title;photo.width=hero.width;photo.height=hero.height;});
  $(".card-back .hero-mask").src=first.reference.mask;
  $(".card-front .hero-mask").src=first.refined.mask;
  $(".hero-visual-note").textContent=`First-click comparison · TinySAM · COCO image ${hero.image_id}`;
  $(".hero-visual").setAttribute("aria-label",`Original TinySAM versus prompt-adaptive first-click prediction on ${hero.title}`);
  $(".card-back .mini-card-head>span:first-child").textContent="Original TinySAM";
  $(".card-back .mini-card-foot>span").textContent="First-click IoU";
  $$(".hero-prompt").forEach(marker => {
    marker.style.left = (100 * first.points[0].x / hero.width) + "%";
    marker.style.top = (100 * first.points[0].y / hero.height) + "%";
  });
}
function renderBenchmarks() {
  const dataset = benchmarks.datasets[state.dataset], prompt = state.benchmarkPrompt, stage = state.benchmarkStage;
  const stageLabel = stage === 0 ? prompt === "point" ? "ONE CLICK" : "ONE BOX" : `+${stage} CORRECTIVE CLICK${stage > 1 ? "S" : ""}`;
  const left = 95, width = 390;
  let markup = "";
  for (const value of [0,25,50,75,100]) {
    const x = left + width * value / 100;
    markup += `<line x1="${x}" y1="22" x2="${x}" y2="385" stroke="#eceef5" stroke-dasharray="3 5"/><text x="${x}" y="410" font-size="10" text-anchor="middle" fill="#9ca3b4">${value}</text>`;
  }
  for (const [index,model,color] of [[0,"tinysam","#7860de"],[1,"mobilesam","#268e90"],[2,"vith","#a4783c"]]) {
    const top = 43 + index * 120, ours = dataset[model][prompt][stage], baseline = dataset.original[model][prompt][stage];
    const gain = dataset.gains[model][prompt][stage], historical = dataset.baseline_kind[model] === "historical_reference";
    markup += `<text x="10" y="${top}" font-size="13" font-weight="600" fill="#454b62">${names[model]}</text><text x="10" y="${top+27}" font-size="10" fill="#9fa6b7">Original${historical?" †":""}</text><rect x="${left}" y="${top+12}" width="${width*baseline/100}" height="21" rx="4" fill="#cbd0dd"/><text x="${left+width*baseline/100+7}" y="${top+27}" font-size="11" fill="#888fa2">${baseline.toFixed(2)}</text>`;
    markup += `<text x="10" y="${top+62}" font-size="10" fill="${color}">Ours</text><rect x="${left}" y="${top+47}" width="${width*ours/100}" height="21" rx="4" fill="${color}"/><text x="${left+width*ours/100+7}" y="${top+62}" font-size="11" font-weight="600" fill="${color}">${ours.toFixed(2)}</text><text x="579" y="${top+41}" font-size="15" font-weight="650" text-anchor="end" fill="${color}">+${gain.toFixed(2)}<tspan font-size="9"> pp${historical?" †":""}</tspan></text>`;
    const prefix = {tinysam:"tiny",mobilesam:"mobile",vith:"vith"}[model];
    $(`#${prefix}-result`).innerHTML = ours.toFixed(2) + "<span>%</span>";
    $(`#${prefix}-original`).textContent = `Original${historical?" †":""}: ${baseline.toFixed(2)}%`;
    $(`#${prefix}-gain`).textContent = `+${gain.toFixed(2)} pp ${historical?"reported difference †":"vs original"}`;
  }
  $("#benchmark-chart").innerHTML = markup;
  $("#benchmark-chart").setAttribute("aria-label", `${dataset.label}; ${stageLabel}; paper reported original versus refined IoU.`);
  $("#chart-title").textContent = dataset.label;
  $("#chart-coverage").textContent = dataset.coverage + ". " + dataset.baseline_note;
  $("#stat-prompt").textContent = $("#stat-prompt-mobile").textContent = $("#stat-prompt-vith").textContent = prompt.toUpperCase();
  $("#stat-stage").textContent = $("#stat-stage-mobile").textContent = $("#stat-stage-vith").textContent = stageLabel;
  $("#target-count").textContent = number(dataset.targets);
  $("#image-count").textContent = number(dataset.images) + " valid images";
}
function wireBenchmarks() {
  $$('[data-dataset]').forEach(button=>button.addEventListener('click',()=>{state.dataset=button.dataset.dataset;active($$('[data-dataset]'),button);renderBenchmarks();}));
  $$('[data-benchmark-prompt]').forEach(button=>button.addEventListener('click',()=>{state.benchmarkPrompt=button.dataset.benchmarkPrompt;active($$('[data-benchmark-prompt]'),button);renderBenchmarks();}));
  $$('[data-benchmark-stage]').forEach(button=>button.addEventListener('click',()=>{state.benchmarkStage=Number(button.dataset.benchmarkStage);active($$('[data-benchmark-stage]'),button);renderBenchmarks();}));
  $("#results-table").innerHTML = ["coco","lvis","sa1b"].flatMap(key=>["tinysam","mobilesam","vith"].map(model=>{
    const data=benchmarks.datasets[key], historical=data.baseline_kind[model]==='historical_reference';
    const cells=['point','box'].flatMap(prompt=>[0,1,2].map(stage=>`<td class="paired-result-cell"><span>${data.original[model][prompt][stage].toFixed(2)} → <b>${data[model][prompt][stage].toFixed(2)}</b></span><small>+${data.gains[model][prompt][stage].toFixed(2)} pp${historical?' †':''}</small></td>`)).join('');
    return `<tr><td><b>${data.label}</b><span class="table-model">${names[model]}${historical?' †':''}</span></td>${cells}</tr>`;
  })).join('');
  $("#paper-every-table").innerHTML=paperTiming.rows.filter(row=>row.backbone==='SAM ViT-H'&&row.policy!=='Hierarchical').map(row=>`<tr><td><b>${row.policy==='Dense'?'SAM ViT-H / Dense':'FSD-SAM / ViT-H'}</b></td><td class="${row.policy==='FSD-SAM'?'ours-cell':''}">${number(row.milliseconds)} ms</td><td>${row.speedup.toFixed(2)}×</td><td>${row.ar300.toFixed(3)}%</td><td>${row.delta_ar===null?'—':row.delta_ar.toFixed(3)+' pp'}</td></tr>`).join('');
  renderBenchmarks();
}
function renderModels() {
  const release = "https://github.com/" + manifest.repository + "/releases/download/" + manifest.release + "/";
  const descriptions = {tinysam: "A compact backbone with prompt-adaptive candidate selection and local correction.", mobilesam: "A distilled image encoder paired with interaction and box refinement modules.", vith: "The full SAM ViT-H backbone with the same six-module refinement interface."};
  $("#model-cards").innerHTML = Object.entries(manifest.models).map(([name, model], index) => {
    const size = name === "vith" ? (model.bytes / 1e9).toFixed(2) : (model.bytes / 1e6).toFixed(1);
    const href = model.assets.length === 1 ? release + model.filename : config.release;
    return `<article class="card model-card"><div class="method-top"><span class="method-icon ${index === 1 ? "teal-bg" : "violet-bg"}">${svg("layers")}</span><span class="mono-label">CHECKPOINT / 0${index + 1}</span></div><h3>${names[name]}</h3><p class="model-descriptor">${descriptions[name]}</p><div class="model-size"><b>${size}</b><span>${name === "vith" ? "GB" : "MB"} / complete .pth</span></div><div class="model-features"><span>6 refinement modules</span><span>Frozen base</span><span>SHA256</span></div><pre>python download_models.py\n  --model ${name}</pre><a class="button button-light" href="${href}" target="_blank" rel="noopener">${name === "vith" ? "Get checkpoint parts" : "Download checkpoint"}${svg("download")}</a><div class="model-hash">SHA256 · ${model.sha256.slice(0, 12)}…${model.sha256.slice(-8)}</div></article>`;
  }).join("");
}

const live = {session: null, image: null, mask: null, points: [], box: null, prompt: "point", label: 1, rounds: 0, previousCount: 0, busy: false, drag: null, imageData: null};
async function api(path, body) {
  const response = await fetch("api/" + path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  let payload;
  try {payload = await response.json();} catch {throw new Error("Inference service returned an unreadable response.");}
  if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Inference request could not be completed.");
  return payload;
}
function liveStatus(text, busy = false) {
  live.busy = busy; $("#live-status").textContent = text;
  $("#run-inference").disabled = busy || !live.session || (live.prompt !== "everything" && (live.rounds >= 3 || (!live.points.length && !live.box) || (live.rounds > 0 && live.points.length === live.previousCount)));
  $("#reset-prompts").disabled = busy || !live.session;
  $("#live-model").disabled = busy;
  $("#image-upload").disabled = busy;
}
function drawLive() {
  if (!live.image) return;
  const canvas = $("#live-canvas"), context = canvas.getContext("2d");
  canvas.width = live.image.naturalWidth; canvas.height = live.image.naturalHeight;
  context.drawImage(live.image, 0, 0);
  if (live.mask) {context.globalAlpha = .5; context.drawImage(live.mask, 0, 0); context.globalAlpha = 1;}
  drawPrompts(context, live.points, live.drag ? [Math.min(live.drag.x, live.drag.endX), Math.min(live.drag.y, live.drag.endY), Math.max(live.drag.x, live.drag.endX), Math.max(live.drag.y, live.drag.endY)] : live.box, Math.max(canvas.width, canvas.height) / 640);
}
async function newLiveSession() {
  if (!live.imageData) return;
  liveStatus("Encoding image…", true);
  try {
    const data = await api("session", {image: live.imageData, model: $("#live-model").value});
    live.session = data.session_id; live.points = []; live.box = null; live.mask = null; live.rounds = 0; live.previousCount = 0;
    drawLive(); liveStatus("Image encoded. Add a prompt to begin.");
  } catch (error) {live.session = null; liveStatus(error.message); toast(error.message);}
}
async function resetLive() {
  if (!live.session || live.busy) return;
  await api("reset", {session_id: live.session});
  live.points = []; live.box = null; live.mask = null; live.rounds = 0; live.previousCount = 0;
  drawLive(); liveStatus("Prompts reset. Select a new target.");
}
function canvasPoint(event) {
  const canvas = $("#live-canvas"), rect = canvas.getBoundingClientRect();
  return {x: Math.min(canvas.width - 1, Math.max(0, (event.clientX - rect.left) * canvas.width / rect.width)), y: Math.min(canvas.height - 1, Math.max(0, (event.clientY - rect.top) * canvas.height / rect.height))};
}
async function enableLive() {
  const health = await json("api/health");
  if (!health.live) return;
  $("#live-workspace").hidden = false; $("#live-entry").hidden = true;
  $("#live-device").textContent = health.device.toUpperCase() + " INFERENCE";
  $("#image-upload").addEventListener("change", async event => {
    const file = event.target.files[0];
    if (!file) return;
    if (file.size > 15 * 1024 * 1024) {toast("Choose an image smaller than 15 MB."); return;}
    const url = URL.createObjectURL(file);
    try {
      const source = await image(url), canvas = document.createElement("canvas");
      const scale = Math.min(1, 1800 / Math.max(source.naturalWidth, source.naturalHeight));
      canvas.width = Math.round(source.naturalWidth * scale); canvas.height = Math.round(source.naturalHeight * scale);
      canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height);
      live.imageData = canvas.toDataURL("image/jpeg", .92);
      live.image = await image(live.imageData);
      $("#upload-placeholder").hidden = true;
      drawLive(); await newLiveSession();
    } catch (error) {toast("The image could not be opened.");}
    finally {URL.revokeObjectURL(url); pictures.delete(url);}
  });
  $("#live-model").addEventListener("change", () => newLiveSession());
  $$("[data-live-prompt]").forEach(button => button.addEventListener("click", async () => {
    if (live.busy) return;
    await resetLive(); live.prompt = button.dataset.livePrompt; active($$("[data-live-prompt]"), button);
    $("#live-every-options").hidden = live.prompt !== "everything";
    $("#live-label").hidden = live.prompt === "everything";
    $("#live-instruction").textContent = live.prompt === "everything" ? "No clicks required. Run automatic generation to segment instances across the image." : live.prompt === "box" ? "Drag a box around the target. After prediction, add corrective clicks." : "Click on the target. Add a foreground click to refine the result.";
    liveStatus("Mode changed. Ready for segmentation.");
  }));
  $$("[data-label]").forEach(button => button.addEventListener("click", () => {live.label = Number(button.dataset.label); active($$("[data-label]"), button);}));
  const canvas = $("#live-canvas");
  canvas.addEventListener("pointerdown", event => {
    if (!live.session || live.busy || live.rounds >= 3 || live.prompt === "everything") return;
    const point = canvasPoint(event);
    if (live.prompt === "box" && live.rounds === 0) {
      live.drag = {...point, endX: point.x, endY: point.y}; canvas.setPointerCapture(event.pointerId);
    } else {
      if (live.points.length > live.previousCount) {toast("Predict this prompt before adding another click."); return;}
      if (!live.rounds && !live.label) {toast("Begin with a foreground click."); return;}
      live.points.push({...point, label: live.label}); drawLive(); liveStatus("Prompt added. Ready to predict.");
    }
  });
  canvas.addEventListener("pointermove", event => {
    if (!live.drag) return;
    const point = canvasPoint(event); live.drag.endX = point.x; live.drag.endY = point.y; drawLive();
  });
  canvas.addEventListener("pointerup", event => {
    if (!live.drag) return;
    const point = canvasPoint(event);
    const box = [Math.min(live.drag.x, point.x), Math.min(live.drag.y, point.y), Math.max(live.drag.x, point.x), Math.max(live.drag.y, point.y)];
    live.drag = null;
    if (box[2] - box[0] < 3 || box[3] - box[1] < 3) {drawLive(); toast("Drag to define a box around the target."); return;}
    live.box = box; drawLive(); liveStatus("Box defined. Ready to predict.");
  });
  canvas.addEventListener("pointercancel", () => {live.drag = null; drawLive();});
  $("#run-inference").addEventListener("click", async () => {
    liveStatus("Predicting mask…", true);
    try {
      if (live.prompt === "everything") {
        const data = await api("everything", {session_id:live.session, grid:Number($("#live-every-grid").value), method:$("#live-every-method").value});
        live.mask = await opaqueMask(data.mask); drawLive();
        liveStatus(`${data.masks} instances · ${data.total_points} prompts · ${data.native_points} completed requests · ${data.seconds.toFixed(2)} s`);
        return;
      }
      const data = await api("predict", {session_id: live.session, points: live.points, box: live.box});
      live.mask = await opaqueMask(data.mask); live.rounds = data.round; live.previousCount = live.points.length;
      drawLive(); liveStatus(`Round ${data.round} · candidate ${data.candidate} · ${data.seconds.toFixed(2)} s${data.round === 3 ? " · complete; reset to select another object" : " · add a corrective click to continue"}`);
    } catch (error) {liveStatus(error.message); toast(error.message);}
  });
  $("#reset-prompts").addEventListener("click", () => resetLive().catch(error => toast(error.message)));
}
async function start() {
  [records, benchmarks, manifest, config, everything, everyComparison, paperTiming] = await Promise.all([json("examples.json"), json("benchmarks.json"), json("models.json"), json("config.json"), json("everything.json"), json("every-comparison.json"), json("paper-everything.json")]);
  const parameters = new URLSearchParams(location.search);
  if (parameters.get("embed") === "showcase") document.body.classList.add("embedded-explorer");
  if (["tinysam","mobilesam"].includes(parameters.get("model"))) state.model=parameters.get("model");
  if (["point","box","everything"].includes(parameters.get("prompt"))) state.prompt=parameters.get("prompt");
  active($$("[data-prompt]"),$(`[data-prompt="${state.prompt}"]`));
  $("#example-model").value=state.model;
  wireExamples(); wireBenchmarks(); renderModels(); wireEverything(); await renderExample();
  $("#copy-quickstart").addEventListener("click", () => copy($("#quickstart-code").textContent.trim()));
  $(".dialog-close").addEventListener("click", () => $("#local-demo-dialog").close());
  $("#open-live").addEventListener("click", () => {
    if (config.space_url && /^https:\/\/huggingface\.co\/spaces\//.test(config.space_url)) window.open(config.space_url, "_blank", "noopener");
    else $("#local-demo-dialog").showModal();
  });
  if (document.body.dataset.live === "true") {
    if (new URLSearchParams(location.search).get("embed") === "1") document.body.classList.add("embedded-live");
    await enableLive();
  }
}
function setEveryControls(enabled) {
  $("#round-controls-title").hidden = enabled;
  $(".round-buttons").hidden = enabled;
  $(".prompt-key").hidden = enabled;
  $("#every-instance-control").hidden = true;
  $("#every-result-view").hidden = !enabled;
  $("#every-comparison-actions").hidden = !enabled;
  $("#example-model").closest("label").hidden = enabled;
  $(".opacity-control").hidden = enabled;
  $("#refined-canvas").hidden = false;
  $("#prediction-grid").classList.remove("is-flow");
}
async function renderEverything(version) {
  const pair = everyComparison.examples.fruit;
  $("#every-result-view .pill").textContent = `${pair.grid} × ${pair.grid} · ${number(pair.grid*pair.grid)} prompt points`;
  state.example = "fruit";
  $("#example-list").innerHTML = '<button class="example-option active" aria-pressed="true"><img src="assets/examples/fruit.jpg" alt=""><span><b>Orange bowl</b><small>Same image · same ViT-H</small></span></button>';
  $("#example-list").dataset.mode = "everything";
  $("#reference-heading").textContent = "SAM ViT-H / Dense";
  $("#refined-heading").textContent = "FSD-SAM / ViT-H";
  $("#reference-score").textContent = state.pairShown ? `${pair.results.dense.total_ms.toFixed(1)} ms` : "Ready";
  $("#refined-score").textContent = state.pairShown ? `${pair.results.fsd.total_ms.toFixed(1)} ms` : "Ready";
  $(".explorer-note").textContent = "Both paths use the same frozen ViT-H weights, image, grid and SAM filters. Recorded demo inference; use the live studio for your own image.";
  $("#example-caption").textContent = state.pairShown ? `${pair.device} · FP32 · ${pair.grid} × ${pair.grid} grid · ${number(pair.grid*pair.grid)} points` : `One button reveals both measured ${pair.grid} × ${pair.grid} results.`;
  $("#example-metric-note").textContent = "Recorded comparison · encode + masks";
  $("#trajectory-heading").textContent = "SAME IMAGE / SAME BACKBONE";
  $("#example-trajectory").innerHTML = ["dense","fsd"].map(method => {
    const row = pair.results[method];
    return `<div class="trajectory-value"><span>${method === "dense" ? "SAM ViT-H" : "FSD-SAM"}</span><b>${state.pairShown ? row.total_ms.toFixed(1)+" ms" : "Ready"}</b><small>${state.pairShown ? `${row.info.masks} masks · ${row.info.native_points}/${row.info.total_points} completed` : "Press Show comparison"}</small></div>`;
  }).join("");
  $("#every-comparison-note").textContent = state.pairShown ? `Measured image encoding (${pair.results.dense.encoding_ms.toFixed(1)} ms) is counted in both totals; queue, warm-up, loading and overlay rendering are excluded. This recorded demo is not a benchmark speedup claim.` : "Recorded real inference. Run both models in the live demo for new measurements.";
  for (const [field,method] of [["reference","dense"],["refined","fsd"]]) {
    const photo = await image(state.pairShown ? pair.results[method].overlay : pair.image);
    if(version !== renderVersion)return;
    const canvas=$(`#${field}-canvas`);
    canvas.width=photo.naturalWidth;canvas.height=photo.naturalHeight;
    canvas.getContext("2d").drawImage(photo,0,0);
    canvas.setAttribute("aria-label",state.pairShown?`${method === "dense" ? "Dense ViT-H" : "FSD-SAM"} measured instance masks`:"Input image before comparison");
  }
  $("#image-source").href="assets/credits.md";$("#image-source").hidden=false;
}
function wireEverything() {
  $("#run-every-comparison").addEventListener("click",()=>{
    state.pairShown=true;
    renderExample().catch(error=>toast(error.message));
  });
}
start().catch(error => {toast(error.message); $("#example-caption").textContent = error.message; console.error(error);});
