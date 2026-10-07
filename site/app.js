"use strict";
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const names = {tinysam: "TinySAM", mobilesam: "MobileSAM", vith: "SAM ViT-H"};
const state = {example: null, model: "tinysam", prompt: "point", round: 1, opacity: .5, dataset: "coco", benchmarkPrompt: "point", everyView: "flow"};
const pictures = new Map();
let records, benchmarks, manifest, config, everything, renderVersion = 0, toastTimer;
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
      const label = gain !== null && gain > 1 ? `<em class="gain-tag">+${gain.toFixed(1)} pp · first round</em>` : "";
      return `<button class="example-option" data-example="${row.id}" aria-pressed="false"><img src="${row.image}" alt="" width="60" height="60"><span><b>${row.title}</b><small>${row.dataset} · Image ${row.image_id}</small>${label}</span>${svg("arrow")}</button>`;
    }).join("");
    list.dataset.mode = key;
  }
  active($$("[data-example]"),$(`[data-example="${state.example}"]`));
}
async function renderExample() {
  const version = ++renderVersion;
  refreshExampleOptions();
  setEveryControls(state.prompt === "everything");
  if (state.prompt === "everything") return renderEverything(version);
  $("#reference-heading").textContent = "Original " + names[state.model];
  $("#refined-heading").textContent = "Prompt-adaptive " + names[state.model];
  $("#refined-canvas").hidden = false;
  $("#fsd-workflow-panel").hidden = true;
  $("#prediction-grid").classList.remove("is-flow");
  $("#reference-canvas").setAttribute("aria-label","Original lightweight model prediction with the same prompt");
  $("#refined-canvas").setAttribute("aria-label","Refined model mask and prompts");
  $("#example-metric-note").textContent = "Measured legacy IoU · curated example";
  $("#trajectory-heading").textContent = "REFINED TRAJECTORY";
  $(".explorer-note").textContent = "Original lightweight checkpoint vs. prompt-adaptive refinement, with identical prompts. Each method keeps its own mask feedback. IoU is measured against the COCO annotation; these are curated individual examples.";
  const sample = records.examples.find(item => item.id === state.example);
  const trajectory = sample.trajectories[state.model][state.prompt];
  const round = trajectory[state.round - 1];
  $("#example-caption").textContent = `${sample.dataset} · image ${sample.image_id} · object ${sample.annotation_id} · round ${state.round}`;
  $("#reference-score").textContent = round.reference.iou_percent.toFixed(2) + "%";
  $("#refined-score").textContent = round.refined.iou_percent.toFixed(2) + "%";
  $("#example-trajectory").innerHTML = trajectory.map(row => `<div class="trajectory-value${row.round === state.round ? " active" : ""}"><span>ROUND 0${row.round}</span><b>${row.refined.iou_percent.toFixed(2)}%</b><small>Original ${row.reference.iou_percent.toFixed(2)}%</small></div>`).join("");
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
  const dataset = benchmarks.datasets[state.dataset];
  const prompt = state.benchmarkPrompt;
  const tiny = dataset.tinysam[prompt], mobile = dataset.mobilesam[prompt];
  const lower = Math.max(0, Math.floor(Math.min(...tiny, ...mobile) / 5) * 5 - 5);
  const upper = Math.min(100, Math.ceil(Math.max(...tiny, ...mobile) / 5) * 5 + 5);
  const y = value => 225 - (value - lower) / (upper - lower) * 188;
  const xs = [75, 307, 539];
  let markup = `<defs><linearGradient id="violet-chart" x1="0" x2="0" y1="0" y2="1"><stop stop-color="#7860de" stop-opacity=".13"/><stop offset="1" stop-color="#7860de" stop-opacity="0"/></linearGradient></defs>`;
  for (let step = 0; step <= 4; step++) {
    const value = lower + (upper - lower) * step / 4, position = y(value);
    markup += `<line x1="55" y1="${position}" x2="558" y2="${position}" stroke="#e9ebf2" stroke-dasharray="3 5"/><text x="38" y="${position + 4}" font-size="10" text-anchor="end" fill="#a9afbc">${value.toFixed(0)}</text>`;
  }
  const points = values => values.map((value, index) => `${xs[index]},${y(value)}`).join(" ");
  markup += `<polygon points="75,225 ${points(tiny)} 539,225" fill="url(#violet-chart)"/>`;
  for (const [values, color, offset] of [[tiny, "#7860de", -15], [mobile, "#268e90", 26]]) {
    markup += `<polyline points="${points(values)}" stroke="${color}" stroke-width="2.5" fill="none"/>`;
    values.forEach((value, index) => {
      markup += `<circle cx="${xs[index]}" cy="${y(value)}" r="4.5" fill="white" stroke="${color}" stroke-width="2"/><text x="${xs[index]}" y="${y(value) + offset}" font-size="11" font-weight="600" text-anchor="middle" fill="${color}">${value.toFixed(3)}</text>`;
    });
  }
  xs.forEach((x, index) => {markup += `<text x="${x}" y="259" font-size="10" text-anchor="middle" fill="#a2a9b7">ROUND 0${index + 1}</text>`;});
  $("#benchmark-chart").innerHTML = markup;
  $("#benchmark-chart").setAttribute("aria-label", `${dataset.label}, ${prompt} prompt legacy mIoU across three rounds. TinySAM: ${tiny.join(", ")}. MobileSAM: ${mobile.join(", ")}.`);
  $("#chart-title").textContent = dataset.label;
  $("#chart-coverage").textContent = dataset.coverage;
  $("#stat-prompt").textContent = $("#stat-prompt-mobile").textContent = prompt.toUpperCase();
  $("#tiny-result").innerHTML = tiny[2].toFixed(3) + "<span>%</span>";
  $("#mobile-result").innerHTML = mobile[2].toFixed(3) + "<span>%</span>";
  $("#tiny-gain").textContent = `+${(tiny[2] - tiny[0]).toFixed(3)} pp across rounds`;
  $("#mobile-gain").textContent = `+${(mobile[2] - mobile[0]).toFixed(3)} pp across rounds`;
  $("#target-count").textContent = number(dataset.targets);
  $("#image-count").textContent = number(dataset.images) + " valid images";
}
function wireBenchmarks() {
  $$("[data-dataset]").forEach(button => button.addEventListener("click", () => {
    state.dataset = button.dataset.dataset; active($$("[data-dataset]"), button); renderBenchmarks();
  }));
  $$("[data-benchmark-prompt]").forEach(button => button.addEventListener("click", () => {
    state.benchmarkPrompt = button.dataset.benchmarkPrompt; active($$("[data-benchmark-prompt]"), button); renderBenchmarks();
  }));
  $("#results-table").innerHTML = ["sa1b", "coco", "lvis"].map(key => {
    const data = benchmarks.datasets[key];
    return `<tr><td>${data.label}</td>${[data.tinysam.point[2], data.tinysam.box[2], data.mobilesam.point[2], data.mobilesam.box[2]].map(value => `<td>${value.toFixed(3)}</td>`).join("")}</tr>`;
  }).join("");
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
    $("#live-instruction").textContent = live.prompt === "everything" ? "No clicks required. Run automatic generation to segment instances across the image." : live.prompt === "box" ? "Drag a box around the target. After prediction, add corrective clicks." : "Click on the target. Add a background click to exclude a region.";
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
  [records, benchmarks, manifest, config, everything] = await Promise.all([json("examples.json"), json("benchmarks.json"), json("models.json"), json("config.json"), json("everything.json")]);
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
  $("#every-instance-control").hidden = !enabled || state.everyView !== "masks";
  $("#every-result-view").hidden = !enabled;
}
async function renderEverything(version) {
  const sample = records.examples.find(row => row.id === state.example);
  const record = everything.examples[state.example][state.model];
  const selector = $("#every-instance");
  const key = state.example + ":" + state.model;
  if (selector.dataset.record !== key) {
    selector.innerHTML = '<option value="all">All instances</option>' + record.instances.map((row,index) => `<option value="${index}">Instance ${String(index+1).padStart(2,"0")} · ${number(row.area)} px</option>`).join("");
    selector.dataset.record = key;
  }
  const selected = selector.value === "all" ? null : Number(selector.value);
  const isFlow = state.everyView === "flow";
  $("#reference-heading").textContent = "Automatic sampling";
  $("#refined-heading").textContent = isFlow ? "FSD decoding walkthrough" : selected === null ? "FSD-SAM instances" : `Instance ${String(selected+1).padStart(2,"0")}`;
  $("#reference-score").textContent = `${record.info.grid} × ${record.info.grid}`;
  $("#refined-score").textContent = isFlow ? `${record.info.native_points} / ${record.info.total_points} requests` : selected === null ? `${record.instances.length} masks` : `${number(record.instances[selected].area)} px`;
  $("#refined-canvas").hidden = isFlow;
  $("#fsd-workflow-panel").hidden = !isFlow;
  $("#prediction-grid").classList.toggle("is-flow", isFlow);
  const frameURL = `assets/fsd-workflow.html?total=${record.info.total_points}&native=${record.info.native_points}&guard=${record.info.local_guard_points}&masks=${record.info.masks}`;
  const frame = $("#fsd-workflow-frame");
  if (frame.getAttribute("src") !== frameURL) frame.src = frameURL;
  $("#example-metric-note").textContent = "Prompt-free · recorded CPU inference";
  $("#trajectory-heading").textContent = "GENERATION SUMMARY";
  $(".explorer-note").textContent = "FSD-SAM with the frozen backbone. Colors distinguish instances, not semantic classes. No annotated prompts are used.";
  $("#example-caption").textContent = `${sample.dataset} · image ${sample.image_id} · Everything · ${names[state.model]}`;
  $("#example-trajectory").innerHTML = [["SAMPLED PROMPTS",record.info.total_points],["NATIVE REQUESTS",record.info.native_points],["INSTANCES",record.info.masks]].map(([label,value]) => `<div class="trajectory-value"><span>${label}</span><b>${number(value)}</b></div>`).join("");
  $("#every-download").href = record.predictions;
  const photo = await image(sample.image);
  const original = $("#reference-canvas"), context = original.getContext("2d");
  original.width = sample.width; original.height = sample.height;
  original.setAttribute("aria-label","Original image with automatic sampling grid");
  context.drawImage(photo,0,0);
  context.fillStyle = "#ffffffdd";
  context.strokeStyle = "#378e97dd"; context.lineWidth = 1;
  for (let y=0;y<record.info.grid;y++) for (let x=0;x<record.info.grid;x++) {
    context.beginPath(); context.arc((x+.5)*sample.width/record.info.grid,(y+.5)*sample.height/record.info.grid,2.4,0,Math.PI*2); context.fill(); context.stroke();
  }
  const canvas = $("#refined-canvas"), target = canvas.getContext("2d");
  canvas.width = sample.width; canvas.height = sample.height;
  canvas.setAttribute("aria-label",selected===null?"Automatic instance masks":"Selected automatic instance mask");
  target.drawImage(photo,0,0);
  const instances = record.instances.map((row,index) => ({...row,index})).filter(row => selected===null || selected===row.index).sort((a,b) => b.area-a.area);
  const loaded = await Promise.all(instances.map(async row => ({row,mask:await image(row.mask)})));
  if (version !== renderVersion) return;
  target.globalAlpha = state.opacity;
  for (const item of loaded) target.drawImage(item.mask,0,0);
  target.globalAlpha = 1;
  $("#image-source").href = "assets/credits.md";
  $("#image-source").hidden = false;
}
function wireEverything() {
  $$("[data-every-view]").forEach(button => button.addEventListener("click", () => {
    state.everyView = button.dataset.everyView;
    active($$("[data-every-view]"), button);
    renderExample().catch(error => toast(error.message));
  }));
  $("#every-instance").addEventListener("change", () => renderExample().catch(error => toast(error.message)));
  $("#try-everything").addEventListener("click", () => {
    state.prompt = "everything";
    state.example = "fruit";
    active($$("[data-prompt]"),$("[data-prompt=everything]"));
    $("#explorer").scrollIntoView({behavior:"smooth"});
    renderExample().catch(error => toast(error.message));
  });
}
start().catch(error => {toast(error.message); $("#example-caption").textContent = error.message; console.error(error);});
