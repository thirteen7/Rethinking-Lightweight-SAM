"use strict";
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const names = {tinysam: "TinySAM", mobilesam: "MobileSAM", vith: "SAM ViT-H"};
const state = {example: "bear", model: "tinysam", prompt: "point", round: 1, opacity: .5, dataset: "coco", benchmarkPrompt: "point"};
const pictures = new Map();
let records, benchmarks, manifest, config, renderVersion = 0, toastTimer;
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
async function renderExample() {
  const version = ++renderVersion;
  const sample = records.examples.find(item => item.id === state.example);
  const trajectory = sample.trajectories[state.model][state.prompt];
  const round = trajectory[state.round - 1];
  $("#example-caption").textContent = `${sample.dataset} · image ${sample.image_id} · object ${sample.annotation_id} · round ${state.round}`;
  $("#reference-score").textContent = round.reference.iou_percent.toFixed(2) + "%";
  $("#refined-score").textContent = round.refined.iou_percent.toFixed(2) + "%";
  $("#example-trajectory").innerHTML = trajectory.map(row => `<div class="trajectory-value${row.round === state.round ? " active" : ""}"><span>ROUND 0${row.round}</span><b>${row.refined.iou_percent.toFixed(2)}%</b></div>`).join("");
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
  $("#example-model").addEventListener("change", event => {state.model = event.target.value; renderExample().catch(error => toast(error.message));});
  $$("[data-example]").forEach(button => button.addEventListener("click", () => {
    state.example = button.dataset.example; active($$("[data-example]"), button); renderExample().catch(error => toast(error.message));
  }));
  $$("[data-prompt]").forEach(button => button.addEventListener("click", () => {
    state.prompt = button.dataset.prompt; active($$("[data-prompt]"), button); renderExample().catch(error => toast(error.message));
  }));
  $$("[data-round]").forEach(button => button.addEventListener("click", () => {
    state.round = Number(button.dataset.round); active($$("[data-round]"), button); renderExample().catch(error => toast(error.message));
  }));
  $("#mask-opacity").addEventListener("input", event => {
    state.opacity = Number(event.target.value) / 100; $("#opacity-value").textContent = event.target.value + "%";
    renderExample().catch(error => toast(error.message));
  });
  const first = records.examples[0].trajectories.tinysam.point[0];
  $("#hero-reference").textContent = first.reference.iou_percent.toFixed(2) + "%";
  $("#hero-refined").textContent = first.refined.iou_percent.toFixed(2) + "%";
  $$(".hero-prompt").forEach(marker => {
    marker.style.left = (100 * first.points[0].x / records.examples[0].width) + "%";
    marker.style.top = (100 * first.points[0].y / records.examples[0].height) + "%";
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
  $("#run-inference").disabled = busy || !live.session || live.rounds >= 3 || (!live.points.length && !live.box) || (live.rounds > 0 && live.points.length === live.previousCount);
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
    $("#live-instruction").textContent = live.prompt === "box" ? "Drag a box around the target. After prediction, add corrective clicks." : "Click on the target. Add a background click to exclude a region.";
  }));
  $$("[data-label]").forEach(button => button.addEventListener("click", () => {live.label = Number(button.dataset.label); active($$("[data-label]"), button);}));
  const canvas = $("#live-canvas");
  canvas.addEventListener("pointerdown", event => {
    if (!live.session || live.busy || live.rounds >= 3) return;
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
      const data = await api("predict", {session_id: live.session, points: live.points, box: live.box});
      live.mask = await opaqueMask(data.mask); live.rounds = data.round; live.previousCount = live.points.length;
      drawLive(); liveStatus(`Round ${data.round} · candidate ${data.candidate} · ${data.seconds.toFixed(2)} s${data.round === 3 ? " · complete; reset to select another object" : " · add a corrective click to continue"}`);
    } catch (error) {liveStatus(error.message); toast(error.message);}
  });
  $("#reset-prompts").addEventListener("click", () => resetLive().catch(error => toast(error.message)));
}
async function start() {
  [records, benchmarks, manifest, config] = await Promise.all([json("examples.json"), json("benchmarks.json"), json("models.json"), json("config.json")]);
  wireExamples(); wireBenchmarks(); renderModels(); await renderExample();
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
start().catch(error => {toast(error.message); $("#example-caption").textContent = error.message; console.error(error);});
