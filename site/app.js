const MODEL_META = {
  "578K": { file: "models/578K.json?v=2", parameters: "578,864", description: "A 183-wide tied-embedding GRU for quick inline suggestions." },
  "2.4M": { file: "models/2.4M.json?v=2", parameters: "2,429,256", description: "A 380-wide recurrent model for a broader local context." },
  "6.0M": { file: "models/6.0M.json?v=2", parameters: "6,038,532", description: "A 530-wide recurrent model for the fullest preview capacity." }
};

const editor = document.querySelector("#editor");
const mirror = document.querySelector(".editor-mirror");
const typedCopy = document.querySelector("#typed-copy");
const ghostCopy = document.querySelector("#ghost-copy");
const modelSelect = document.querySelector("#model-select");
const inspectorModel = document.querySelector("#inspector-model");
const temperature = document.querySelector("#temperature");
const topK = document.querySelector("#top-k");
const suggestionTray = document.querySelector("#suggestion-tray");
const suggestionText = document.querySelector("#suggestion-text");
const suggestionModel = document.querySelector("#suggestion-model");
const suggestionConfidence = document.querySelector("#suggestion-confidence");
const liveRegion = document.querySelector("#live-region");
const workerError = document.querySelector("#worker-error");
const worker = new Worker("worker.js?v=4", { type: "module" });

let completion = "";
let requestId = 0;
let loadRequestId = 0;
let debounceHandle = 0;
let activeModel = "578K";
let modelReady = false;

function setLoadState(state, text) {
  const status = document.querySelector("#load-status");
  const statusText = document.querySelector("#load-status-text");
  status.dataset.state = state;
  statusText.textContent = text;
}

function announce(text) {
  liveRegion.textContent = "";
  window.setTimeout(() => { liveRegion.textContent = text; }, 10);
}

function updateMirror() {
  typedCopy.textContent = editor.value;
  ghostCopy.textContent = completion;
  const lineCount = editor.value.split("\n").length;
  document.querySelector("#line-gutter").textContent = Array.from({ length: lineCount }, (_, index) => index + 1).join("\n");
  const words = editor.value.trim() ? editor.value.trim().split(/\s+/u).length : 0;
  document.querySelector("#word-count").textContent = `${words} ${words === 1 ? "word" : "words"}`;
  document.querySelector("#context-count").textContent = `${editor.value.length} chars`;
  document.querySelector("#context-preview").textContent = editor.value.length > 420 ? `…${editor.value.slice(-420)}` : editor.value || "Start writing to give the model a local context window.";
  const beforeCaret = editor.value.slice(0, editor.selectionStart);
  const line = beforeCaret.split("\n").length;
  const column = beforeCaret.length - beforeCaret.lastIndexOf("\n");
  document.querySelector("#cursor-status").textContent = `Ln ${line}, Col ${column}`;
}

function clearCompletion() {
  completion = "";
  suggestionTray.hidden = true;
  ghostCopy.textContent = "";
}

function invalidateCompletion() {
  requestId += 1;
  window.clearTimeout(debounceHandle);
}

function requestCompletion() {
  if (!modelReady || !editor.value.trim()) {
    clearCompletion();
    return;
  }
  const currentRequest = requestId;
  document.querySelector("#latency-status").textContent = "Thinking locally…";
  worker.postMessage({ type: "generate", requestId: currentRequest, loadRequestId, prompt: editor.value, temperature: Number(temperature.value), topK: Number(topK.value), maxTokens: 24, seed: 7 });
}

function scheduleCompletion() {
  invalidateCompletion();
  clearCompletion();
  updateMirror();
  debounceHandle = window.setTimeout(requestCompletion, 140);
}

function syncModelControls(value) {
  invalidateCompletion();
  activeModel = value;
  const currentLoadRequest = ++loadRequestId;
  modelSelect.value = value;
  inspectorModel.value = value;
  const meta = MODEL_META[value];
  document.querySelector("#parameter-count").textContent = `${meta.parameters} params`;
  document.querySelector("#model-description").textContent = meta.description;
  clearCompletion();
  modelReady = false;
  workerError.hidden = true;
  setLoadState("loading", `Loading ${value} locally`);
  document.querySelector("#inspector-state").textContent = "Loading";
  worker.postMessage({ type: "load", model: value, url: meta.file, loadRequestId: currentLoadRequest });
  announce(`Loading the ${value} model locally`);
}

function acceptCompletion() {
  if (!completion) return;
  const start = editor.selectionStart;
  const end = editor.selectionEnd;
  editor.value = `${editor.value.slice(0, start)}${completion}${editor.value.slice(end)}`;
  const nextPosition = start + completion.length;
  editor.setSelectionRange(nextPosition, nextPosition);
  clearCompletion();
  updateMirror();
  editor.focus();
  announce("Suggestion accepted");
  scheduleCompletion();
}

function dismissCompletion() {
  invalidateCompletion();
  if (!completion) return;
  clearCompletion();
  document.querySelector("#latency-status").textContent = "Suggestion dismissed";
  announce("Suggestion dismissed");
}

modelSelect.addEventListener("change", () => syncModelControls(modelSelect.value));
inspectorModel.addEventListener("change", () => syncModelControls(inspectorModel.value));
temperature.addEventListener("input", () => {
  document.querySelector("#temperature-value").textContent = Number(temperature.value).toFixed(2);
  scheduleCompletion();
});
topK.addEventListener("input", () => {
  document.querySelector("#top-k-value").textContent = topK.value;
  scheduleCompletion();
});
editor.addEventListener("input", scheduleCompletion);
editor.addEventListener("select", updateMirror);
editor.addEventListener("keyup", updateMirror);
editor.addEventListener("scroll", () => {
  mirror.style.transform = `translateY(-${editor.scrollTop}px)`;
});
editor.addEventListener("keydown", (event) => {
  if (event.key === "Tab" && completion) {
    event.preventDefault();
    acceptCompletion();
  } else if (event.key === "Escape") {
    event.preventDefault();
    dismissCompletion();
  } else if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && completion) {
    event.preventDefault();
    acceptCompletion();
  }
});
document.querySelector("#accept-button").addEventListener("click", acceptCompletion);
function setInspectorOpen(open) {
  const button = document.querySelector("#inspector-toggle");
  const inspector = document.querySelector("#inspector");
  inspector.classList.toggle("is-open", open);
  button.setAttribute("aria-expanded", String(open));
  button.setAttribute("aria-label", open ? "Hide model inspector" : "Show model inspector");
}

document.querySelector("#inspector-toggle").addEventListener("click", () => {
  const button = document.querySelector("#inspector-toggle");
  const inspector = document.querySelector("#inspector");
  const open = inspector.classList.toggle("is-open");
  button.setAttribute("aria-expanded", String(open));
  button.setAttribute("aria-label", open ? "Hide model inspector" : "Show model inspector");
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && document.querySelector("#inspector").classList.contains("is-open")) {
    setInspectorOpen(false);
    document.querySelector("#inspector-toggle").focus();
  }
});

worker.addEventListener("message", (event) => {
  const message = event.data;
  if (message.type === "model-ready") {
    if (message.model !== activeModel || message.loadRequestId !== loadRequestId) return;
    modelReady = true;
    workerError.hidden = true;
    setLoadState("ready", "Ready locally");
    document.querySelector("#inspector-state").textContent = "Local";
    document.querySelector("#latency-status").textContent = "Model ready";
    announce(`${activeModel} model ready in this tab`);
    requestCompletion();
    return;
  }
  if (message.type === "completion" && message.requestId === requestId && message.loadRequestId === loadRequestId && message.model === activeModel) {
    completion = message.completion;
    ghostCopy.textContent = completion;
    suggestionText.textContent = completion || "No continuation for this context yet.";
    suggestionModel.textContent = activeModel;
    suggestionConfidence.textContent = completion ? `${message.elapsedMs.toFixed(0)} ms local inference` : "No continuation";
    suggestionTray.hidden = !completion;
    document.querySelector("#latency-status").textContent = completion ? `${message.elapsedMs.toFixed(0)} ms local` : "No suggestion yet";
    if (completion) announce(`Suggestion available from ${activeModel}`);
    return;
  }
  if (message.type === "error") {
    if (message.loadRequestId !== undefined && message.loadRequestId !== loadRequestId) return;
    if (message.requestId !== undefined && message.requestId !== requestId) return;
    modelReady = false;
    setLoadState("error", "Model unavailable");
    document.querySelector("#inspector-state").textContent = "Error";
    workerError.hidden = false;
    workerError.textContent = message.message;
    document.querySelector("#latency-status").textContent = "Model error";
    announce("The local model could not be loaded");
  }
});

function handleWorkerFailure(message) {
  modelReady = false;
  setLoadState("error", "Model unavailable");
  document.querySelector("#inspector-state").textContent = "Error";
  workerError.hidden = false;
  workerError.textContent = message;
  document.querySelector("#latency-status").textContent = "Model error";
  announce("The local model could not be loaded");
}

worker.addEventListener("error", (event) => handleWorkerFailure(event.message || "The model worker could not start"));
worker.addEventListener("messageerror", () => handleWorkerFailure("The model worker returned an unreadable message"));

window.addEventListener("beforeunload", () => worker.terminate());
updateMirror();
syncModelControls(activeModel);
