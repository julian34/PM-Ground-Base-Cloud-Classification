// CloudDenseNet — frontend: upload citra, panggil API, render analisis.
"use strict";

const $ = (id) => document.getElementById(id);
const el = {
  checkpoint: $("checkpoint"), gradcam: $("gradcam"),
  drop: $("drop"), file: $("file"), dropEmpty: $("dropEmpty"),
  previewWrap: $("previewWrap"), preview: $("preview"),
  stageActions: $("stageActions"), clear: $("clear"),
  empty: $("empty"), resultBody: $("resultBody"),
  topName: $("topName"), topConf: $("topConf"), bars: $("bars"),
  camWrap: $("camWrap"), cam: $("cam"), meta: $("meta"),
  loading: $("loading"), error: $("error"), errorMsg: $("errorMsg"),
};

// Warna identitas per kelas (indeks 0..6, urutan CLASS_NAMES).
// Membawa makna: cumulonimbus (badai) = merah, clearsky = teal.
const PALETTE = ["#2563eb", "#0891b2", "#7c3aed", "#0d9488", "#d97706", "#dc2626", "#64748b"];

let currentFile = null;
let hasResult = false;

// Baca respons sebagai JSON dengan aman. Bila body bukan JSON (mis. 500 teks
// polos "Internal Server Error"), lempar pesan yang terbaca — bukan error
// "Unexpected token 'I'" dari JSON.parse.
async function bacaJson(res) {
  const teks = await res.text();
  try {
    return JSON.parse(teks);
  } catch {
    const cuplik = teks.trim().slice(0, 160) || res.statusText;
    throw new Error(`Server membalas ${res.status} (${res.statusText}): ${cuplik}`);
  }
}

// --- Checkpoint ------------------------------------------------------------
async function loadCheckpoints() {
  try {
    const res = await fetch("/api/checkpoints");
    const data = await bacaJson(res);
    el.checkpoint.innerHTML = "";
    if (!data.checkpoints || data.checkpoints.length === 0) {
      const opt = new Option("tidak ada checkpoint .pth", "");
      opt.disabled = true;
      el.checkpoint.add(opt);
      showError("Tidak ada checkpoint di artifacts/checkpoints/. Latih model dulu.");
      return;
    }
    for (const ck of data.checkpoints) {
      el.checkpoint.add(new Option(ck, ck, ck === data.default, ck === data.default));
    }
  } catch (e) {
    showError("Gagal memuat daftar checkpoint: " + e.message);
  }
}

// --- Dropzone & file -------------------------------------------------------
el.drop.addEventListener("click", () => el.file.click());
el.drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.file.click(); }
});
el.file.addEventListener("change", () => { if (el.file.files.length) setFile(el.file.files[0]); });
el.clear.addEventListener("click", resetFile);

["dragenter", "dragover"].forEach((ev) =>
  el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.remove("drag"); }));
el.drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) setFile(f);
});

function setFile(file) {
  if (!file.type.startsWith("image/")) { showError("Berkas yang dipilih bukan gambar."); return; }
  hideError();
  currentFile = file;
  el.preview.src = URL.createObjectURL(file);
  el.dropEmpty.hidden = true;
  el.previewWrap.hidden = false;
  el.stageActions.hidden = false;
  predict();
}

function resetFile() {
  currentFile = null; hasResult = false;
  el.file.value = "";
  el.previewWrap.hidden = true;
  el.dropEmpty.hidden = false;
  el.stageActions.hidden = true;
  el.resultBody.hidden = true;
  el.empty.hidden = false;
  hideError();
}

el.checkpoint.addEventListener("change", () => currentFile && predict());
el.gradcam.addEventListener("change", () => currentFile && predict());

// --- Prediksi --------------------------------------------------------------
async function predict() {
  if (!currentFile) return;
  hideError();
  el.empty.hidden = true;
  el.loading.hidden = false;
  if (!hasResult) renderSkeleton();

  const fd = new FormData();
  fd.append("file", currentFile);
  fd.append("checkpoint", el.checkpoint.value || "");
  fd.append("gradcam", el.gradcam.checked ? "true" : "false");

  try {
    const res = await fetch("/api/predict", { method: "POST", body: fd });
    const data = await bacaJson(res);
    if (!res.ok) throw new Error(data.detail || "Prediksi gagal.");
    render(data);
    hasResult = true;
  } catch (e) {
    if (!hasResult) { el.resultBody.hidden = true; el.empty.hidden = false; }
    showError(e.message);
  } finally {
    el.loading.hidden = true;
  }
}

const pct = (x) => (x * 100).toFixed(x >= 0.1 ? 1 : 2) + "%";

function renderSkeleton() {
  el.resultBody.hidden = false;
  el.camWrap.hidden = true;
  el.meta.innerHTML = "";
  el.topName.textContent = "—";
  el.topConf.textContent = "··%";
  el.bars.innerHTML = "";
  for (let i = 0; i < 7; i++) {
    const row = document.createElement("div");
    row.className = "bar-row skeleton";
    row.setAttribute("role", "listitem");
    row.innerHTML =
      `<span class="dot"></span><span class="bar-name">····</span>` +
      `<span class="bar-track shimmer"></span><span class="bar-val">··</span>`;
    el.bars.appendChild(row);
  }
}

function render(data) {
  el.empty.hidden = true;
  el.resultBody.hidden = false;
  el.topName.textContent = data.top.nama;
  el.topConf.textContent = pct(data.top.confidence);

  el.bars.innerHTML = "";
  data.probs.forEach((p, i) => {
    const row = document.createElement("div");
    row.className = "bar-row" + (i === 0 ? " top" : "");
    row.setAttribute("role", "listitem");
    const color = PALETTE[p.indeks] || "var(--muted)";
    row.innerHTML =
      `<span class="dot" style="background:${color}"></span>` +
      `<span class="bar-name" title="${p.kelas}">${p.nama}</span>` +
      `<span class="bar-track"><span class="bar-fill"></span></span>` +
      `<span class="bar-val">${pct(p.confidence)}</span>`;
    el.bars.appendChild(row);
    const fill = row.querySelector(".bar-fill");
    if (i === 0) fill.style.background = "var(--signal)"; else fill.style.background = color;
    fill.style.opacity = i === 0 ? "1" : "0.55";
    requestAnimationFrame(() => requestAnimationFrame(() => {
      fill.style.width = (p.confidence * 100).toFixed(2) + "%";
    }));
  });

  if (data.overlay) { el.cam.src = data.overlay; el.camWrap.hidden = false; }
  else el.camWrap.hidden = true;

  el.meta.innerHTML =
    `<span class="chip"><b>ckpt</b> ${data.checkpoint}</span>` +
    `<span class="chip"><b>perangkat</b> ${data.device.toUpperCase()}</span>`;
}

function showError(msg) { el.errorMsg.textContent = msg; el.error.hidden = false; }
function hideError() { el.error.hidden = true; }

loadCheckpoints();
