// CloudDenseNet — frontend batch: pilih satu folder, proses per-potong, isi tabel.
"use strict";

const $ = (id) => document.getElementById(id);
const el = {
  checkpoint: $("checkpoint"),
  drop: $("drop"), folder: $("folder"), files: $("files"),
  pickFolder: $("pickFolder"), pickFiles: $("pickFiles"),
  selCount: $("selCount"),
  runBar: $("runBar"), run: $("run"), reset: $("reset"), csv: $("csv"),
  progress: $("progress"), progressFill: $("progressFill"), progressLabel: $("progressLabel"),
  summaryPanel: $("summaryPanel"), summaryStats: $("summaryStats"), summaryDist: $("summaryDist"),
  empty: $("empty"), tableScroll: $("tableScroll"), rows: $("rows"),
  tableMeta: $("tableMeta"), error: $("error"), errorMsg: $("errorMsg"),
};

// Warna identitas per kelas (indeks 0..6, urutan CLASS_NAMES) — selaras app.js.
const PALETTE = ["#2563eb", "#0891b2", "#7c3aed", "#0d9488", "#d97706", "#dc2626", "#64748b"];
const CHUNK = 24;          // jumlah gambar per request (selaras MAX_BATCH backend)

let selected = [];         // File[] gambar terpilih
let results = [];          // baris hasil terkumpul (utk CSV)
let running = false;

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

// --- Pemilihan folder / berkas --------------------------------------------
el.pickFolder.addEventListener("click", (e) => { e.stopPropagation(); el.folder.click(); });
el.pickFiles.addEventListener("click", (e) => { e.stopPropagation(); el.files.click(); });
el.drop.addEventListener("click", () => el.folder.click());
el.drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.folder.click(); }
});
el.folder.addEventListener("change", () => setSelection(el.folder.files));
el.files.addEventListener("change", () => setSelection(el.files.files));

["dragenter", "dragover"].forEach((ev) =>
  el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.remove("drag"); }));
el.drop.addEventListener("drop", (e) => {
  if (e.dataTransfer.files && e.dataTransfer.files.length) setSelection(e.dataTransfer.files);
});

el.reset.addEventListener("click", resetAll);
el.run.addEventListener("click", processAll);
el.csv.addEventListener("click", downloadCsv);
el.checkpoint.addEventListener("change", () => { if (results.length && !running) processAll(); });

function setSelection(fileList) {
  // Folder bisa berisi non-gambar — saring ke image/* saja.
  selected = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
  selected.sort((a, b) => relName(a).localeCompare(relName(b), "id", { numeric: true }));
  hideError();
  if (selected.length === 0) {
    el.runBar.hidden = true;
    el.selCount.hidden = true;
    showError("Tidak ada berkas gambar (JPG/PNG) pada pilihan itu.");
    return;
  }
  el.selCount.hidden = false;
  el.selCount.textContent = `${selected.length} gambar dipilih`;
  el.runBar.hidden = false;
  el.csv.hidden = true;
}

// Nama relatif bila dari folder (webkitRelativePath), jika tidak pakai nama berkas.
const relName = (f) => f.webkitRelativePath || f.name;

function resetAll() {
  selected = []; results = []; running = false;
  el.folder.value = ""; el.files.value = "";
  el.runBar.hidden = true;
  el.selCount.hidden = true;
  el.progress.hidden = true;
  el.csv.hidden = true;
  el.rows.innerHTML = "";
  el.tableScroll.hidden = true;
  el.tableMeta.innerHTML = "";
  el.summaryPanel.hidden = true;
  el.empty.hidden = false;
  hideError();
}

// --- Proses batch (per-potong) --------------------------------------------
async function processAll() {
  if (!selected.length || running) return;
  running = true;
  hideError();
  results = [];
  el.rows.innerHTML = "";
  el.empty.hidden = true;
  el.tableScroll.hidden = false;
  el.summaryPanel.hidden = true;
  el.csv.hidden = true;
  el.run.disabled = true;
  el.reset.disabled = true;
  el.progress.hidden = false;
  setProgress(0, selected.length);

  const ckpt = el.checkpoint.value || "";
  let meta = null;

  try {
    for (let i = 0; i < selected.length; i += CHUNK) {
      const batch = selected.slice(i, i + CHUNK);
      const fd = new FormData();
      batch.forEach((f) => fd.append("files", f, relName(f)));
      fd.append("checkpoint", ckpt);

      const res = await fetch("/api/predict-batch", { method: "POST", body: fd });
      const data = await bacaJson(res);
      if (!res.ok) throw new Error(data.detail || "Prediksi batch gagal.");
      meta = data;

      data.hasil.forEach((row) => { results.push(row); appendRow(results.length, row); });
      setProgress(Math.min(i + batch.length, selected.length), selected.length);
    }
    renderSummary(meta);
    el.tableMeta.innerHTML =
      `<span class="chip"><b>ckpt</b> ${meta.checkpoint}</span>` +
      `<span class="chip"><b>perangkat</b> ${meta.device.toUpperCase()}</span>`;
    el.csv.hidden = false;
  } catch (e) {
    showError(e.message);
  } finally {
    running = false;
    el.run.disabled = false;
    el.reset.disabled = false;
  }
}

function setProgress(done, total) {
  el.progressFill.style.width = total ? (done / total * 100).toFixed(1) + "%" : "0%";
  el.progressLabel.textContent = `${done} / ${total}`;
}

const pct = (x) => (x * 100).toFixed(x >= 0.1 ? 1 : 2) + "%";

function appendRow(no, row) {
  const tr = document.createElement("tr");
  if (row.error) {
    tr.className = "row-error";
    tr.innerHTML =
      `<td class="col-no">${no}</td>` +
      `<td class="col-file" title="${esc(row.nama_file)}">${esc(row.nama_file)}</td>` +
      `<td class="col-type"><span class="tag-err">gagal</span></td>` +
      `<td class="col-conf" colspan="2">${esc(row.error)}</td>`;
  } else {
    const color = PALETTE[row.indeks] || "var(--muted)";
    const p = (row.confidence * 100).toFixed(2);
    tr.innerHTML =
      `<td class="col-no">${no}</td>` +
      `<td class="col-file" title="${esc(row.nama_file)}">${esc(row.nama_file)}</td>` +
      `<td class="col-type"><span class="dot" style="background:${color}"></span>` +
        `<span class="type-name" title="${esc(row.kelas)}">${esc(row.nama)}</span></td>` +
      `<td class="col-bar"><span class="bar-track"><span class="bar-fill" ` +
        `style="width:${p}%;background:${color}"></span></span></td>` +
      `<td class="col-val">${pct(row.confidence)}</td>`;
  }
  el.rows.appendChild(tr);
}

// --- Ringkasan distribusi --------------------------------------------------
function renderSummary(meta) {
  const ok = results.filter((r) => !r.error);
  const gagal = results.length - ok.length;
  const avg = ok.length ? ok.reduce((s, r) => s + r.confidence, 0) / ok.length : 0;

  el.summaryStats.innerHTML =
    stat(results.length, "total gambar") +
    stat(ok.length, "berhasil") +
    stat(gagal, "gagal", gagal ? "warn" : "") +
    stat(pct(avg), "rata-rata kepercayaan");

  // Hitung per-kelas (pakai indeks agar warna konsisten).
  const counts = new Map();
  ok.forEach((r) => counts.set(r.indeks, (counts.get(r.indeks) || 0) + 1));
  const entries = [...counts.entries()].sort((a, b) => b[1] - a[1]);

  el.summaryDist.innerHTML = "";
  for (const [idx, n] of entries) {
    const r = ok.find((x) => x.indeks === idx);
    const color = PALETTE[idx] || "var(--muted)";
    const w = (n / ok.length * 100).toFixed(1);
    const div = document.createElement("div");
    div.className = "dist-row";
    div.setAttribute("role", "listitem");
    div.innerHTML =
      `<span class="dot" style="background:${color}"></span>` +
      `<span class="dist-name">${esc(r.nama)}</span>` +
      `<span class="bar-track"><span class="bar-fill" style="width:${w}%;background:${color}"></span></span>` +
      `<span class="dist-val">${n} · ${w}%</span>`;
    el.summaryDist.appendChild(div);
  }
  el.summaryPanel.hidden = false;
}

function stat(value, label, mod) {
  return `<div class="stat${mod ? " " + mod : ""}"><div class="stat-val">${value}</div>` +
         `<div class="stat-label">${label}</div></div>`;
}

// --- Ekspor CSV ------------------------------------------------------------
function downloadCsv() {
  const head = ["no", "berkas", "kelas", "jenis_awan", "kepercayaan", "error"];
  const lines = [head.join(",")];
  results.forEach((r, i) => {
    lines.push([
      i + 1,
      csvCell(r.nama_file),
      csvCell(r.kelas || ""),
      csvCell(r.nama || ""),
      r.error ? "" : (r.confidence * 100).toFixed(2) + "%",
      csvCell(r.error || ""),
    ].join(","));
  });
  const blob = new Blob(["﻿" + lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "clouddensenet_hasil_folder.csv";
  a.click();
  URL.revokeObjectURL(a.href);
}

const csvCell = (s) => {
  s = String(s);
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

// --- Util ------------------------------------------------------------------
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function showError(msg) { el.errorMsg.textContent = msg; el.error.hidden = false; }
function hideError() { el.error.hidden = true; }

loadCheckpoints();
