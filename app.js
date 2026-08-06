// app.js - dung chung cho ca 3 trang. Khong dung framework/build step -
// vanilla JS + fetch() goi thang API (cung domain Vercel nen khong can lo
// CORS trong thuc te, du sao API cung da bat allow_origins=["*"]).

const API_BASE = ""; // cung domain (Vercel serve ca static file + /api/* trong 1 project)

function fmtPct(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
  const s = v >= 0 ? "+" : "";
  return `${s}${v.toFixed(1)}%`;
}
function fmtPrice(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
  return v.toFixed(2);
}
function fmtPctPlain(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
  return `${(v * 100).toFixed(0)}%`;
}

async function apiGet(path) {
  const res = await fetch(API_BASE + path);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Lỗi API (${res.status})`);
  }
  return data;
}

async function apiPost(path) {
  const res = await fetch(API_BASE + path, { method: "POST" });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Lỗi API (${res.status})`);
  }
  return data;
}

function downloadCsv(filename, headers, rows) {
  const escape = (v) => {
    const s = String(v ?? "");
    return s.includes(",") || s.includes('"') || s.includes("\n")
      ? '"' + s.replace(/"/g, '""') + '"'
      : s;
  };
  const lines = [headers.map(escape).join(",")];
  for (const r of rows) lines.push(r.map(escape).join(","));
  const blob = new Blob(["﻿" + lines.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
