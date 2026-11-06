/* GeoAI 운영 콘솔 화면. 외부 라이브러리 없이 동작한다(실행 중 인터넷 의존 금지). */
"use strict";

const STATUS_KO = {
  Requested: "요청됨", Accepted: "접수됨", Planned: "대기", Running: "진행 중",
  Succeeded: "완료", Failed: "실패", Rejected: "거절됨",
};
const PERF_KO = {
  request: "요청", accept: "접수", reject: "거절", progress: "진행",
  result: "결과", failure: "실패", event: "알림",
};
const METRIC_KO = {
  IoU: "IoU", mIoU: "mIoU", F1: "F1", precision: "정밀도", recall: "재현율",
  object_F1: "객체 F1", object_precision: "객체 정밀도", object_recall: "객체 재현율",
  pixel_accuracy: "화소 정확도",
  foreground_IoU: "변화 여부 IoU", foreground_F1: "변화 여부 F1",
  foreground_precision: "변화 여부 정밀도", foreground_recall: "변화 여부 재현율",
};
/* 지표 이름: 개념이 하나면 그 개념 이름을, 여럿이면 "개념 평균"을 붙인다. */
function metricLabel(key, domainIri) {
  const d = catalog && catalog.domains.find((x) => x.iri === domainIri);
  const concepts = (d && d.label_concepts) || [];
  if (key === "IoU" || key === "F1") {
    if (concepts.length === 1) return `${key} (${concepts[0].label})`;
    if (concepts.length > 1) return `${key} (유형 평균)`;
  }
  return METRIC_KO[key] || key;
}
function domainSlug(iri) {
  const m = String(iri).match(/\/domain\/([^#]+)/);
  return m ? m[1] : "default";
}
function rgb(c) { return `rgb(${c[0]},${c[1]},${c[2]})`; }
const AGENT_KO = {
  "geo-data-production-agent": "공간정보 생성·가공 에이전트",
  "preprocessing-agent": "전처리 에이전트",
  "model-training-agent": "모델·학습 에이전트",
  "analysis-evaluation-agent": "분석·평가 에이전트",
  "studio-agent": "Studio Agent",
  "ops-console": "운영 콘솔",
};
const GAI = "https://ontology.geoai.local/core#";

const $ = (sel, root = document) => root.querySelector(sel);
const view = $("#view");
let catalog = null;
let timer = null;

function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function fmt(v, digits = 3) {
  if (v === null || v === undefined) return "—";
  return typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(digits)) : esc(v);
}
function time(iso) {
  const d = new Date(iso);
  return d.toLocaleString("ko-KR", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
function pill(status) {
  return `<span class="pill ${esc(status)}">${esc(STATUS_KO[status] || status)}</span>`;
}
function objUrl(uri) {
  return `/api/objects?uri=${encodeURIComponent(uri)}`;
}
function shortName(uri) {
  return esc(String(uri).split("/").slice(-2).join("/"));
}

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    const err = new Error(body.detail || body.reason || `HTTP ${r.status}`);
    err.status = r.status;
    err.body = body;
    throw err;
  }
  return body;
}

function toast(msg, critical = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = critical ? "toast critical" : "toast";
  t.hidden = false;
  clearTimeout(toast.h);
  toast.h = setTimeout(() => { t.hidden = true; }, critical ? 8000 : 3500);
}

function poll(fn, ms) {
  clearTimeout(timer);
  const tick = async () => {
    try { await fn(); } catch (e) { console.error(e); }
    timer = setTimeout(tick, ms);
  };
  timer = setTimeout(tick, ms);
}

async function loadCatalog() {
  if (!catalog) {
    catalog = await api("/api/catalog");
    $("#ontology-version").textContent = `온톨로지 ${catalog.ontology_version}`;
  }
  return catalog;
}

/* 위험 알림 띠: 모든 화면 상단 */
function renderBanner(alerts) {
  const b = $("#alert-banner");
  if (!alerts || !alerts.length) { b.hidden = true; return; }
  const first = alerts[0];
  b.innerHTML = `<span class="dot" aria-hidden="true"></span>
    <span>주의가 필요한 작업 ${alerts.length}건 · 최근: ${esc(first.title)} — ${esc(first.label)}</span>
    <a href="#/tasks/${encodeURIComponent(first.task_id)}">바로 보기</a>`;
  b.hidden = false;
}

/* ---------------------------------------------------------------- 현황 */
async function dashboard() {
  const draw = async () => {
    const o = await api("/api/overview");
    renderBanner(o.alerts);
    const running = (o.counts.Running || 0) + (o.counts.Planned || 0);
    const bad = (o.counts.Failed || 0) + (o.counts.Rejected || 0);
    view.innerHTML = `
      <div class="page-head">
        <div><h1>작업 현황</h1><div class="muted">GeoAI Studio에 요청한 작업과 진행 상태</div></div>
        <a href="#/new"><button class="primary" type="button">새 작업 요청</button></a>
      </div>
      <div class="grid cols-4">
        <div class="card stat"><div class="k">진행 중</div><div class="v">${running}</div></div>
        <div class="card stat"><div class="k">완료</div><div class="v">${o.counts.Succeeded || 0}</div></div>
        <div class="card stat ${bad ? "critical" : ""}"><div class="k">실패·거절</div><div class="v">${bad}</div></div>
        <div class="card stat"><div class="k">전체</div><div class="v">${o.tasks.length}</div></div>
      </div>
      ${o.alerts.length ? `<div class="card" style="margin-top:16px"><h2>위험 알림</h2><ul class="alerts">${o.alerts.map((a) => `
        <li><a href="#/tasks/${encodeURIComponent(a.task_id)}"><strong>${esc(a.title)}</strong> · ${esc(a.domain_label)} ${esc(a.label)}
        <div class="muted">${esc(a.detail || "사유 없음")} · ${time(a.at)}</div></a></li>`).join("")}</ul></div>` : ""}
      <div class="card" style="margin-top:16px">
        <h2>최근 작업</h2>
        ${o.tasks.length ? `<div class="table-wrap"><table>
          <thead><tr><th>상태</th><th>분야</th><th>요청 작업</th><th>프로젝트</th><th class="num">IoU</th><th class="num">F1</th><th>갱신</th></tr></thead>
          <tbody>${o.tasks.map((t) => `
            <tr class="link ${t.critical ? "critical" : ""}" data-href="#/tasks/${encodeURIComponent(t.task_id)}">
              <td>${pill(t.status)}</td><td>${esc(t.domain_label)}</td><td>${esc(t.label)}</td>
              <td class="mono">${esc(t.project_id)}</td>
              <td class="num">${fmt(t.metrics.IoU)}</td><td class="num">${fmt(t.metrics.F1)}</td>
              <td class="muted">${time(t.updated_at)}</td>
            </tr>`).join("")}</tbody></table></div>`
          : `<div class="empty">아직 요청한 작업이 없습니다. <a href="#/new">새 작업</a>에서 시작하세요.</div>`}
      </div>`;
    view.querySelectorAll("tr.link").forEach((tr) => tr.addEventListener("click", () => { location.hash = tr.dataset.href; }));
  };
  await draw();
  poll(draw, 4000);
}

/* ---------------------------------------------------------------- 새 작업 */
const PLAN_ORDER = ["GeoDataProductionTask", "PreprocessingTask", "LabelingTask", "ModelTrainingTask", "EvaluationTask"];

async function newTask() {
  const [cat, assets, models] = await Promise.all([loadCatalog(), api("/api/assets"), api("/api/models")]);
  const domains = cat.domains;
  const building = domains.find((d) => d.iri.includes("building-extraction")) || domains[0];
  const targets = cat.task_types.filter((t) => ["EvaluationTask", "ModelTrainingTask", "LabelingTask", "PreprocessingTask", "GeoDataProductionTask"].includes(t.task_type.split("#")[1]));
  const evalTask = targets.find((t) => t.task_type.endsWith("#EvaluationTask"));
  const sample = assets.vectors.find((v) => v.uri.includes("sample/"));
  const preset = (assets.presets || [])[0];
  const opt = (list, chosen, none) => `${none ? `<option value="">${esc(none)}</option>` : ""}${list.map((x) => `<option value="${esc(x.uri)}" ${x.uri === chosen ? "selected" : ""}>${shortName(x.uri)}</option>`).join("")}`;
  const acqCard = (i) => {
    const p = (preset && preset.acquisitions[i]) || {};
    return `<fieldset class="acq" data-acq="${i}"><legend>시점 ${i + 1} <span class="muted">(${i === 0 ? "이전" : "이후"})</span></legend>
      <label>취득 일자 <input type="date" name="acquired_at" value="${esc(p.acquired_at || "")}" required></label>
      <label>영상 <select name="image">${opt(assets.images, p.image, "선택")}</select></label>
      <label>라이다 점군 <select name="lidar">${opt(assets.point_clouds || [], p.lidar, "없음")}</select></label>
      <label>건물 윤곽 <select name="buildings">${opt(assets.vectors, p.buildings, "없음")}</select></label>
      <label>토지피복 <select name="landcover">${opt(assets.vectors, p.landcover, "없음")}</select></label>
    </fieldset>`;
  };
  const buildingModels = models.models.filter((m) => String(m.domain || "").includes("building-extraction"));
  view.innerHTML = `
    <div class="page-head"><div><h1>새 작업 요청</h1>
      <div class="muted">영상과 참조 벡터를 고르면 Studio Agent가 온톨로지 규칙으로 단계를 계획합니다.</div></div></div>
    <form class="card" id="task-form" novalidate>
      <div class="row"><label class="k" for="f-domain">분야</label><div>
        <select id="f-domain">${domains.map((d) => `<option value="${esc(d.iri)}" ${d === building ? "selected" : ""}>${esc(d.label)}</option>`).join("")}</select>
        <div class="hint" id="f-domain-hint"></div></div></div>
      <div class="row"><label class="k" for="f-target">요청 작업</label><div>
        <select id="f-target">${targets.map((t) => `<option value="${esc(t.task_type)}" ${t === evalTask ? "selected" : ""}>${esc(t.label)}</option>`).join("")}</select>
        <div class="plan-preview" id="f-plan" style="margin-top:8px"></div>
        <div class="hint">성능 검증을 고르면 수집·가공부터 평가까지 한 번에 실행합니다.</div></div></div>
      <div class="row" id="f-multi" hidden><span class="k">시점별 입력</span><div>
        <div class="acq-grid">${acqCard(0)}${acqCard(1)}</div>
        <div class="hint">두 시점 영상은 겹치는 범위의 공통 격자로 맞춥니다. 라이다는 지물 높이 입력이 되고, 시점별 건물 윤곽·토지피복을 대조해 변화 라벨(신축·멸실·변경·토지 변화)을 만듭니다.</div></div></div>
      <div class="row" id="f-init" hidden><label class="k" for="f-pre">초기 가중치</label><div>
        <select id="f-pre"><option value="">없음 (처음부터 학습)</option>${buildingModels.map((m, i) => `<option value="${esc(m.uri)}" ${i === buildingModels.length - 1 ? "selected" : ""}>건물 추출 모델 · 검증 IoU ${fmt(m.val_IoU)} · ${esc(m.task_id.slice(0, 8))}</option>`).join("")}</select>
        <div class="hint">${buildingModels.length ? "건물 추출 모델의 영상 특징을 시점마다 옮겨 심고 학습을 시작합니다." : "학습된 건물 추출 모델이 없습니다. 건물 추출 작업을 먼저 돌리면 여기서 고를 수 있습니다."}</div></div></div>
      <div class="row" id="f-single-images"><span class="k">입력 영상</span><div>
        <div class="choices" id="f-images">${assets.images.length ? assets.images.map((a, i) => `
          <label class="choice"><input type="checkbox" name="image" value="${esc(a.uri)}" ${i === 0 ? "checked" : ""} ${a.error ? "disabled" : ""}>
          <span><span class="mono">${shortName(a.uri)}</span><br><small class="muted">${a.error ? esc(a.error) : `${esc(a.crs)} · ${fmt(a.ground_sample_distance, 2)} m · ${a.width}×${a.height}`}</small></span></label>`).join("")
          : `<div class="empty">등록된 영상이 없습니다. <a href="#/data">원본 자료</a>에서 올리세요.</div>`}</div></div></div>
      <div class="row" id="f-single-labels"><label class="k" for="f-labels">참조 벡터 (라벨 원천)</label><div>
        <select id="f-labels"><option value="">선택 안 함</option>${assets.vectors.map((v) => `<option value="${esc(v.uri)}" ${v === sample ? "selected" : ""}>${shortName(v.uri)}</option>`).join("")}</select>
        <div class="hint">건물 윤곽 GeoJSON. 라벨링 단계에서 타일마다 마스크로 바꿉니다.</div></div></div>
      <div class="row"><span class="k">완료 기준</span><div><div id="f-criteria"></div>
        <button type="button" class="ghost small" id="f-add-crit">기준 추가</button>
        <div class="hint">비워 두면 분야 온톨로지의 권장 기준을 씁니다. 미달이면 빨간색으로 알립니다.</div></div></div>
      <div class="row"><span class="k">학습 설정</span><div class="inline">
        <label>세대 수 <input type="number" id="f-epochs" value="20" min="1" max="500"></label>
        <label>타일 크기 <select id="f-tile"><option>128</option><option selected>256</option><option>512</option></select></label>
        <label>프로젝트 <input type="text" id="f-project" value="${esc(cat.default_project)}" maxlength="64"></label>
      </div></div>
      <div id="f-error"></div>
      <div class="actions"><button type="submit" class="primary" id="f-submit">작업 요청</button></div>
    </form>`;

  const critBox = $("#f-criteria");
  const addCrit = (c = { metric: "IoU", operator: ">=", threshold: 0.6 }) => {
    const row = document.createElement("div");
    row.className = "crit-row";
    row.innerHTML = `<input type="text" value="${esc(c.metric)}" aria-label="지표">
      <select aria-label="비교">${[">=", ">", "<=", "<", "=="].map((o) => `<option ${o === c.operator ? "selected" : ""}>${o}</option>`).join("")}</select>
      <input type="number" step="0.01" value="${esc(c.threshold)}" aria-label="기준값">
      <button type="button" class="ghost small" aria-label="기준 삭제">삭제</button>`;
    row.querySelector("button").addEventListener("click", () => row.remove());
    critBox.appendChild(row);
  };
  let lastSlug = cat.default_project;
  const isMulti = () => {
    const d = domains.find((x) => x.iri === $("#f-domain").value);
    return (d && d.acquisitions || 1) >= 2;
  };
  const syncDomain = () => {
    const d = domains.find((x) => x.iri === $("#f-domain").value);
    critBox.innerHTML = "";
    (d.default_criteria || []).forEach(addCrit);
    const multi = isMulti();
    $("#f-domain-hint").textContent = `라벨 개념: ${(d.label_concepts || []).map((c) => c.label).join(", ") || "없음"}${multi ? ` · 시점 ${d.acquisitions}개 이상 필요` : ""}`;
    $("#f-multi").hidden = !multi;
    $("#f-init").hidden = !multi;
    $("#f-single-images").hidden = multi;
    $("#f-single-labels").hidden = multi;
    const proj = $("#f-project");
    if (proj.value === lastSlug) { lastSlug = domainSlug(d.iri); proj.value = lastSlug; }
  };
  const syncPlan = () => {
    const local = $("#f-target").value.split("#")[1];
    const upto = PLAN_ORDER.indexOf(local);
    const chain = PLAN_ORDER.slice(0, upto + 1);
    $("#f-plan").innerHTML = chain.map((t) => `<span class="chip">${esc(cat.labels[GAI + t] || t)}</span>`).join('<span class="arrow">→</span>');
  };
  $("#f-domain").addEventListener("change", syncDomain);
  $("#f-target").addEventListener("change", syncPlan);
  $("#f-add-crit").addEventListener("click", () => addCrit({ metric: "F1", operator: ">=", threshold: 0.7 }));
  syncDomain();
  syncPlan();

  $("#task-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const multi = isMulti();
    const images = multi ? [] : [...view.querySelectorAll("input[name=image]:checked")].map((i) => i.value);
    const errBox = $("#f-error");
    errBox.innerHTML = "";
    const acquisitions = multi ? [...view.querySelectorAll("fieldset.acq")].map((f) => {
      const v = (n) => f.querySelector(`[name=${n}]`).value;
      return {
        acquired_at: v("acquired_at"), image_uris: v("image") ? [v("image")] : [], lidar_uris: v("lidar") ? [v("lidar")] : [],
        building_source: v("buildings") || null, landcover_source: v("landcover") || null,
      };
    }) : [];
    const problem = !multi ? (images.length ? "" : "입력 영상을 하나 이상 고르세요.")
      : acquisitions.some((a) => !a.acquired_at || !a.image_uris.length) ? "두 시점 모두 취득 일자와 영상을 고르세요."
      : acquisitions[0].acquired_at === acquisitions[1].acquired_at ? "두 시점의 취득 일자가 같습니다."
      : acquisitions[0].acquired_at > acquisitions[1].acquired_at ? "시점 1이 시점 2보다 앞선 일자여야 합니다." : "";
    if (problem) {
      errBox.innerHTML = `<div class="callout critical">${esc(problem)}</div>`;
      return;
    }
    const criteria = [...critBox.querySelectorAll(".crit-row")].map((r) => {
      const [m, o, t] = r.querySelectorAll("input, select");
      return { metric: m.value.trim(), operator: o.value, threshold: parseFloat(t.value) };
    }).filter((c) => c.metric && !Number.isNaN(c.threshold));
    const body = {
      project_id: $("#f-project").value.trim(),
      domain: $("#f-domain").value,
      task_type: $("#f-target").value,
      image_uris: images,
      acquisitions,
      label_source: multi ? null : ($("#f-labels").value || null),
      criteria,
      parameters: { epochs: parseInt($("#f-epochs").value, 10) || 20, tile_size: parseInt($("#f-tile").value, 10) },
    };
    if (multi && $("#f-pre").value) body.parameters.pretrained_uri = $("#f-pre").value;
    const btn = $("#f-submit");
    btn.disabled = true;
    btn.textContent = "요청 중…";
    try {
      const r = await api("/api/tasks", { method: "POST", body: JSON.stringify(body) });
      toast(`작업을 접수했습니다. 단계 ${r.plan.length}개`);
      location.hash = `#/tasks/${encodeURIComponent(r.task_id)}`;
    } catch (e) {
      const b = e.body || {};
      const v = (b.violations || []).map((x) => `<li>${esc(x.message)}${x.path ? ` <span class="mono">(${esc(x.path.split("#").pop())})</span>` : ""}</li>`).join("");
      errBox.innerHTML = `<div class="callout critical">요청이 거절되었습니다: ${esc(b.detail || e.message)}${v ? `<ul>${v}</ul>` : ""}</div>`;
      btn.disabled = false;
      btn.textContent = "작업 요청";
    }
  });
}

/* ---------------------------------------------------------------- 작업 상세 */
function curve(history) {
  if (!history || history.length < 2) return "";
  const W = 560, H = 200, P = 34;
  const n = history.length;
  const x = (i) => P + (i * (W - 2 * P)) / (n - 1);
  const maxLoss = Math.max(...history.map((h) => h.loss)) || 1;
  const yIoU = (v) => H - P + 6 - v * (H - 2 * P);
  const yLoss = (v) => H - P + 6 - (v / maxLoss) * (H - 2 * P);
  const path = (f, key) => history.map((h, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${f(h[key]).toFixed(1)}`).join(" ");
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="세대별 학습 손실과 검증 IoU">
    <line class="axis" x1="${P}" y1="${H - P + 6}" x2="${W - P}" y2="${H - P + 6}"/>
    <line class="axis" x1="${P}" y1="${P - 6}" x2="${P}" y2="${H - P + 6}"/>
    <text x="${P - 6}" y="${P}" text-anchor="end">1.0</text><text x="${P - 6}" y="${H - P + 6}" text-anchor="end">0</text>
    <text x="${W - P}" y="${H - 8}" text-anchor="end">세대 ${n}</text>
    <path d="${path(yLoss, "loss")}" fill="none" stroke="var(--wait)" stroke-width="2" stroke-dasharray="4 3"/>
    <path d="${path(yIoU, "val_IoU")}" fill="none" stroke="var(--primary)" stroke-width="2.5"/>
  </svg>
  <div class="legend"><span><i style="background:var(--primary)"></i>검증 IoU</span><span><i style="background:var(--wait)"></i>학습 손실(상대값)</span></div>`;
}

function conceptLabel(domainIri, local) {
  const d = catalog && catalog.domains.find((x) => x.iri === domainIri);
  const c = d && (d.label_concepts || []).find((x) => x.iri.endsWith(`#${local}`));
  return c ? c.label : local;
}

function criteriaTable(criteria, finished, domainIri) {
  if (!criteria.length) return `<div class="muted">완료 기준 없음</div>`;
  return `<table><thead><tr><th>지표</th><th>기준</th><th class="num">측정값</th><th>판정</th></tr></thead><tbody>
    ${criteria.map((c) => `<tr class="${finished && !c.met ? "unmet" : ""}"><td>${esc(metricLabel(c.metric, domainIri))}</td>
      <td class="mono">${esc(c.operator)} ${fmt(c.threshold, 2)}</td><td class="num">${fmt(c.value)}</td>
      <td>${!finished ? '<span class="muted">대기</span>' : c.met ? '<span class="verdict-ok">충족</span>' : '<span class="verdict-bad">미달</span>'}</td></tr>`).join("")}
  </tbody></table>`;
}

async function taskDetail(id) {
  const draw = async () => {
    const [d, o] = await Promise.all([api(`/api/tasks/${encodeURIComponent(id)}`), api("/api/overview")]);
    renderBanner(o.alerts);
    const t = d.task;
    const finished = ["Succeeded", "Failed", "Rejected"].includes(t.status);
    const rep = d.artifacts.report;
    const model = d.artifacts.model;
    const metrics = rep ? rep.metrics : t.metrics;
    const failedStep = d.steps.find((s) => s.critical);
    const dom = t.domain;
    const classes = (rep && rep.class_summary) || [];
    const multiClass = classes.length > 1;
    const dated = d.steps.flatMap((s) => s.outputs).filter((o) => o.acquired_at && o.asset_type === GAI + "OrthoImage");
    const dates = [...new Set(dated.map((o) => o.acquired_at))].sort();
    const lidarDates = [...new Set(d.inputs.filter((a) => a.asset_type === GAI + "LidarPointCloud").map((a) => a.acquired_at))];
    const inputDates = [...new Set(d.inputs.map((a) => a.acquired_at).filter(Boolean))].sort();
    const shownDates = dates.length ? dates : inputDates;
    const metricKeys = multiClass ? ["foreground_F1", "foreground_IoU", "IoU", "F1", "object_F1", "mIoU"] : ["IoU", "F1", "precision", "recall", "object_F1", "mIoU"];
    view.innerHTML = `
      <div class="page-head">
        <div><div class="muted"><a href="#/">현황</a> / 작업</div>
          <h1>${esc(t.domain_label)} · ${esc(t.label)} ${pill(t.status)}</h1>
          <div class="muted mono">${esc(t.task_id)} · 프로젝트 ${esc(t.project_id)} · 요청 ${time(t.created_at)}</div>
          ${shownDates.length > 1 ? `<div class="muted">비교 시점 ${shownDates.map(esc).join(" → ")}${lidarDates.length ? ` · 라이다 ${lidarDates.length}시점` : ""}</div>` : ""}</div>
        <div style="min-width:220px"><div class="muted">전체 진행 ${fmt(d.overall_percent, 0)}%</div>
          <div class="bar wide"><span class="${t.critical ? "fail" : t.status === "Succeeded" ? "done" : "live"}" style="width:${d.overall_percent}%"></span></div></div>
      </div>
      ${t.critical ? `<div class="callout critical" style="margin:0 0 16px">${t.criteria_failed ? "완료 기준 미달 — 운영자 판단이 필요합니다. " : t.status === "Rejected" ? "요청이 거절되었습니다. " : "작업이 실패했습니다. "}${esc(t.detail)}</div>` : ""}
      ${t.status === "Succeeded" ? `<div class="callout ok" style="margin:0 0 16px">모든 단계가 끝났고 완료 기준을 충족했습니다.</div>` : ""}
      <div class="card"><h2>처리 단계</h2>
        <div class="pipeline">${d.steps.map((s, i) => `
          <div class="step ${esc(s.status)}">
            <div class="node">${s.status === "Succeeded" ? "✓" : s.critical ? "!" : i + 1}</div>
            <div class="name">${esc(s.label.replace(/ 작업$/, ""))}</div>
            <div class="agent">${esc(AGENT_KO[s.agent] || s.agent)}</div>
            <div>${pill(s.status)}</div>
            <div class="bar"><span class="${s.status === "Succeeded" ? "done" : s.critical ? "fail" : ""}" style="width:${s.percent || 0}%"></span></div>
            <div class="note">${esc(s.critical ? s.detail : s.note)}</div>
          </div>`).join("")}</div>
        ${failedStep ? `<div class="callout critical">${esc(failedStep.label)} 단계 실패: ${esc(failedStep.detail)}</div>` : ""}
      </div>
      <div class="grid cols-2" style="margin-top:16px">
        <div class="card"><h2>완료 기준 판정</h2>${criteriaTable(t.criteria, finished && Object.keys(t.metrics).length > 0, dom)}</div>
        <div class="card"><h2>성능 지표 ${rep ? `<small class="muted">(${rep.split === "test" ? "시험" : "검증"} 집합 ${rep.tiles}장)</small>` : ""}</h2>
          ${metrics && Object.keys(metrics).length ? `<div class="metrics">${metricKeys.filter((k) => k in metrics).map((k) => `
            <div class="metric"><div class="k">${esc(metricLabel(k, dom))}</div><div class="v">${fmt(metrics[k])}</div></div>`).join("")}</div>
            ${rep ? `<div class="muted" style="margin-top:8px">객체: 예측 ${rep.objects.predicted} · 정답 ${rep.objects.reference} · 대응 ${rep.objects.matched} (IoU ≥ ${rep.objects.iou_threshold})</div>` : ""}`
          : `<div class="muted">평가 단계가 끝나면 표시됩니다.</div>`}</div>
      </div>
      ${rep || model ? `<div class="grid cols-2" style="margin-top:16px">
        ${rep ? (multiClass ? `<div class="card preview"><h2>변화 지도 미리보기</h2>
          ${rep.preview_layout === "pair" ? `<div class="muted" style="margin-bottom:6px">타일마다 왼쪽 ${esc((rep.acquisitions || [])[0] || "이전")} · 오른쪽 ${esc((rep.acquisitions || []).slice(-1)[0] || "이후")} + 변화</div>` : ""}
          <img src="${objUrl(rep.preview_uri)}" alt="시험 타일 위에 예측 변화(색 채움)와 정답 변화(같은 색 윤곽)를 겹친 영상">
          <div class="legend">${classes.map((c) => `<span><i style="background:${rgb(c.color)}"></i>${esc(c.label)}</span>`).join("")}<span class="muted">채움 = 예측, 윤곽 = 정답</span></div></div>`
          : `<div class="card preview"><h2>시험 타일 미리보기</h2><img src="${objUrl(rep.preview_uri)}" alt="시험 타일 위에 예측 건물(호박색)과 정답 윤곽(청록색)을 겹친 영상">
          <div class="legend"><span><i style="background:#e0a030"></i>예측 건물</span><span><i style="background:var(--teal)"></i>정답 윤곽</span></div></div>`) : ""}
        ${model ? `<div class="card chart"><h2>학습 이력</h2>${curve(model.history)}
          <div class="muted" style="margin-top:6px">최고 검증 IoU ${fmt(model.val_IoU)} (세대 ${model.best_epoch}) · 매개변수 ${(model.parameters / 1e6).toFixed(2)}M · 입력 ${(model.bands || []).length || 3}밴드 · ${esc(model.device)} · ${fmt(model.train_seconds, 0)}초</div>
          ${model.pretrained ? `<div class="muted">초기 가중치: ${String(model.pretrained.domain || "").includes("building-extraction") ? "건물 추출 모델" : "사전 학습 모델"} · 옮긴 층 ${model.pretrained.copied + model.pretrained.adapted}/${model.pretrained.total} (첫 층 시점별 분배 ${model.pretrained.adapted})</div>` : ""}</div>` : ""}
      </div>` : ""}
      ${multiClass ? `<div class="card" style="margin-top:16px"><h2>변화 유형별 결과</h2><div class="table-wrap"><table>
        <thead><tr><th>유형</th><th class="num">IoU</th><th class="num">F1</th><th class="num">예측 면적(m²)</th><th class="num">정답 면적(m²)</th><th class="num">객체 대응/예측/정답</th></tr></thead>
        <tbody>${classes.map((c) => `<tr><td><i class="swatch" style="background:${rgb(c.color)}"></i>${esc(c.label)}</td>
          <td class="num">${fmt(c.IoU)}</td><td class="num">${fmt(c.F1)}</td><td class="num">${fmt(c.predicted_m2, 0)}</td><td class="num">${fmt(c.reference_m2, 0)}</td>
          <td class="num">${c.objects_matched ?? "—"}/${c.objects_predicted ?? "—"}/${c.objects_reference ?? "—"}</td></tr>`).join("")}</tbody></table></div>
        <div class="muted" style="margin-top:6px">예측 변화 다각형은 단계별 결과의 predictions.geojson 으로 내려받습니다.</div></div>` : ""}
      <div class="card" style="margin-top:16px"><h2>단계별 결과</h2>
        <div class="table-wrap"><table><thead><tr><th>단계</th><th>상태</th><th>주요 지표</th><th>산출물</th></tr></thead><tbody>
        ${d.steps.map((s) => `<tr class="${s.critical ? "critical" : ""}"><td>${esc(s.label)}</td><td>${pill(s.status)}</td>
          <td class="mono">${Object.entries(s.metrics).slice(0, 5).map(([k, v]) => `${esc(k)} ${fmt(v)}`).join(" · ") || "—"}</td>
          <td class="files">${s.outputs.map((a) => `<a href="${objUrl(a.storage_uri)}" target="_blank" rel="noopener">${shortName(a.storage_uri)}</a>`).join("") || "—"}</td></tr>`).join("")}
        </tbody></table></div>
        ${d.artifacts.labels ? `<div class="muted" style="margin-top:8px">라벨 품질: 지물 ${d.artifacts.labels.quality.features}개 · 기하 수정 ${d.artifacts.labels.quality.fixed_invalid}개 · 라벨 화소 비율 ${fmt(d.artifacts.labels.quality.positive_pixel_ratio)}${d.artifacts.labels.quality.change_polygons ? ` · 도출 변화 ${Object.entries(d.artifacts.labels.quality.change_polygons).map(([k, v]) => `${esc(conceptLabel(dom, k))} ${v}`).join(", ")}` : ""}</div>` : ""}
        ${d.artifacts.ard ? `<div class="muted">타일 ${d.artifacts.ard.tile_size}px · 학습 ${d.artifacts.ard.split_counts.train} / 검증 ${d.artifacts.ard.split_counts.val} / 시험 ${d.artifacts.ard.split_counts.test}</div>` : ""}
      </div>
      <div class="card" style="margin-top:16px"><details><summary>메시지 기록 ${d.timeline.length}건</summary>
        <ul class="timeline">${d.timeline.map((m) => `<li><span class="muted">${time(m.at).split(" ").pop()}</span>
          <span class="perf-${esc(m.performative)}">${esc(PERF_KO[m.performative] || m.performative)}</span>
          <span>${esc(AGENT_KO[m.sender] || m.sender)} → ${esc(AGENT_KO[m.recipient] || m.recipient || "구독자")} ${m.note ? `· ${esc(m.note)}` : ""}</span></li>`).join("")}</ul>
      </details></div>`;
    if (finished) clearTimeout(timer);
    return finished;
  };
  const finished = await draw();
  if (!finished) poll(async () => { if (await draw()) clearTimeout(timer); }, 2000);
}

/* ---------------------------------------------------------------- 원본 자료 */
async function dataView() {
  const assets = await api("/api/assets");
  view.innerHTML = `
    <div class="page-head"><div><h1>원본 자료</h1><div class="muted">데이터·모델 저장소 raw 버킷</div></div></div>
    <div class="card"><h2>자료 올리기</h2>
      <form id="up-form" class="inline"><input type="file" id="up-file" accept=".tif,.tiff,.geojson,.json,.las" required>
      <button class="primary" type="submit">올리기</button></form>
      <div class="hint muted" style="margin-top:6px">GeoTIFF(좌표 참조 체계 포함, 3밴드 이상), 라이다 LAS, 건물 윤곽·토지피복 GeoJSON. 다시기 분야는 새 작업 화면에서 시점별 취득 일자를 지정합니다.</div></div>
    <div class="card"><h2>영상 ${assets.images.length}건</h2>
      ${assets.images.length ? `<div class="table-wrap"><table><thead><tr><th>주소</th><th>좌표 참조 체계</th><th class="num">해상도(m)</th><th class="num">크기</th></tr></thead><tbody>
      ${assets.images.map((a) => `<tr><td class="mono">${esc(a.uri)}</td><td>${a.error ? `<span class="verdict-bad">${esc(a.error)}</span>` : esc(a.crs)}</td>
        <td class="num">${fmt(a.ground_sample_distance, 2)}</td><td class="num">${a.width ?? "—"}×${a.height ?? "—"}</td></tr>`).join("")}</tbody></table></div>`
      : `<div class="empty">영상이 없습니다.</div>`}</div>
    <div class="card"><h2>라이다 점군 ${(assets.point_clouds || []).length}건</h2>
      ${(assets.point_clouds || []).length ? `<ul>${assets.point_clouds.map((v) => `<li class="mono">${esc(v.uri)} <span class="muted">(${(v.size / 1048576).toFixed(1)} MB)</span></li>`).join("")}</ul>` : `<div class="empty">라이다 점군이 없습니다.</div>`}</div>
    <div class="card"><h2>참조 벡터 ${assets.vectors.length}건</h2>
      ${assets.vectors.length ? `<ul>${assets.vectors.map((v) => `<li class="mono">${esc(v.uri)} <span class="muted">(${(v.size / 1024).toFixed(0)} KB)</span></li>`).join("")}</ul>` : `<div class="empty">참조 벡터가 없습니다.</div>`}</div>`;
  $("#up-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const f = $("#up-file").files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    const r = await fetch("/api/assets", { method: "POST", body: fd });
    const body = await r.json().catch(() => ({}));
    if (r.ok) { toast(`올렸습니다: ${body.uri}`); dataView(); } else { toast(body.detail || "올리기 실패", true); }
  });
}

/* ---------------------------------------------------------------- 경로 */
async function route() {
  clearTimeout(timer);
  const h = location.hash || "#/";
  const name = h.startsWith("#/tasks/") ? "" : h === "#/new" ? "new" : h === "#/data" ? "data" : "dashboard";
  document.querySelectorAll(".nav nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === name));
  try {
    await loadCatalog();
    if (h === "#/new" || h === "#/data") api("/api/overview").then((o) => renderBanner(o.alerts)).catch(() => {});
    if (h.startsWith("#/tasks/")) await taskDetail(decodeURIComponent(h.slice(8)));
    else if (h === "#/new") await newTask();
    else if (h === "#/data") await dataView();
    else await dashboard();
  } catch (e) {
    view.innerHTML = `<div class="callout critical">화면을 불러오지 못했습니다: ${esc(e.message)}</div>`;
  }
}

(function theme() {
  const saved = (() => { try { return localStorage.getItem("geoai-theme"); } catch { return null; } })();
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme-toggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme === "dark"
      || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("geoai-theme", document.documentElement.dataset.theme); } catch { /* 저장 불가 환경 */ }
  });
})();

window.addEventListener("hashchange", route);
route();
