/**
 * GroundCheck — Frontend Application Controller
 * Connects Document Corpus Management, Evidence-Grounded Q&A with Citation Highlighting,
 * Claim-by-Claim Verification Auditing, and Reproducible Evaluation Benchmarking.
 */

const state = {
  documents: [],
  queries: [],
  evalRuns: [],
  activeEvalRun: null,
  benchmarkInfo: null,
  theme: "dark",
};

// ---------------------------------------------------------------------------
// Utility Helpers
// ---------------------------------------------------------------------------

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function showToast(message, isError = false) {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const el = document.createElement("div");
  el.className = `toast ${isError ? "error" : ""}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => {
    el.remove();
  }, 4200);
}

async function apiFetch(url, options = {}) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    let errText = `HTTP ${resp.status}`;
    try {
      const data = await resp.json();
      if (data && data.detail) errText = data.detail;
    } catch (_) {
      // ignore
    }
    throw new Error(errText);
  }
  return resp.json();
}

function formatAnswerWithClickableCitations(markdownText) {
  const lines = (markdownText || "").split("\n").filter((ln) => ln.trim().length > 0);
  const items = lines.map((line) => {
    const cleaned = line.replace(/^[-*•]\s+/, "").trim();
    const safe = escapeHtml(cleaned);
    // Replace [doc-id::ch-001] with clickable citation badge
    const withChips = safe.replace(
      /\[([a-zA-Z0-9_\-]+::ch-\d{3})\]/g,
      (_, chunkId) =>
        `<button type="button" class="citation-chip" data-cite-id="${chunkId}" title="Click to inspect supporting passage ${chunkId}">[${chunkId}]</button>`
    );
    return `<li>${withChips}</li>`;
  });
  return `<ul>${items.join("")}</ul>`;
}

// ---------------------------------------------------------------------------
// Navigation & Theme
// ---------------------------------------------------------------------------

function switchTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    const isMatch = btn.getAttribute("data-tab") === tabName;
    btn.classList.toggle("active", isMatch);
    btn.setAttribute("aria-selected", isMatch ? "true" : "false");
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${tabName}`);
  });
}

function initTabsAndControls() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      switchTab(btn.getAttribute("data-tab"));
    });
  });

  const themeBtn = document.getElementById("theme-toggle-btn");
  themeBtn.addEventListener("click", () => {
    state.theme = state.theme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", state.theme);
  });

  // Range sliders live value labels
  const topKInput = document.getElementById("input-top-k");
  const alphaInput = document.getElementById("input-alpha");
  const minScoreInput = document.getElementById("input-min-score");

  topKInput.addEventListener("input", () => {
    document.getElementById("label-top-k").textContent = topKInput.value;
  });
  alphaInput.addEventListener("input", () => {
    document.getElementById("label-alpha").textContent = parseFloat(alphaInput.value).toFixed(2);
  });
  minScoreInput.addEventListener("input", () => {
    document.getElementById("label-min-score").textContent = parseFloat(minScoreInput.value).toFixed(2);
  });

  // Preset query buttons
  document.querySelectorAll(".preset-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-query");
      const input = document.getElementById("question-input");
      input.value = q;
      runQuery(q);
    });
  });

  // Query form submit
  document.getElementById("query-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = document.getElementById("question-input").value.trim();
    if (q) runQuery(q);
  });

  // File upload dropzone & submit
  const fileInput = document.getElementById("file-upload-input");
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    document.getElementById("selected-file-label").textContent = file
      ? `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`
      : "PDF, Markdown, Plain Text, or HTML";
  });

  document.getElementById("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = fileInput.files[0];
    if (!file) {
      showToast("Please select a document file to upload first.", true);
      return;
    }
    const btn = document.getElementById("upload-submit-btn");
    btn.disabled = true;
    btn.textContent = "Indexing...";
    try {
      const formData = new FormData();
      formData.append("file", file);
      const doc = await apiFetch("/api/documents/upload", {
        method: "POST",
        body: formData,
      });
      showToast(`Indexed '${doc.filename}' (${doc.chunk_count} chunks).`);
      fileInput.value = "";
      document.getElementById("selected-file-label").textContent = "PDF, Markdown, Plain Text, or HTML";
      await loadDocuments();
      inspectDocument(doc.doc_id);
    } catch (err) {
      showToast(`Upload failed: ${err.message}`, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "Upload & Index Document";
    }
  });

  document.getElementById("reseed-corpus-btn").addEventListener("click", async () => {
    try {
      await apiFetch("/api/documents/seed", { method: "POST" });
      showToast("Restored built-in technical reference corpus.");
      await loadDocuments();
    } catch (err) {
      showToast(`Failed to restore corpus: ${err.message}`, true);
    }
  });

  // Evaluation buttons
  document.getElementById("run-single-eval-btn").addEventListener("click", () => {
    triggerSingleEvaluation();
  });
  document.getElementById("run-ablation-suite-btn").addEventListener("click", () => {
    triggerAblationSuite();
  });
}

// ---------------------------------------------------------------------------
// Research & Evidence-Grounded Q&A Workflow
// ---------------------------------------------------------------------------

async function runQuery(questionText) {
  const submitBtn = document.getElementById("ask-submit-btn");
  submitBtn.disabled = true;
  submitBtn.textContent = "Retrieving & Verifying...";

  const payload = {
    question: questionText,
    retrieval_mode: document.getElementById("select-retrieval-mode").value,
    top_k: parseInt(document.getElementById("input-top-k").value, 10),
    alpha: parseFloat(document.getElementById("input-alpha").value),
    min_evidence_score: parseFloat(document.getElementById("input-min-score").value),
    generation_mode: document.getElementById("select-gen-mode").value,
  };

  try {
    const resp = await apiFetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderQueryResponse(resp);
    await loadRecentQueries();
  } catch (err) {
    showToast(`Query error: ${err.message}`, true);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Run Grounded Investigation";
  }
}

function renderQueryResponse(resp) {
  const container = document.getElementById("query-result-container");
  const v = resp.verification;
  const t = resp.telemetry;

  const verdictLabels = {
    grounded: "VERIFIED GROUNDED ANSWER",
    partially_grounded: "PARTIALLY GROUNDED — CHECK EXCERPTS",
    conflicted_evidence: "CONFLICTING EVIDENCE DETECTED ACROSS SOURCES",
    insufficient_evidence: "ABSTAINED — INSUFFICIENT EVIDENCE IN CORPUS",
  };

  let conflictHtml = "";
  if (v.has_conflict && v.conflicts && v.conflicts.length > 0) {
    const pairs = v.conflicts
      .map(
        (c) => `
        <div style="margin-bottom:0.5rem;">${escapeHtml(c.conflict_description)}</div>
        <div class="conflict-pair-grid">
          <div class="conflict-side">
            <strong>${escapeHtml(c.source_a)}</strong>
            <button type="button" class="citation-chip" data-cite-id="${escapeHtml(c.chunk_id_a)}">[${escapeHtml(c.chunk_id_a)}]</button>
            <p style="margin-top:0.35rem;">"${escapeHtml(c.excerpt_a)}"</p>
          </div>
          <div class="conflict-side">
            <strong>${escapeHtml(c.source_b)}</strong>
            <button type="button" class="citation-chip" data-cite-id="${escapeHtml(c.chunk_id_b)}">[${escapeHtml(c.chunk_id_b)}]</button>
            <p style="margin-top:0.35rem;">"${escapeHtml(c.excerpt_b)}"</p>
          </div>
        </div>`
      )
      .join("");
    conflictHtml = `
      <div class="conflict-box">
        <h4>&#9888; Cross-Document Specification Discrepancy</h4>
        ${pairs}
      </div>`;
  }

  const claimsHtml = (v.claims || [])
    .map((cl) => {
      const citeBtns = (cl.cited_chunk_ids || [])
        .map(
          (cid) =>
            `<button type="button" class="citation-chip" data-cite-id="${escapeHtml(cid)}">[${escapeHtml(cid)}]</button>`
        )
        .join(" ");
      return `
        <div class="claim-item">
          <div class="claim-header">
            <span class="claim-badge badge-${escapeHtml(cl.status)}">${escapeHtml(cl.status.replace("_", " "))}</span>
            <span style="font-family:var(--font-mono);font-size:0.75rem;">Support: ${(cl.support_score * 100).toFixed(0)}% ${citeBtns}</span>
          </div>
          <div style="font-size:0.84rem;">${escapeHtml(cl.claim_text)}</div>
          <div class="claim-explanation">${escapeHtml(cl.explanation)}</div>
        </div>`;
    })
    .join("");

  container.innerHTML = `
    <div class="panel-card">
      <div class="verdict-banner verdict-${escapeHtml(v.verdict)}">
        <div>
          <div class="verdict-title">${verdictLabels[v.verdict] || escapeHtml(v.verdict)}</div>
          <div class="verdict-note">${escapeHtml(v.epistemic_note)}</div>
        </div>
        <div class="verdict-metrics-strip">
          <div class="mini-stat">Groundedness: <strong>${(v.overall_groundedness_score * 100).toFixed(1)}%</strong></div>
          <div class="mini-stat">Citation Precision: <strong>${(v.citation_precision * 100).toFixed(1)}%</strong></div>
          <div class="mini-stat">Claims Verified: <strong>${v.supported_claim_count}/${v.total_substantive_claims}</strong></div>
        </div>
      </div>

      ${conflictHtml}

      <div class="card-header">
        <h2>Evidence-Grounded Answer</h2>
        <span class="card-subtitle">Click any inline <code>[doc_id::ch-XXX]</code> badge to highlight the exact source passage on the right</span>
      </div>
      <div class="answer-prose">
        ${formatAnswerWithClickableCitations(resp.answer_markdown)}
      </div>

      <div class="card-header">
        <h2>Claim-by-Claim Verification Audit</h2>
        <span class="card-subtitle">Sentence-level lexical, bigram, and numeric-fidelity entailment breakdown</span>
      </div>
      <div class="claim-list">
        ${claimsHtml}
      </div>

      <div class="telemetry-bar">
        <span>Total Latency: <strong>${t.total_latency_ms.toFixed(2)} ms</strong> (Ret: ${t.retrieval_latency_ms.toFixed(1)}ms, Syn: ${t.generation_latency_ms.toFixed(1)}ms, Ver: ${t.verification_latency_ms.toFixed(1)}ms)</span>
        <span>Tokens Est: <strong>${t.total_tokens_est}</strong></span>
        <span>Actual Cost: <strong>$${t.actual_api_cost_usd.toFixed(4)}</strong> (Ref LLM Est: $${t.reference_llm_cost_usd.toFixed(5)})</span>
        <span>Mode: <strong>${escapeHtml(t.execution_mode)}</strong></span>
      </div>
    </div>
  `;

  renderRetrievedPassages(resp.retrieved_passages || []);

  // Attach click handlers for citation chips
  container.querySelectorAll(".citation-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const targetId = chip.getAttribute("data-cite-id");
      highlightPassageCard(targetId);
    });
  });
}

function renderRetrievedPassages(passages) {
  const listEl = document.getElementById("retrieved-passages-list");
  document.getElementById("evidence-count-label").textContent = `${passages.length} passages retrieved`;

  if (!passages.length) {
    listEl.innerHTML = `<p class="empty-muted">No passages retrieved.</p>`;
    return;
  }

  listEl.innerHTML = passages
    .map((p) => {
      const matchedPills = (p.matched_terms || [])
        .map((t) => `<span class="term-pill">${escapeHtml(t)}</span>`)
        .join("");
      return `
        <div class="passage-card" id="passage-card-${escapeHtml(p.chunk_id)}" data-chunk-id="${escapeHtml(p.chunk_id)}">
          <div class="passage-top-bar">
            <span class="passage-id-tag">#${p.rank} [${escapeHtml(p.chunk_id)}]</span>
            <span class="claim-badge badge-${p.combined_score >= 0.30 ? "supported" : p.combined_score >= 0.18 ? "partially_supported" : "unsupported"}">
              ${escapeHtml(p.relevance_tier)} (${p.combined_score.toFixed(3)})
            </span>
          </div>
          <div class="passage-meta">
            <strong>${escapeHtml(p.filename)}</strong> &bull; Sec: ${escapeHtml(p.section_title)} ${p.page_number ? `&bull; Page ${p.page_number}` : ""}
          </div>
          <div class="score-bar-row">
            <span>Combined: <strong>${p.combined_score.toFixed(3)}</strong></span>
            <span>BM25: <strong>${p.bm25_score.toFixed(3)}</strong></span>
            <span>Semantic: <strong>${p.semantic_score.toFixed(3)}</strong></span>
          </div>
          <div class="passage-body">${escapeHtml(p.text)}</div>
          <div class="matched-terms-row">
            ${matchedPills}
          </div>
        </div>
      `;
    })
    .join("");
}

function highlightPassageCard(chunkId) {
  document.querySelectorAll(".passage-card").forEach((el) => {
    el.classList.remove("highlighted");
  });
  const target = document.getElementById(`passage-card-${chunkId}`);
  if (target) {
    target.classList.add("highlighted");
    target.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } else {
    showToast(`Passage ${chunkId} is outside current view.`, true);
  }
}

async function loadRecentQueries() {
  try {
    const queries = await apiFetch("/api/queries?limit=12");
    state.queries = queries;
    const container = document.getElementById("recent-queries-list");
    if (!queries.length) {
      container.innerHTML = `<p class="empty-muted">No queries executed yet.</p>`;
      return;
    }
    container.innerHTML = queries
      .map(
        (q) => `
        <div class="history-item" data-query-id="${escapeHtml(q.query_id)}">
          <div style="font-weight:600;font-size:0.8rem;margin-bottom:0.2rem;">${escapeHtml(q.question.slice(0, 75))}${q.question.length > 75 ? "..." : ""}</div>
          <div style="display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:0.7rem;color:var(--text-muted);">
            <span>${escapeHtml(q.verdict)}</span>
            <span>${(q.groundedness_score * 100).toFixed(0)}% | ${q.latency_ms.toFixed(0)}ms</span>
          </div>
        </div>`
      )
      .join("");

    container.querySelectorAll(".history-item").forEach((item) => {
      item.addEventListener("click", async () => {
        const qid = item.getAttribute("data-query-id");
        try {
          const fullResp = await apiFetch(`/api/queries/${qid}`);
          document.getElementById("question-input").value = fullResp.question;
          renderQueryResponse(fullResp);
        } catch (err) {
          showToast(`Could not load query: ${err.message}`, true);
        }
      });
    });
  } catch (_) {
    // ignore
  }
}

// ---------------------------------------------------------------------------
// Document Corpus Management & Chunk Inspection
// ---------------------------------------------------------------------------

async function loadDocuments() {
  try {
    const docs = await apiFetch("/api/documents");
    state.documents = docs;
    document.getElementById("nav-doc-count").textContent = docs.length;
    document.getElementById("active-corpus-hint").textContent = `Searching across ${docs.length} indexed technical documents`;

    const listEl = document.getElementById("corpus-doc-list");
    if (!docs.length) {
      listEl.innerHTML = `<p class="empty-muted">No documents in corpus. Upload a file or click "Restore Sample Corpus".</p>`;
      return;
    }

    listEl.innerHTML = docs
      .map(
        (d) => `
        <div class="doc-item" data-doc-id="${escapeHtml(d.doc_id)}">
          <div class="doc-item-header">
            <span class="doc-item-title">${escapeHtml(d.title)}</span>
            <button type="button" class="btn-danger doc-delete-btn" data-del-id="${escapeHtml(d.doc_id)}" title="Remove document">Delete</button>
          </div>
          <div class="doc-item-meta">
            ${escapeHtml(d.filename)} &bull; ${d.chunk_count} chunks &bull; ~${d.total_tokens_est} tokens ${d.is_sample ? "&bull; [Sample]" : ""}
          </div>
        </div>`
      )
      .join("");

    listEl.querySelectorAll(".doc-item").forEach((el) => {
      el.addEventListener("click", (e) => {
        if (e.target.classList.contains("doc-delete-btn")) return;
        inspectDocument(el.getAttribute("data-doc-id"));
      });
    });

    listEl.querySelectorAll(".doc-delete-btn").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const docId = btn.getAttribute("data-del-id");
        try {
          await apiFetch(`/api/documents/${docId}`, { method: "DELETE" });
          showToast(`Deleted document '${docId}'.`);
          await loadDocuments();
        } catch (err) {
          showToast(`Delete failed: ${err.message}`, true);
        }
      });
    });
  } catch (err) {
    showToast(`Failed to load documents: ${err.message}`, true);
  }
}

async function inspectDocument(docId) {
  try {
    const doc = await apiFetch(`/api/documents/${docId}`);
    const viewer = document.getElementById("document-detail-view");
    const chunksHtml = (doc.chunks || [])
      .map(
        (c) => `
        <div class="passage-card">
          <div class="passage-top-bar">
            <span class="passage-id-tag">[${escapeHtml(c.chunk_id)}]</span>
            <span style="font-family:var(--font-mono);font-size:0.73rem;color:var(--text-muted);">
              Section: ${escapeHtml(c.section_title)} &bull; ${c.char_count} chars (~${c.token_estimate} tokens)
            </span>
          </div>
          <div class="passage-body">${escapeHtml(c.text)}</div>
        </div>`
      )
      .join("");

    viewer.innerHTML = `
      <div class="card-header">
        <h2>${escapeHtml(doc.title)}</h2>
        <span class="card-subtitle">
          Doc ID: <code>${escapeHtml(doc.doc_id)}</code> &bull; File: <code>${escapeHtml(doc.filename)}</code> &bull;
          Type: ${escapeHtml(doc.file_type.toUpperCase())} &bull; ${doc.chunk_count} segmented passages &bull; SHA-256: <code>${escapeHtml(doc.content_hash.slice(0, 16))}...</code>
        </span>
      </div>
      <div class="passages-container" style="max-height:72vh;">
        ${chunksHtml}
      </div>
    `;
  } catch (err) {
    showToast(`Error inspecting document: ${err.message}`, true);
  }
}

// ---------------------------------------------------------------------------
// AI Evaluation & Reproducible Benchmarking Dashboard
// ---------------------------------------------------------------------------

async function loadEvaluationDashboard() {
  try {
    const [info, runs] = await Promise.all([
      apiFetch("/api/evaluations/benchmark-info"),
      apiFetch("/api/evaluations/runs?limit=20"),
    ]);
    state.benchmarkInfo = info;
    state.evalRuns = runs;
    document.getElementById("nav-eval-count").textContent = runs.length;

    renderMetricsGuide(info.metrics_guide || []);
    renderEvalRunsTable(runs);
    if (runs.length > 0) {
      selectEvaluationRun(runs[0]);
    }
  } catch (err) {
    showToast(`Failed to load evaluation data: ${err.message}`, true);
  }
}

function renderMetricsGuide(metricsList) {
  const grid = document.getElementById("metrics-guide-grid");
  grid.innerHTML = metricsList
    .map(
      (m) => `
      <div class="metric-doc-card">
        <div class="metric-doc-title">
          <span>${escapeHtml(m.name)}</span>
          <span class="${m.measurement_type === "Measured" ? "tag-meas" : "tag-est"}">${escapeHtml(m.measurement_type)}</span>
        </div>
        <code class="metric-doc-formula">${escapeHtml(m.formula)}</code>
        <p class="metric-doc-desc">${escapeHtml(m.interpretation)}</p>
      </div>`
    )
    .join("");
}

function renderEvalRunsTable(runs) {
  const tbody = document.getElementById("eval-runs-tbody");
  if (!runs.length) {
    tbody.innerHTML = `<tr><td colspan="10" class="empty-muted">No benchmark runs stored yet. Click "Run Experiment" or "Run 3-Way Ablation Suite" above.</td></tr>`;
    return;
  }

  tbody.innerHTML = runs
    .map(
      (r) => `
      <tr data-run-id="${escapeHtml(r.run_id)}" class="${state.activeEvalRun && state.activeEvalRun.run_id === r.run_id ? "selected-row" : ""}">
        <td><strong>${escapeHtml(r.experiment_name)}</strong></td>
        <td><code>${escapeHtml(r.retrieval_mode)}</code> (k=${r.top_k}, &alpha;=${r.alpha.toFixed(2)})</td>
        <td><strong>${(r.overall_pass_rate * 100).toFixed(1)}%</strong></td>
        <td>${(r.mean_recall_at_k * 100).toFixed(1)}%</td>
        <td>${r.mean_mrr.toFixed(3)}</td>
        <td>${r.mean_ndcg_at_k.toFixed(3)}</td>
        <td>${(r.mean_groundedness * 100).toFixed(1)}%</td>
        <td>${(r.abstention_accuracy * 100).toFixed(1)}%</td>
        <td>${r.latency_p50_ms.toFixed(1)} ms</td>
        <td>$${r.total_reference_cost_usd.toFixed(5)}</td>
      </tr>`
    )
    .join("");

  tbody.querySelectorAll("tr[data-run-id]").forEach((row) => {
    row.addEventListener("click", () => {
      const runId = row.getAttribute("data-run-id");
      const found = state.evalRuns.find((item) => item.run_id === runId);
      if (found) selectEvaluationRun(found);
    });
  });
}

function selectEvaluationRun(run) {
  state.activeEvalRun = run;
  document.querySelectorAll("#eval-runs-tbody tr").forEach((tr) => {
    tr.classList.toggle("selected-row", tr.getAttribute("data-run-id") === run.run_id);
  });

  // Render KPI summary cards
  const kpiGrid = document.getElementById("eval-kpi-grid");
  kpiGrid.innerHTML = `
    <div class="kpi-card">
      <div class="kpi-label"><span>Behavior Pass Rate</span><span class="tag-meas">Measured</span></div>
      <div class="kpi-value">${(run.overall_pass_rate * 100).toFixed(1)}%</div>
      <div class="kpi-sub">Across ${run.total_cases} benchmark cases</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label"><span>Evidence Recall@${run.top_k}</span><span class="tag-meas">Measured</span></div>
      <div class="kpi-value">${(run.mean_recall_at_k * 100).toFixed(1)}%</div>
      <div class="kpi-sub">MRR: ${run.mean_mrr.toFixed(3)} | NDCG: ${run.mean_ndcg_at_k.toFixed(3)}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label"><span>Citation Precision</span><span class="tag-meas">Measured</span></div>
      <div class="kpi-value">${(run.mean_citation_precision * 100).toFixed(1)}%</div>
      <div class="kpi-sub">Mean Groundedness: ${(run.mean_groundedness * 100).toFixed(1)}%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label"><span>Abstention &amp; Conflict</span><span class="tag-meas">Measured</span></div>
      <div class="kpi-value">${(run.abstention_accuracy * 100).toFixed(0)}% / ${(run.conflict_detection_rate * 100).toFixed(0)}%</div>
      <div class="kpi-sub">Unanswerable / Conflict detection</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label"><span>Latency (p50 / p95)</span><span class="tag-meas">Measured</span></div>
      <div class="kpi-value">${run.latency_p50_ms.toFixed(1)} ms</div>
      <div class="kpi-sub">p95: ${run.latency_p95_ms.toFixed(1)} ms (Total: ${run.total_duration_ms.toFixed(0)} ms)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label"><span>Billed vs Ref LLM Cost</span><span class="tag-est">Est.</span></div>
      <div class="kpi-value">$${run.total_actual_cost_usd.toFixed(4)}</div>
      <div class="kpi-sub">Ref Commercial API: $${run.total_reference_cost_usd.toFixed(5)}</div>
    </div>
  `;

  document.getElementById("active-run-cases-title").textContent =
    `Per-Case Benchmark Breakdown — ${run.experiment_name} (${run.run_id})`;

  const casesTbody = document.getElementById("eval-cases-tbody");
  casesTbody.innerHTML = (run.case_results || [])
    .map(
      (c) => `
      <tr>
        <td><code>${escapeHtml(c.case_id)}</code></td>
        <td><span class="preset-tag tag-${c.category === "conflicting" ? "conflict" : c.category === "unanswerable" ? "abstain" : c.category === "multi_hop" ? "multihop" : "factual"}">${escapeHtml(c.category)}</span></td>
        <td>
          <div style="font-weight:600;margin-bottom:0.2rem;">${escapeHtml(c.question)}</div>
          <div style="font-size:0.75rem;color:var(--text-muted);">${escapeHtml(c.answer_preview)}</div>
        </td>
        <td><code>${escapeHtml(c.expected_behavior)}</code></td>
        <td><code>${escapeHtml(c.actual_verdict)}</code></td>
        <td>${(c.recall_at_k * 100).toFixed(0)}%</td>
        <td>${c.mrr.toFixed(2)}</td>
        <td>${(c.term_coverage * 100).toFixed(0)}%</td>
        <td>${(c.groundedness_score * 100).toFixed(0)}%</td>
        <td>${c.latency_ms.toFixed(1)} ms</td>
        <td>
          <span class="claim-badge ${c.passed_behavior_check ? "badge-supported" : "badge-unsupported"}">
            ${c.passed_behavior_check ? "PASS" : "FAIL"}
          </span>
        </td>
      </tr>`
    )
    .join("");
}

async function triggerSingleEvaluation() {
  const btn = document.getElementById("run-single-eval-btn");
  btn.disabled = true;
  btn.textContent = "Evaluating 10 Cases...";

  const mode = document.getElementById("eval-mode-select").value;
  const topK = parseInt(document.getElementById("eval-topk-select").value, 10);
  const nameInput = document.getElementById("eval-exp-name").value.trim();
  const alpha = mode === "bm25" ? 1.0 : mode === "semantic" ? 0.0 : 0.55;

  try {
    const run = await apiFetch("/api/evaluations/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        experiment_name: nameInput || null,
        retrieval_mode: mode,
        top_k: topK,
        alpha: alpha,
        min_evidence_score: 0.18,
        generation_mode: "extractive",
      }),
    });
    showToast(`Benchmark '${run.experiment_name}' completed: ${(run.overall_pass_rate * 100).toFixed(0)}% pass rate.`);
    await loadEvaluationDashboard();
    selectEvaluationRun(run);
  } catch (err) {
    showToast(`Evaluation failed: ${err.message}`, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Experiment";
  }
}

async function triggerAblationSuite() {
  const btn = document.getElementById("run-ablation-suite-btn");
  btn.disabled = true;
  btn.textContent = "Running 3-Way Ablation...";
  const topK = parseInt(document.getElementById("eval-topk-select").value, 10);

  try {
    const runs = await apiFetch(`/api/evaluations/compare-suite?top_k=${topK}`, {
      method: "POST",
    });
    showToast(`Completed 3-way ablation study (${runs.length} experiments).`);
    await loadEvaluationDashboard();
  } catch (err) {
    showToast(`Ablation suite failed: ${err.message}`, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run 3-Way Ablation Suite";
  }
}

// ---------------------------------------------------------------------------
// Application Bootstrap
// ---------------------------------------------------------------------------

async function bootstrap() {
  initTabsAndControls();
  try {
    const health = await apiFetch("/api/health");
    document.getElementById("app-version-pill").textContent = `v${health.version}`;
    if (health.llm_configured) {
      document.getElementById("header-mode-text").textContent = `LLM Configured (${health.llm_model})`;
    } else {
      document.getElementById("header-mode-text").textContent = "Local Extractive Mode ($0.00 API Cost)";
    }
  } catch (_) {
    // ignore
  }

  await Promise.all([
    loadDocuments(),
    loadRecentQueries(),
    loadEvaluationDashboard(),
  ]);

  // Run an initial evaluation automatically if none exist so the dashboard has real computed metrics immediately
  if (state.evalRuns.length === 0) {
    try {
      const initialRun = await apiFetch("/api/evaluations/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          experiment_name: "Default Hybrid Baseline (k=5, α=0.55)",
          retrieval_mode: "hybrid",
          top_k: 5,
          alpha: 0.55,
          min_evidence_score: 0.18,
          generation_mode: "extractive",
        }),
      });
      await loadEvaluationDashboard();
      selectEvaluationRun(initialRun);
    } catch (_) {
      // ignore
    }
  }
}

document.addEventListener("DOMContentLoaded", bootstrap);
