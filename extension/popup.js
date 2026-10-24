

const API_BASE = "http://localhost:5000";

// DOM Selector Helpers
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// Global Elements
const statusIndicator = $("#status-indicator");
const btnKebabMenu = $("#btn-kebab-menu");
const kebabDropdown = $("#kebab-dropdown");
const menuItemHistory = $("#menu-item-history");
const menuItemClearHistory = $("#menu-item-clear-history");
const btnHistoryBack = $("#btn-history-back");
const tabBtns = $$(".tab-nav .tab-btn");
const tabContents = $$(".tab-content");
const loadingOverlay = $("#loading-overlay");
const loadingText = $("#loading-text");

// Text Elements
const textInput = $("#text-input");
const textSourceUrl = $("#text-source-url");
const btnGrabText = $("#btn-grab-text");
const btnAnalyzeText = $("#btn-analyze-text");
const textResults = $("#text-results");
const textVerdict = $("#text-verdict");
const textConfidenceBar = $("#text-confidence-bar");
const textConfidenceValue = $("#text-confidence-value");
const textClaimsSection = $("#text-claims-section");
const textClaimsCountBadge = $("#text-claims-count-badge");
const textClaimsList = $("#text-claims-list");
const textExplanations = $("#text-explanations");
const textSources = $("#text-sources");

// Image Elements
const imageUrlInput = $("#image-url-input");
const imageWebpageUrl = $("#image-webpage-url");
const btnVerifyUrl = $("#btn-verify-url");
const imageFileInput = $("#image-file-input");
const uploadArea = $("#upload-area");
const uploadPreview = $("#upload-preview");
const previewImg = $("#preview-img");
const previewName = $("#preview-name");
const btnUploadAnalyze = $("#btn-upload-analyze");
const btnFreeSearch = $("#btn-free-search");
const imageResults = $("#image-results");
const imageVerdict = $("#image-verdict");
const imageConfidenceBar = $("#image-confidence-bar");
const imageConfidenceValue = $("#image-confidence-value");
const imageDetails = $("#image-details");
const imageOcrBox = $("#image-ocr-box");
const imageOcrText = $("#image-ocr-text");
const imageClaimsList = $("#image-claims-list");
const imageExplanations = $("#image-explanations");
const imageSourcesWrapper = $("#image-sources-wrapper");
const imageSources = $("#image-sources");
const imageResultPreviewContainer = $("#image-result-preview-container");
const imageResultPreview = $("#image-result-preview");

// History Elements
const historyList = $("#history-list");
const historySearchInput = $("#history-search-input");
const btnHistorySearch = $("#btn-history-search");
const historyFilterPills = $$(".history-filter-pills .pill-btn");
const historyPagination = $("#history-pagination");
const btnHistoryPrev = $("#btn-history-prev");
const btnHistoryNext = $("#btn-history-next");
const historyPageInfo = $("#history-page-info");

let selectedFile = null;
let lastActiveTab = "text";
let currentHistoryFilter = "all";
let currentHistoryPage = 1;
let currentHistoryQuery = "";
let totalHistoryPages = 1;

// =============================================
//  INITIALIZATION
// =============================================

document.addEventListener("DOMContentLoaded", () => {
  checkBackend();
  setupTabs();
  setupKebabMenu();
  setupText();
  setupImage();
  setupHistory();
  checkPendingAction();
});

async function checkPendingAction() {
  try {
    const data = await chrome.storage.local.get([
      "pendingAction",
      "pendingImageUrl",
      "pendingText",
      "pendingSourceUrl",
      "pendingPageTitle",
    ]);
    if (!data.pendingAction) return;

    await chrome.storage.local.remove([
      "pendingAction",
      "pendingImageUrl",
      "pendingText",
      "pendingSourceUrl",
      "pendingPageTitle",
    ]);

    if (data.pendingAction === "verify-image" && data.pendingImageUrl) {
      switchTab("image");
      imageUrlInput.value = data.pendingImageUrl;
      if (data.pendingSourceUrl) imageWebpageUrl.value = data.pendingSourceUrl;
      btnFreeSearch.style.display = "block";
      btnFreeSearch.onclick = () =>
        chrome.tabs.create({ url: `https://lens.google.com/uploadbyurl?url=${encodeURIComponent(data.pendingImageUrl)}` });
      verifyImgUrl(data.pendingImageUrl, data.pendingSourceUrl, data.pendingPageTitle);
    }

    if (data.pendingAction === "verify-text" && data.pendingText) {
      switchTab("text");
      textInput.value = data.pendingText;
      if (data.pendingSourceUrl) textSourceUrl.value = data.pendingSourceUrl;
    }
  } catch (e) {
    console.warn("[FactGuard] Error reading pending action:", e);
  }
}

async function checkBackend() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      statusIndicator.className = "status-dot status-online";
      statusIndicator.title = "FactGuard Backend: Connected";
    } else {
      statusIndicator.className = "status-dot status-offline";
      statusIndicator.title = "FactGuard Backend: Degraded";
    }
  } catch {
    statusIndicator.className = "status-dot status-offline";
    statusIndicator.title = "FactGuard Backend: Offline (Start python backend/app.py)";
  }
}

function switchTab(tabId) {
  if (kebabDropdown) kebabDropdown.classList.add("hidden");
  if (btnKebabMenu) btnKebabMenu.classList.remove("active");

  if (tabId !== "history") {
    lastActiveTab = tabId;
  }

  tabBtns.forEach((b) => b.classList.remove("active"));
  tabContents.forEach((c) => c.classList.remove("active"));

  const targetBtn = $(`#tab-${tabId}-btn`);
  const targetContent = $(`#tab-${tabId}`);
  if (targetBtn) targetBtn.classList.add("active");
  if (targetContent) targetContent.classList.add("active");

  if (tabId === "history") {
    loadHistory();
  }
}

function setupTabs() {
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      switchTab(btn.dataset.tab);
    });
  });
}

function setupKebabMenu() {
  if (btnKebabMenu) {
    btnKebabMenu.addEventListener("click", (e) => {
      e.stopPropagation();
      kebabDropdown?.classList.toggle("hidden");
      btnKebabMenu.classList.toggle("active");
    });
  }

  if (menuItemHistory) {
    menuItemHistory.addEventListener("click", () => {
      switchTab("history");
    });
  }

  if (menuItemClearHistory) {
    menuItemClearHistory.addEventListener("click", async () => {
      kebabDropdown?.classList.add("hidden");
      btnKebabMenu?.classList.remove("active");
      if (!confirm("Are you sure you want to clear all verification history? This cannot be undone.")) return;

      try {
        const res = await fetch(`${API_BASE}/history`, { method: "DELETE" });
        if (res.ok) {
          currentHistoryPage = 1;
          loadHistory();
        } else {
          alert("Failed to clear history.");
        }
      } catch {
        alert("Error connecting to backend.");
      }
    });
  }

  if (btnHistoryBack) {
    btnHistoryBack.addEventListener("click", () => {
      switchTab(lastActiveTab || "text");
    });
  }

  document.addEventListener("click", (e) => {
    if (kebabDropdown && !kebabDropdown.contains(e.target) && !btnKebabMenu?.contains(e.target)) {
      kebabDropdown.classList.add("hidden");
      btnKebabMenu?.classList.remove("active");
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && kebabDropdown && !kebabDropdown.classList.contains("hidden")) {
      kebabDropdown.classList.add("hidden");
      btnKebabMenu?.classList.remove("active");
    }
  });
}

// =============================================
//  TEXT FACT-CHECKING
// =============================================

function setupText() {
  btnGrabText.addEventListener("click", async () => {
    const originalText = btnGrabText.textContent;
    btnGrabText.textContent = "Grabbing...";
    btnGrabText.disabled = true;

    try {
      let tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tabs.length || tabs[0].url?.startsWith("chrome-extension://")) {
        tabs = await chrome.tabs.query({ active: true });
        tabs = tabs.filter((t) => t.url && !t.url.startsWith("chrome-extension://") && !t.url.startsWith("chrome://"));
      }

      if (!tabs.length) {
        textInput.placeholder = "No active webpage found. Highlight text on a page first.";
        return;
      }

      const tab = tabs[0];
      if (tab.url && !textSourceUrl.value) {
        textSourceUrl.value = tab.url;
      }

      let grabbed = "";
      try {
        const res = await chrome.tabs.sendMessage(tab.id, { action: "GET_SELECTED_TEXT" });
        grabbed = res?.text?.trim() || "";
      } catch (_) {}

      if (!grabbed) {
        try {
          const [result] = await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: () => window.getSelection().toString().trim(),
          });
          grabbed = result?.result?.trim() || "";
        } catch (_) {}
      }

      if (grabbed) {
        textInput.value = grabbed;
        textInput.placeholder = "";
      } else {
        textInput.placeholder = "No text selected. Highlight text on the page, then click Grab.";
      }
    } catch (e) {
      console.warn("[FactGuard] Grab text error:", e);
      textInput.placeholder = "Error grabbing text. Please select text manually.";
    } finally {
      btnGrabText.textContent = originalText;
      btnGrabText.disabled = false;
    }
  });

  btnAnalyzeText.addEventListener("click", async () => {
    const text = textInput.value.trim();
    if (text.length < 10) {
      alert("Please provide at least 10 characters to analyze.");
      return;
    }

    const sourceUrl = textSourceUrl.value.trim() || null;
    let webpageTitle = null;

    // Try to auto-grab webpage title if URL is from active tab
    try {
      const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (activeTab && activeTab.url === sourceUrl) {
        webpageTitle = activeTab.title || null;
      }
    } catch (_) {}

    showLoad("Extracting claims & searching web evidence...");
    textResults.classList.add("hidden");

    try {
      const res = await fetch(`${API_BASE}/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          source_url: sourceUrl,
          webpage_title: webpageTitle,
        }),
        signal: AbortSignal.timeout(35000),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        alert(data.error || `Server error (${res.status})`);
        return;
      }

      renderText(data);
    } catch (e) {
      if (e.name === "TimeoutError") {
        alert("Verification timed out. Web evidence retrieval is searching multiple external sources.");
      } else {
        alert("FactGuard AI backend is unavailable.\nPlease check that the server is running at http://localhost:5000.");
      }
    } finally {
      hideLoad();
    }
  });
}

function renderText(data) {
  textResults.classList.remove("hidden");
  const isFake = data.prediction === "FAKE";
  textVerdict.textContent = isFake ? "LIKELY FAKE / MISLEADING" : "LIKELY AUTHENTIC";
  textVerdict.className = `verdict ${isFake ? "verdict-fake" : "verdict-real"}`;

  const pct = Math.round((data.confidence || 0) * 100);
  textConfidenceValue.textContent = `${pct}%`;
  textConfidenceBar.style.width = `${pct}%`;

  // Render Claim-by-Claim Cards
  renderClaimsList(data.claims || [], textClaimsList, textClaimsCountBadge);

  // Model & Fallback Explanations
  const modeBadge = data.is_free_huggingface_model
    ? `<div class="mode-badge">🤗 Hugging Face Free Model Active (${data.model_name ? data.model_name.split("/").pop() : "BERT"})</div>`
    : "";

  textExplanations.innerHTML =
    modeBadge +
    (data.explanations || [])
      .map((e) => `<div class="explanation-item">${escapeHtml(e)}</div>`)
      .join("");

  // Corroborating Articles
  textSources.innerHTML =
    (data.sources || [])
      .map(
        (s) => `
    <div class="source-item">
      <a href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer" class="source-link">${escapeHtml(s.title || s.url)}</a>
      <span style="font-size:10px; opacity:0.7;"> (${Math.round((s.similarity || 0) * 100)}% relevance)</span>
    </div>
  `
      )
      .join("") || '<div style="font-size:11px; color:var(--text-muted);">No matching news articles found.</div>';
}

// =============================================
//  CLAIM-BY-CLAIM RENDERING HELPER
// =============================================

function renderClaimsList(claims, container, countBadge) {
  if (!claims || claims.length === 0) {
    if (countBadge) countBadge.textContent = "0 Claims";
    container.innerHTML = `
      <div style="font-size:11px; color:var(--text-muted); font-style:italic; padding:6px 0;">
        No individual factual claims extracted from input.
      </div>`;
    return;
  }

  if (countBadge) {
    countBadge.textContent = `${claims.length} Claim${claims.length > 1 ? "s" : ""}`;
  }

  const STATUS_CONFIG = {
    Supported: { cls: "status-supported", icon: "✓" },
    Contradicted: { cls: "status-contradicted", icon: "✗" },
    "Mixed Evidence": { cls: "status-mixed", icon: "⚖" },
    "Insufficient Evidence": { cls: "status-insufficient", icon: "❓" },
    "Not Verifiable": { cls: "status-unverifiable", icon: "⚪" },
  };

  container.innerHTML = claims
    .map((c, idx) => {
      const cfg = STATUS_CONFIG[c.status] || STATUS_CONFIG["Insufficient Evidence"];
      const supporting = c.supporting_evidence || [];
      const contradicting = c.contradicting_evidence || [];

      let evidenceHtml = "";

      if (supporting.length > 0) {
        evidenceHtml += `
          <div class="claim-evidence-group">
            <span class="evidence-header-label" style="color:var(--color-supported);">Supporting Evidence (${supporting.length})</span>
            ${supporting.map((ev) => renderEvidenceCard(ev)).join("")}
          </div>
        `;
      }

      if (contradicting.length > 0) {
        evidenceHtml += `
          <div class="claim-evidence-group">
            <span class="evidence-header-label" style="color:var(--color-contradicted);">Contradicting Evidence (${contradicting.length})</span>
            ${contradicting.map((ev) => renderEvidenceCard(ev)).join("")}
          </div>
        `;
      }

      if (supporting.length === 0 && contradicting.length === 0) {
        evidenceHtml += `
          <div style="font-size:10px; color:var(--text-muted); font-style:italic;">
            No verified external citations found for this specific claim.
          </div>
        `;
      }

      const limitationsHtml = c.uncertainty_or_limitations
        ? `<div class="claim-limitations"><strong>Assessment Note:</strong> ${escapeHtml(c.uncertainty_or_limitations)}</div>`
        : "";

      return `
        <div class="claim-card">
          <div class="claim-header">
            <span class="claim-text">#${idx + 1}. "${escapeHtml(c.claim)}"</span>
            <span class="claim-status ${cfg.cls}">${cfg.icon} ${escapeHtml(c.status)}</span>
          </div>

          <div class="claim-explanation">${escapeHtml(c.explanation || "")}</div>

          ${evidenceHtml}
          ${limitationsHtml}
        </div>
      `;
    })
    .join("");
}

function renderEvidenceCard(ev) {
  const pub = ev.publisher || "Web Source";
  const dateStr = ev.published_date ? ` • ${escapeHtml(ev.published_date)}` : "";
  const tierStr = ev.source_tier ? `<span class="evidence-tier-pill">Tier ${ev.source_tier}</span>` : "";

  return `
    <div class="evidence-entry">
      <div class="evidence-meta-row">
        <span class="evidence-publisher">${escapeHtml(pub)}${dateStr}</span>
        ${tierStr}
      </div>
      <a href="${escapeHtml(ev.url)}" target="_blank" rel="noopener noreferrer" class="evidence-link">
        ${escapeHtml(ev.title || ev.url)}
      </a>
      ${ev.snippet ? `<div class="evidence-snippet">"${escapeHtml(ev.snippet)}"</div>` : ""}
    </div>
  `;
}

// =============================================
//  IMAGE FACT-CHECKING & FORENSICS
// =============================================

function setupImage() {
  btnVerifyUrl.addEventListener("click", () => {
    const url = imageUrlInput.value.trim();
    const pageUrl = imageWebpageUrl.value.trim() || null;
    if (url) {
      verifyImgUrl(url, pageUrl);
      btnFreeSearch.style.display = "block";
      btnFreeSearch.onclick = () =>
        chrome.tabs.create({ url: `https://lens.google.com/uploadbyurl?url=${encodeURIComponent(url)}` });
    }
  });

  uploadArea.addEventListener("click", () => imageFileInput.click());

  imageFileInput.addEventListener("change", (e) => {
    if (e.target.files.length) {
      selectedFile = e.target.files[0];
      previewImg.src = URL.createObjectURL(selectedFile);
      uploadPreview.classList.remove("hidden");
      $(".upload-content")?.classList.add("hidden");
      btnUploadAnalyze.disabled = false;
    }
  });

  btnUploadAnalyze.addEventListener("click", async () => {
    if (!selectedFile) return;
    const pageUrl = imageWebpageUrl.value.trim() || null;

    showLoad("Running OCR, forensics, & evidence discovery...");
    try {
      const formData = new FormData();
      formData.append("image", selectedFile);
      if (pageUrl) {
        formData.append("webpage_url", pageUrl);
      }

      const res = await fetch(`${API_BASE}/upload-image`, {
        method: "POST",
        body: formData,
        signal: AbortSignal.timeout(40000),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        alert(data.error || `Upload analysis error (${res.status})`);
        return;
      }
      renderImage(data);
    } catch (e) {
      if (e.name === "TimeoutError") {
        alert("Image upload analysis timed out.");
      } else {
        alert("FactGuard AI backend is unavailable.\nPlease ensure the backend is running at http://localhost:5000.");
      }
    } finally {
      hideLoad();
    }
  });
}

async function verifyImgUrl(url, webpageUrl = null, webpageTitle = null) {
  showLoad("Verifying image URL, OCR claims, & forensics...");
  try {
    const res = await fetch(`${API_BASE}/verify-image-url`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url,
        webpage_url: webpageUrl,
        webpage_title: webpageTitle,
      }),
      signal: AbortSignal.timeout(40000),
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      alert(data.error || `Verification error (${res.status})`);
      return;
    }
    renderImage(data, url);
  } catch (e) {
    if (e.name === "TimeoutError") {
      alert("Image URL analysis timed out.");
    } else {
      alert("FactGuard AI backend is unavailable.\nPlease ensure the backend is running at http://localhost:5000.");
    }
  } finally {
    hideLoad();
  }
}

function renderImage(data, sourceUrl = null) {
  imageResults.classList.remove("hidden");

  if (sourceUrl) {
    imageResultPreview.src = sourceUrl;
    imageResultPreviewContainer.classList.remove("hidden");
  } else if (selectedFile) {
    imageResultPreview.src = URL.createObjectURL(selectedFile);
    imageResultPreviewContainer.classList.remove("hidden");
  } else {
    imageResultPreviewContainer.classList.add("hidden");
  }

  const CATEGORY_MAP = {
    ai_generated: {
      text: "🤖 AI-GENERATED MEDIA",
      cls: "verdict-ai",
      barColor: "#9c27b0",
      scoreLabel: "AI Forensic Confidence",
    },
    edited: {
      text: "✂️ EDITED / TAMPERED",
      cls: "verdict-edited",
      barColor: "#ff9800",
      scoreLabel: "Tampering Score",
    },
    fake: {
      text: "⛔ MISLEADING CONTEXT",
      cls: "verdict-fake",
      barColor: "#f44336",
      scoreLabel: "Risk Score",
    },
    unverified: {
      text: "⚠️ UNVERIFIED / UNCORROBORATED",
      cls: "verdict-unverified",
      barColor: "#8b5cf6",
      scoreLabel: "Uncertainty Level",
    },
    real: {
      text: "✅ AUTHENTIC MEDIA",
      cls: "verdict-real",
      barColor: "#4caf50",
      scoreLabel: "Confidence",
    },
  };

  const category = data.image_category || "real";
  const catData = CATEGORY_MAP[category] || CATEGORY_MAP.real;
  const catConf = Math.round((data.category_confidence || 0) * 100);
  const aiPct = Math.round((data.visual_forensics?.ai_confidence ?? data.ai_confidence ?? 0) * 100);
  const editPct = Math.round((data.visual_forensics?.edit_score ?? data.edit_score ?? 0) * 100);

  imageVerdict.textContent = catData.text;
  imageVerdict.className = `verdict ${catData.cls}`;

  const barLabel = $("#image-confidence-wrapper .confidence-label");
  if (barLabel) barLabel.textContent = catData.scoreLabel;
  imageConfidenceValue.textContent = `${catConf}%`;
  imageConfidenceBar.style.width = `${catConf}%`;
  imageConfidenceBar.style.background = catData.barColor;

  const webMatches = data.web_full_matches || 0;
  const reuseCount = data.similar_count || 0;

  imageDetails.innerHTML = `
    <div class="detail-item">
      <div class="detail-label">Visual Status</div>
      <div class="detail-value" style="color:${catData.barColor};font-size:10px;font-weight:700;">${escapeHtml(catData.text)}</div>
    </div>
    <div class="detail-item">
      <div class="detail-label">AI Score</div>
      <div class="detail-value" style="color:${aiPct >= 75 ? "#9c27b0" : "#4caf50"};font-weight:700;">${aiPct}%</div>
    </div>
    <div class="detail-item">
      <div class="detail-label">Edit Score</div>
      <div class="detail-value" style="color:${editPct >= 55 ? "#ff9800" : "#4caf50"};font-weight:700;">${editPct}%</div>
    </div>
    <div class="detail-item">
      <div class="detail-label">Detection Mode</div>
      <div class="detail-value" style="font-size:10px;">${escapeHtml(data.verification_mode || "HF ViT")}</div>
    </div>
    <div class="detail-item">
      <div class="detail-label">Web Appearances</div>
      <div class="detail-value">${webMatches}</div>
    </div>
    <div class="detail-item">
      <div class="detail-label">pHash Reuse</div>
      <div class="detail-value" style="color:${reuseCount > 0 ? "#f44336" : "#4caf50"}">${reuseCount}</div>
    </div>`;

  // OCR Text & Claims in Image
  if (data.ocr_text && data.ocr_text.trim()) {
    imageOcrBox.classList.remove("hidden");
    imageOcrText.textContent = data.ocr_text.trim();
  } else {
    imageOcrBox.classList.add("hidden");
  }

  // Render Claims extracted from image
  renderClaimsList(data.claims || [], imageClaimsList, null);

  // Forensic assessment explanations
  const imgModeBadge = data.is_free_huggingface_model
    ? `<div class="mode-badge">🤗 Hugging Face Free ViT (${data.verification_mode === "api_hybrid" ? "Hybrid API" : "Autonomous Fallback"})</div>`
    : "";

  imageExplanations.innerHTML =
    imgModeBadge +
    (data.explanations || []).map((e) => `<div class="explanation-item">${escapeHtml(e)}</div>`).join("");

  // Matching Image Appearances
  const matches = [...(data.web_search?.full_matching_images || []), ...(data.web_search?.partial_matching_images || [])];
  if (matches.length) {
    imageSourcesWrapper.style.display = "block";
    imageSources.className = "image-matches-grid";
    imageSources.innerHTML = matches
      .slice(0, 10)
      .map(
        (m) => `
      <a href="${escapeHtml(m.url)}" target="_blank" rel="noopener noreferrer"><img src="${escapeHtml(m.url)}" onerror="this.style.display='none'"></a>
    `
      )
      .join("");
  } else {
    imageSourcesWrapper.style.display = "none";
  }
}

// =============================================
//  HISTORY & SOURCE AUDIT LOG
// =============================================

function setupHistory() {
  historyFilterPills.forEach((pill) => {
    pill.addEventListener("click", () => {
      historyFilterPills.forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      currentHistoryFilter = pill.dataset.type;
      currentHistoryPage = 1;
      loadHistory();
    });
  });

  btnHistorySearch.addEventListener("click", () => {
    currentHistoryQuery = historySearchInput.value.trim();
    currentHistoryPage = 1;
    loadHistory();
  });

  historySearchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      currentHistoryQuery = historySearchInput.value.trim();
      currentHistoryPage = 1;
      loadHistory();
    }
  });

  btnHistoryPrev.addEventListener("click", () => {
    if (currentHistoryPage > 1) {
      currentHistoryPage--;
      loadHistory();
    }
  });

  btnHistoryNext.addEventListener("click", () => {
    if (currentHistoryPage < totalHistoryPages) {
      currentHistoryPage++;
      loadHistory();
    }
  });
}

async function loadHistory() {
  historyList.innerHTML = `<div style="text-align:center; padding:16px; color:var(--text-muted); font-size:12px;">Loading history...</div>`;
  historyPagination.classList.add("hidden");

  try {
    const params = new URLSearchParams({
      item_type: currentHistoryFilter,
      query: currentHistoryQuery,
      page: currentHistoryPage,
      per_page: 5,
    });

    const res = await fetch(`${API_BASE}/history?${params.toString()}`, {
      signal: AbortSignal.timeout(10000),
    });

    if (!res.ok) {
      historyList.innerHTML = `<div class="history-empty">Failed to load history (${res.status}).</div>`;
      return;
    }

    const data = await res.json();
    totalHistoryPages = data.total_pages || 1;
    renderHistoryItems(data.items || []);

    if (totalHistoryPages > 1) {
      historyPagination.classList.remove("hidden");
      historyPageInfo.textContent = `Page ${data.page} of ${totalHistoryPages} (${data.total} items)`;
      btnHistoryPrev.disabled = data.page <= 1;
      btnHistoryNext.disabled = data.page >= totalHistoryPages;
    } else {
      historyPagination.classList.add("hidden");
    }
  } catch (e) {
    historyList.innerHTML = `<div class="history-empty">Backend offline. Cannot load verification history.</div>`;
  }
}

function renderHistoryItems(items) {
  if (!items || items.length === 0) {
    historyList.innerHTML = `<div class="history-empty">No verification records found.</div>`;
    return;
  }

  historyList.innerHTML = items
    .map((item) => {
      const isImg = item.item_type === "image";
      const typeLabel = isImg ? "🖼️ Image" : "📝 Text";
      const dateStr = item.created_at ? new Date(item.created_at).toLocaleDateString() : "";
      const sourceUrl = item.source_url || item.webpage_url;

      let sourceHtml = `<span style="opacity:0.7;">Source: Unknown</span>`;
      if (sourceUrl) {
        sourceHtml = `Source: <a href="${escapeHtml(sourceUrl)}" target="_blank" rel="noopener noreferrer" class="history-source-link">${escapeHtml(item.webpage_title || sourceUrl)}</a>`;
      }

      // Claims count and verdicts
      const claims = item.claims || [];
      let claimSummaryHtml = "";
      if (claims.length > 0) {
        claimSummaryHtml = `<span style="font-size:10px; font-weight:700;">${claims.length} Claim(s): ${claims.map((c) => c.status).slice(0, 2).join(", ")}</span>`;
      } else if (item.top_verdict) {
        claimSummaryHtml = `<span style="font-size:10px; font-weight:700;">Verdict: ${escapeHtml(item.top_verdict)}</span>`;
      }

      const displayText = item.submitted_text || (isImg ? item.image_identifier : "No text");

      return `
        <div class="history-card" data-id="${item.id}">
          <div class="history-card-top">
            <span class="history-type-badge">${typeLabel}</span>
            <span class="history-date">${escapeHtml(dateStr)}</span>
          </div>

          <div class="history-card-body">
            "${escapeHtml(displayText)}"
          </div>

          <div class="history-source-info">
            ${sourceHtml}
          </div>

          <div class="history-card-footer">
            <div class="history-verdict-summary">
              ${claimSummaryHtml}
            </div>
            <div class="history-actions">
              <button class="btn btn-secondary btn-sm btn-details" data-id="${item.id}">Details</button>
              <button class="btn btn-danger btn-sm btn-delete" data-id="${item.id}">Delete</button>
            </div>
          </div>

          <div class="history-expanded-details hidden" id="history-details-${item.id}">
            <!-- Dynamically populated on expand -->
          </div>
        </div>
      `;
    })
    .join("");

  // Attach event handlers for Delete and Details
  $$(".btn-delete").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      if (!confirm("Are you sure you want to delete this verification record?")) return;

      try {
        const res = await fetch(`${API_BASE}/history/${id}`, { method: "DELETE" });
        if (res.ok) {
          loadHistory();
        } else {
          alert("Failed to delete record.");
        }
      } catch {
        alert("Error connecting to backend.");
      }
    });
  });

  $$(".btn-details").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      const detailsContainer = $(`#history-details-${id}`);
      if (!detailsContainer) return;

      if (!detailsContainer.classList.contains("hidden")) {
        detailsContainer.classList.add("hidden");
        btn.textContent = "Details";
        return;
      }

      btn.textContent = "Loading...";
      try {
        const res = await fetch(`${API_BASE}/history/${id}`);
        if (!res.ok) throw new Error();
        const data = await res.json();
        const entry = data.entry;

        let detailsHtml = '<div style="margin-top:8px; border-top:1px dashed var(--border); padding-top:8px;">';

        if (entry.claims && entry.claims.length > 0) {
          detailsHtml += '<div style="font-size:11px; font-weight:700; margin-bottom:6px;">Verified Claims:</div>';
          detailsHtml += '<div class="claims-list">';
          detailsHtml += entry.claims
            .map(
              (c) => `
            <div class="claim-card" style="padding:6px;">
              <div style="font-size:11px; font-weight:600;">${escapeHtml(c.claim)}</div>
              <div style="font-size:10px; color:var(--text-muted); margin-top:2px;">Status: <strong>${escapeHtml(c.status)}</strong></div>
              ${
                c.supporting_evidence?.length
                  ? `<div style="font-size:9px; margin-top:2px;">Supporting: ${c.supporting_evidence
                      .map((ev) => `<a href="${escapeHtml(ev.url)}" target="_blank" class="source-link">${escapeHtml(ev.publisher || ev.url)}</a>`)
                      .join(", ")}</div>`
                  : ""
              }
              ${
                c.contradicting_evidence?.length
                  ? `<div style="font-size:9px; margin-top:2px;">Contradicting: ${c.contradicting_evidence
                      .map((ev) => `<a href="${escapeHtml(ev.url)}" target="_blank" class="source-link">${escapeHtml(ev.publisher || ev.url)}</a>`)
                      .join(", ")}</div>`
                  : ""
              }
            </div>
          `
            )
            .join("");
          detailsHtml += "</div>";
        } else {
          detailsHtml += `<div style="font-size:11px; color:var(--text-muted);">No decomposed claims recorded.</div>`;
        }

        detailsHtml += "</div>";
        detailsContainer.innerHTML = detailsHtml;
        detailsContainer.classList.remove("hidden");
        btn.textContent = "Hide";
      } catch {
        alert("Failed to load full record details.");
        btn.textContent = "Details";
      }
    });
  });
}

// =============================================
//  HELPERS
// =============================================

function showLoad(msg) {
  loadingText.textContent = msg;
  loadingOverlay.classList.remove("hidden");
}

function hideLoad() {
  loadingOverlay.classList.add("hidden");
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
