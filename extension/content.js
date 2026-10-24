/**
 * FactGuard AI — Content Script (Manifest V3)
 * Handles text selection caching, in-page image collection,
 * rate-limited page scanning queue, and 4-way visual verification badges.
 */

(() => {
  let lastClickedImage = null;
  let lastSelectedText = "";
  const scannedUrlsCache = new Map(); // Cache results for the page session

  // Cache text selection continuously so clicking popup button doesn't clear it
  document.addEventListener("selectionchange", () => {
    const sel = window.getSelection().toString().trim();
    if (sel.length > 0) {
      lastSelectedText = sel;
    }
  });

  // Track right-clicked images for context menus
  document.addEventListener("contextmenu", (e) => {
    if (e.target.tagName === "IMG") {
      lastClickedImage = e.target.src;
    }
  });

  // =============================================
  //  MESSAGE ROUTING
  // =============================================

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    switch (message.action) {
      case "GET_SELECTED_TEXT": {
        const liveText = window.getSelection().toString().trim();
        const textToReturn = liveText || lastSelectedText;
        sendResponse({ text: textToReturn });
        if (textToReturn) lastSelectedText = "";
        break;
      }

      case "GET_CLICKED_IMAGE": {
        sendResponse({ url: lastClickedImage || "" });
        break;
      }

      case "GET_PAGE_IMAGES": {
        const images = getPageImages();
        sendResponse({ images });
        break;
      }

      case "SCAN_PAGE_IMAGES": {
        scanPageImagesWithConcurrency(message.apiBase || "http://localhost:5000", 2);
        sendResponse({ started: true });
        break;
      }

      case "HIGHLIGHT_IMAGE": {
        highlightImageByUrl(message.url, message.result);
        sendResponse({ done: true });
        break;
      }

      case "REMOVE_BADGES": {
        removeBadges();
        sendResponse({ done: true });
        break;
      }

      default:
        sendResponse({ error: "Unknown action" });
    }
    return true;
  });

  // =============================================
  //  IMAGE COLLECTION & FILTERING
  // =============================================

  function getPageImages() {
    const imgs = Array.from(document.querySelectorAll("img"));
    const images = [];
    const seen = new Set();

    for (const img of imgs) {
      const src = img.src || img.dataset.src || "";
      if (!src || !src.startsWith("http")) continue;

      // Filter out tiny icons, SVGs, data URIs, and ads
      const isSvg = src.toLowerCase().endsWith(".svg") || src.includes(".svg?");
      const isTiny = (img.naturalWidth > 0 && img.naturalWidth < 120) || (img.naturalHeight > 0 && img.naturalHeight < 120);
      const isAd = img.closest("header, footer, nav, aside, [class*='ad-'], [class*='avatar'], [class*='icon']") !== null;

      if (isSvg || isTiny || isAd || seen.has(src)) continue;

      seen.add(src);
      images.push({
        url: src,
        alt: img.alt || "",
        width: img.naturalWidth || img.clientWidth,
        height: img.naturalHeight || img.clientHeight,
        area: (img.naturalWidth || img.clientWidth) * (img.naturalHeight || img.clientHeight),
      });
    }

    // Sort by largest image first and cap at top 15 images to prevent flooding
    images.sort((a, b) => b.area - a.area);
    return images.slice(0, 15);
  }

  // =============================================
  //  CONTROLLED CONCURRENCY QUEUE
  // =============================================

  async function scanPageImagesWithConcurrency(apiBase, concurrency = 2) {
    const images = getPageImages();

    if (images.length === 0) {
      showNotification("No scannable content images found on this page.", "info");
      return;
    }

    showNotification(
      `Scanning top ${images.length} prominent image(s)...`,
      "info"
    );

    let processed = 0;
    let suspicious = 0;
    let currentIndex = 0;

    async function worker() {
      while (currentIndex < images.length) {
        const item = images[currentIndex++];
        if (!item) break;

        try {
          let result = scannedUrlsCache.get(item.url);

          if (!result) {
            // Fetch from backend with 12s timeout
            const response = await fetch(`${apiBase}/verify-image-url`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                url: item.url,
                webpage_url: window.location.href,
                webpage_title: document.title,
              }),
              signal: AbortSignal.timeout(15000),
            });

            if (response.ok) {
              result = await response.json();
              scannedUrlsCache.set(item.url, result);
            }
          }

          if (result) {
            highlightImageByUrl(item.url, result);
            const cat = result.image_category || "";
            if (cat === "ai_generated" || cat === "edited" || cat === "fake" || result.is_suspicious) {
              suspicious++;
            }
          }
        } catch (err) {
          console.warn(`[FactGuard] Failed scanning ${item.url}:`, err);
        } finally {
          processed++;
          try {
            chrome.runtime.sendMessage({
              action: "SCAN_PROGRESS",
              processed,
              total: images.length,
              suspicious,
            });
          } catch (_) {}
        }
      }
    }

    // Launch worker pool
    const workers = [];
    for (let i = 0; i < concurrency; i++) {
      workers.push(worker());
    }
    await Promise.all(workers);

    showNotification(
      `Scan complete: ${processed} images checked, ${suspicious} suspicious.`,
      suspicious > 0 ? "warning" : "success"
    );
  }

  // =============================================
  //  VISUAL BADGES & 4-WAY HIGHLIGHTING
  // =============================================

  function highlightImageByUrl(url, result) {
    const imgs = document.querySelectorAll("img");
    imgs.forEach((img) => {
      if (img.src === url || img.dataset.src === url) {
        applyBadge(img, result);
      }
    });
  }

  function applyBadge(imgElement, result) {
    const existingBadge = imgElement.parentElement?.querySelector(".factguard-badge");
    if (existingBadge) existingBadge.remove();

    const category = result.image_category || (result.is_ai_generated ? "ai_generated" : (result.is_reused ? "fake" : "real"));

    const BADGE_CONFIG = {
      ai_generated: {
        label: "AI 🤖",
        outline: "3px solid #9c27b0",
        bg: "rgba(156, 39, 176, 0.95)",
        color: "#fff",
      },
      edited: {
        label: "EDITED ✂️",
        outline: "3px solid #ff9800",
        bg: "rgba(255, 152, 0, 0.95)",
        color: "#fff",
      },
      fake: {
        label: "MISLEADING ⛔",
        outline: "3px solid #f44336",
        bg: "rgba(244, 67, 54, 0.95)",
        color: "#fff",
      },
      real: {
        label: "REAL ✅",
        outline: "3px solid #4caf50",
        bg: "rgba(76, 175, 80, 0.95)",
        color: "#fff",
      },
    };

    const cfg = BADGE_CONFIG[category] || BADGE_CONFIG.real;

    let wrapper = imgElement.parentElement;
    if (!wrapper?.classList.contains("factguard-wrapper")) {
      wrapper = document.createElement("div");
      wrapper.className = "factguard-wrapper";
      wrapper.style.cssText = `
        position: relative;
        display: inline-block;
      `;
      imgElement.parentNode.insertBefore(wrapper, imgElement);
      wrapper.appendChild(imgElement);
    }

    imgElement.style.outline = cfg.outline;
    imgElement.style.outlineOffset = "-3px";

    const badge = document.createElement("div");
    badge.className = "factguard-badge";
    badge.style.cssText = `
      position: absolute;
      top: 8px;
      right: 8px;
      padding: 4px 10px;
      border-radius: 6px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.5px;
      z-index: 10000;
      pointer-events: auto;
      cursor: default;
      box-shadow: 0 2px 8px rgba(0,0,0,0.35);
      backdrop-filter: blur(4px);
      background: ${cfg.bg};
      color: ${cfg.color};
      transition: transform 0.2s ease;
    `;
    badge.textContent = cfg.label;

    // Hover tooltip
    const tooltip = document.createElement("div");
    tooltip.className = "factguard-tooltip";
    tooltip.style.cssText = `
      position: absolute;
      top: 36px;
      right: 8px;
      max-width: 280px;
      padding: 10px 14px;
      background: rgba(15, 15, 26, 0.95);
      color: #e0e0e0;
      border: 1px solid rgba(0, 212, 255, 0.3);
      border-radius: 8px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-size: 11px;
      line-height: 1.45;
      z-index: 10001;
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.2s ease;
      box-shadow: 0 4px 16px rgba(0,0,0,0.5);
    `;

    const explanations = result.explanations || [result.category_reason || ""];
    tooltip.innerHTML = explanations
      .slice(0, 3)
      .map((e) => `<div style="margin-bottom: 4px;">${e}</div>`)
      .join("");

    badge.addEventListener("mouseenter", () => {
      badge.style.transform = "scale(1.08)";
      tooltip.style.opacity = "1";
    });
    badge.addEventListener("mouseleave", () => {
      badge.style.transform = "scale(1)";
      tooltip.style.opacity = "0";
    });

    wrapper.appendChild(badge);
    wrapper.appendChild(tooltip);
  }

  function removeBadges() {
    document.querySelectorAll(".factguard-badge, .factguard-tooltip").forEach((el) => el.remove());
    document.querySelectorAll("img").forEach((img) => {
      img.style.outline = "";
      img.style.outlineOffset = "";
    });
    document.querySelectorAll(".factguard-wrapper").forEach((wrapper) => {
      const img = wrapper.querySelector("img");
      if (img) {
        wrapper.parentNode.insertBefore(img, wrapper);
        wrapper.remove();
      }
    });
  }

  // Toast Notification Helper
  function showNotification(message, type = "info") {
    const existing = document.querySelector(".factguard-notification");
    if (existing) existing.remove();

    const colors = {
      info: { bg: "rgba(0, 212, 255, 0.95)", text: "#000" },
      success: { bg: "rgba(0, 255, 136, 0.95)", text: "#000" },
      warning: { bg: "rgba(255, 165, 2, 0.95)", text: "#000" },
      error: { bg: "rgba(255, 71, 87, 0.95)", text: "#fff" },
    };
    const color = colors[type] || colors.info;

    const toast = document.createElement("div");
    toast.className = "factguard-notification";
    toast.textContent = `🛡️ ${message}`;
    toast.style.cssText = `
      position: fixed;
      bottom: 24px;
      right: 24px;
      padding: 12px 20px;
      background: ${color.bg};
      color: ${color.text};
      border-radius: 10px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-size: 13px;
      font-weight: 600;
      z-index: 99999;
      box-shadow: 0 4px 16px rgba(0,0,0,0.3);
      animation: factguardSlideIn 0.3s ease;
      max-width: 400px;
    `;

    document.body.appendChild(toast);
    setTimeout(() => {
      toast.remove();
    }, 4000);
  }
})();
