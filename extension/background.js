/**
 * FactGuard AI — Background Service Worker (Manifest V3)
 * Handles context menus, message routing, and API communication with timeouts.
 */

const API_BASE = "http://localhost:5000";

// =============================================
//  CONTEXT MENU INITIALIZATION
// =============================================

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "factguard-verify-text",
    title: "Verify with FactGuard AI",
    contexts: ["selection"],
  });

  chrome.contextMenus.create({
    id: "factguard-verify-image",
    title: "Check Image with FactGuard AI",
    contexts: ["image"],
  });

  console.log("[FactGuard AI] Context menus initialized.");
});

// =============================================
//  CONTEXT MENU CLICK HANDLER
// =============================================

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "factguard-verify-text" && info.selectionText) {
    await chrome.storage.local.set({
      pendingText: info.selectionText,
      pendingAction: "verify-text",
      pendingSourceUrl: tab?.url || "",
      pendingPageTitle: tab?.title || "",
    });

    chrome.windows.create({
      url: chrome.runtime.getURL("popup.html"),
      type: "popup",
      width: 480,
      height: 640,
    });
  }

  if (info.menuItemId === "factguard-verify-image" && info.srcUrl) {
    await chrome.storage.local.set({
      pendingImageUrl: info.srcUrl,
      pendingAction: "verify-image",
      pendingSourceUrl: tab?.url || "",
      pendingPageTitle: tab?.title || "",
    });

    chrome.windows.create({
      url: chrome.runtime.getURL("popup.html"),
      type: "popup",
      width: 480,
      height: 640,
    });
  }
});

// =============================================
//  MESSAGE HANDLER
// =============================================

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "VERIFY_TEXT") {
    verifyText(message.text, message.source_url, message.webpage_title)
      .then((result) => sendResponse({ success: true, data: result }))
      .catch((err) => sendResponse({ success: false, error: err.message }));
    return true;
  }

  if (message.action === "VERIFY_IMAGE_URL") {
    verifyImageUrl(message.url, message.webpage_url, message.webpage_title)
      .then((result) => sendResponse({ success: true, data: result }))
      .catch((err) => sendResponse({ success: false, error: err.message }));
    return true;
  }

  if (message.action === "CHECK_BACKEND") {
    checkBackendHealth()
      .then((result) => sendResponse({ success: true, data: result }))
      .catch((err) => sendResponse({ success: false, error: err.message }));
    return true;
  }

  if (message.action === "CLEAR_BADGE") {
    chrome.action.setBadgeText({ text: "" });
    sendResponse({ success: true });
    return false;
  }
});

// =============================================
//  API FUNCTIONS WITH TIMEOUTS
// =============================================

async function verifyText(text, source_url = null, webpage_title = null) {
  try {
    const payload = { text };
    if (source_url) payload.source_url = source_url;
    if (webpage_title) payload.webpage_title = webpage_title;

    const response = await fetch(`${API_BASE}/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(25000),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.error || `Server returned error status ${response.status}`);
    }

    return await response.json();
  } catch (err) {
    if (err.name === "TimeoutError") {
      throw new Error("Request to FactGuard backend timed out after 25 seconds.");
    }
    throw err;
  }
}

async function verifyImageUrl(url, webpage_url = null, webpage_title = null) {
  try {
    const payload = { url };
    if (webpage_url) payload.webpage_url = webpage_url;
    if (webpage_title) payload.webpage_title = webpage_title;

    const response = await fetch(`${API_BASE}/verify-image-url`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(30000),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.error || `Server returned error status ${response.status}`);
    }

    return await response.json();
  } catch (err) {
    if (err.name === "TimeoutError") {
      throw new Error("Image verification timed out after 30 seconds.");
    }
    throw err;
  }
}

async function checkBackendHealth() {
  const response = await fetch(`${API_BASE}/health`, {
    method: "GET",
    signal: AbortSignal.timeout(4000),
  });

  if (!response.ok) {
    throw new Error("Backend service returned an unhealthy status.");
  }

  return response.json();
}
