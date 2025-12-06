// Content script - runs on all pages
// This file ensures the extension can access page content
// The actual extraction happens in popup.js via executeScript

console.log('Page Content Downloader extension loaded');

// Listen for messages from popup if needed
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'ping') {
    sendResponse({ status: 'ready' });
  }
  return true;
});