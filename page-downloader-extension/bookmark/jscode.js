javascript:(function(){
  // Enhanced complete page capture with shadow DOM & iframes
  
  function traverseAllElements(root, depth = 0) {
    let content = [];
    const MAX_DEPTH = 10;
    
    if (depth > MAX_DEPTH) return content;
    
    // Get all elements including shadow DOM
    const walker = document.createTreeWalker(
      root,
      NodeFilter.SHOW_ELEMENT,
      null,
      false
    );
    
    let node;
    while (node = walker.nextNode()) {
      // Capture shadow DOM recursively
      if (node.shadowRoot) {
        content.push(`\n<!-- SHADOW DOM: ${node.tagName} -->`);
        content.push(node.shadowRoot.innerHTML);
        content.push(...traverseAllElements(node.shadowRoot, depth + 1));
      }
    }
    
    return content;
  }
  
  // Main capture function
  async function captureComplete() {
    console.log('🔄 Capturing page...');
    
    // Wait for any pending lazy loads
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    let html = document.documentElement.outerHTML;
    
    // Add shadow DOM content
    const shadowContent = traverseAllElements(document.documentElement);
    if (shadowContent.length > 0) {
      html += '\n\n<!-- SHADOW DOM CONTENT -->\n' + shadowContent.join('\n');
    }
    
    // Capture iframes
    const iframes = document.querySelectorAll('iframe');
    iframes.forEach((iframe, idx) => {
      try {
        const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
        if (iframeDoc) {
          html += `\n\n<!-- IFRAME ${idx}: ${iframe.src || 'inline'} -->\n`;
          html += iframeDoc.documentElement.outerHTML;
        }
      } catch(e) {
        html += `\n<!-- IFRAME ${idx}: Cross-origin blocked -->\n`;
      }
    });
    
    // Capture window state (React/Vue apps)
    const stateData = {};
    try {
      // Common state locations
      if (window.__INITIAL_STATE__) stateData.initialState = window.__INITIAL_STATE__;
      if (window.__PRELOADED_STATE__) stateData.preloadedState = window.__PRELOADED_STATE__;
      if (window.__NEXT_DATA__) stateData.nextData = window.__NEXT_DATA__;
      
      if (Object.keys(stateData).length > 0) {
        html += '\n\n<!-- APPLICATION STATE -->\n';
        html += `<script type="application/json">${JSON.stringify(stateData, null, 2)}</script>`;
      }
    } catch(e) { /* State extraction failed */ }
    
    // Create filename with consistent format for Python script
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('.')[0];
    const filename = `page_${timestamp}.html`;
    
    // Download
    const blob = new Blob([html], {type: 'text/html;charset=utf-8'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    const sizeKB = (blob.size / 1024).toFixed(2);
    console.log(`✅ Downloaded: ${filename} (${sizeKB} KB)`);
    
    // Visual feedback
    const toast = document.createElement('div');
    toast.style.cssText = 'position:fixed;top:20px;right:20px;background:#10b981;color:white;padding:16px 24px;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.15);z-index:999999;font-family:system-ui;font-size:14px;';
    toast.textContent = `✓ Saved ${filename} (${sizeKB} KB)`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
  }
  
  captureComplete().catch(e => {
    console.error('❌ Capture failed:', e);
    alert('Failed to capture page: ' + e.message);
  });
})();
