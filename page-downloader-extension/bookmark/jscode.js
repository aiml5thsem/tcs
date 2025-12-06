javascript:(function(){
  // Capture complete rendered DOM including shadow DOM & nested iframes
  function getAllContent(element = document.documentElement) {
    let html = element.outerHTML;
    
    // Capture shadow DOM if present
    if (element.shadowRoot) {
      html += `<!-- SHADOW DOM -->\n${element.shadowRoot.innerHTML}`;
    }
    
    // Process iframes
    document.querySelectorAll('iframe').forEach((iframe, idx) => {
      try {
        const iframeContent = iframe.contentDocument?.documentElement.outerHTML || 
                             iframe.contentWindow?.document.documentElement.outerHTML;
        if (iframeContent) html += `\n<!-- IFRAME ${idx} -->\n${iframeContent}`;
      } catch(e) { /* Cross-origin blocked */ }
    });
    
    return html;
  }
  
  const html = getAllContent();
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const filename = `${timestamp}.html`;
  
  // Download file
  const blob = new Blob([html], {type: 'text/html'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
  
  console.log(`✓ Downloaded: ${filename} (${(blob.size/1024).toFixed(2)} KB)`);
})();
