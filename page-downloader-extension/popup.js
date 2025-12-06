// Get DOM elements
const counterDisplay = document.getElementById('counterDisplay');
const downloadBtn = document.getElementById('downloadBtn');
const resetBtn = document.getElementById('resetBtn');
const setBtn = document.getElementById('setBtn');
const counterInput = document.getElementById('counterInput');
const formatSelect = document.getElementById('formatSelect');
const status = document.getElementById('status');

// Load and display current counter
async function loadCounter() {
  try {
    const result = await chrome.storage.local.get(['counter']);
    const counter = result.counter || 1;
    counterDisplay.textContent = counter;
    return counter;
  } catch (error) {
    console.error('Error loading counter:', error);
    return 1;
  }
}

// Save counter to storage
async function saveCounter(value) {
  try {
    await chrome.storage.local.set({ counter: value });
    counterDisplay.textContent = value;
  } catch (error) {
    console.error('Error saving counter:', error);
  }
}

// Show status message
function showStatus(message, isError = false) {
  status.textContent = message;
  status.className = `status ${isError ? 'error' : 'success'}`;
  setTimeout(() => {
    status.className = 'status';
  }, 3000);
}

// Function to extract page content
function extractPageContent(format) {
  function htmlToMarkdown(element) {
    let markdown = '';
    
    function processNode(node, indent = '') {
      if (node.nodeType === Node.TEXT_NODE) {
        const text = node.textContent.trim();
        if (text) {
          markdown += text + ' ';
        }
        return;
      }
      
      if (node.nodeType !== Node.ELEMENT_NODE) return;
      
      const tag = node.tagName.toLowerCase();
      
      switch(tag) {
        case 'script':
        case 'style':
        case 'noscript':
          return;
        case 'h1':
          markdown += '\n\n# ' + node.textContent.trim() + '\n\n';
          break;
        case 'h2':
          markdown += '\n\n## ' + node.textContent.trim() + '\n\n';
          break;
        case 'h3':
          markdown += '\n\n### ' + node.textContent.trim() + '\n\n';
          break;
        case 'h4':
          markdown += '\n\n#### ' + node.textContent.trim() + '\n\n';
          break;
        case 'h5':
          markdown += '\n\n##### ' + node.textContent.trim() + '\n\n';
          break;
        case 'h6':
          markdown += '\n\n###### ' + node.textContent.trim() + '\n\n';
          break;
        case 'p':
          markdown += '\n\n';
          node.childNodes.forEach(child => processNode(child, indent));
          markdown += '\n\n';
          break;
        case 'br':
          markdown += '  \n';
          break;
        case 'strong':
        case 'b':
          markdown += '**' + node.textContent.trim() + '**';
          break;
        case 'em':
        case 'i':
          markdown += '*' + node.textContent.trim() + '*';
          break;
        case 'code':
          if (node.parentElement.tagName.toLowerCase() !== 'pre') {
            markdown += '`' + node.textContent.trim() + '`';
          } else {
            node.childNodes.forEach(child => processNode(child, indent));
          }
          break;
        case 'pre':
          markdown += '\n\n```\n' + node.textContent.trim() + '\n```\n\n';
          break;
        case 'a':
          const href = node.getAttribute('href') || '';
          markdown += '[' + node.textContent.trim() + '](' + href + ')';
          break;
        case 'img':
          const src = node.getAttribute('src') || '';
          const alt = node.getAttribute('alt') || '';
          markdown += '![' + alt + '](' + src + ')';
          break;
        case 'ul':
        case 'ol':
          markdown += '\n';
          Array.from(node.children).forEach((li, index) => {
            if (li.tagName.toLowerCase() === 'li') {
              const bullet = tag === 'ul' ? '-' : `${index + 1}.`;
              markdown += indent + bullet + ' ';
              li.childNodes.forEach(child => processNode(child, indent + '  '));
              markdown += '\n';
            }
          });
          markdown += '\n';
          break;
        case 'blockquote':
          markdown += '\n> ';
          node.childNodes.forEach(child => processNode(child, indent));
          markdown += '\n';
          break;
        case 'hr':
          markdown += '\n\n---\n\n';
          break;
        case 'table':
          markdown += '\n\n';
          node.childNodes.forEach(child => processNode(child, indent));
          markdown += '\n\n';
          break;
        case 'tr':
          markdown += '| ';
          node.childNodes.forEach(child => {
            if (child.tagName && (child.tagName.toLowerCase() === 'td' || child.tagName.toLowerCase() === 'th')) {
              markdown += child.textContent.trim() + ' | ';
            }
          });
          markdown += '\n';
          break;
        default:
          node.childNodes.forEach(child => processNode(child, indent));
      }
    }
    
    processNode(element);
    return markdown.replace(/\n{3,}/g, '\n\n').trim();
  }

  try {
    if (format === 'md') {
      const title = document.title;
      const url = window.location.href;
      const date = new Date().toLocaleString();
      
      let markdown = `# ${title}\n\n`;
      markdown += `**URL:** ${url}\n\n`;
      markdown += `**Downloaded:** ${date}\n\n`;
      markdown += '---\n\n';
      markdown += htmlToMarkdown(document.body);
      
      return markdown;
    } else {
      return document.documentElement.outerHTML;
    }
  } catch (error) {
    return 'Error extracting content: ' + error.message;
  }
}

// Download button handler
downloadBtn.addEventListener('click', async () => {
  try {
    console.log('Download button clicked');
    downloadBtn.disabled = true;
    downloadBtn.textContent = '⏳ Downloading...';

    // Get current tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    console.log('Current tab:', tab);
    
    if (!tab || !tab.id) {
      throw new Error('No active tab found');
    }

    // Check if we can access this tab
    if (tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://')) {
      throw new Error('Cannot access Chrome internal pages');
    }

    // Get current counter and format
    const counter = await loadCounter();
    const format = formatSelect.value;
    console.log('Counter:', counter, 'Format:', format);

    // Execute script to extract content
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractPageContent,
      args: [format]
    });

    console.log('Script execution results:', results);

    if (!results || !results[0] || !results[0].result) {
      throw new Error('Failed to extract page content');
    }

    const content = results[0].result;
    console.log('Content length:', content.length);

    // Create data URL instead of blob URL
    const dataUrl = 'data:text/plain;charset=utf-8,' + encodeURIComponent(content);
    const filename = `${counter}.${format}`;
    
    console.log('Downloading as:', filename);

    // Download the file
    const downloadId = await chrome.downloads.download({
      url: dataUrl,
      filename: filename,
      saveAs: false
    });

    console.log('Download started, ID:', downloadId);

    // Increment counter
    await saveCounter(counter + 1);
    
    showStatus(`✓ Downloaded as ${filename}`);
    
  } catch (error) {
    console.error('Download error:', error);
    showStatus(`Error: ${error.message}`, true);
  } finally {
    downloadBtn.disabled = false;
    downloadBtn.textContent = '📥 Download Page';
  }
});

// Reset button handler
resetBtn.addEventListener('click', async () => {
  console.log('Reset button clicked');
  try {
    await saveCounter(1);
    showStatus('✓ Counter reset to 1');
  } catch (error) {
    console.error('Reset error:', error);
    showStatus('Error resetting counter', true);
  }
});

// Set counter button handler
setBtn.addEventListener('click', async () => {
  console.log('Set button clicked');
  try {
    const value = parseInt(counterInput.value);
    
    if (isNaN(value) || value < 1) {
      showStatus('Please enter a number ≥ 1', true);
      return;
    }
    
    await saveCounter(value);
    counterInput.value = '';
    showStatus(`✓ Counter set to ${value}`);
  } catch (error) {
    console.error('Set counter error:', error);
    showStatus('Error setting counter', true);
  }
});

// Allow Enter key in input
counterInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') {
    setBtn.click();
  }
});

// Initialize
console.log('Popup script loaded');
loadCounter().then(() => {
  console.log('Counter loaded successfully');
}).catch(error => {
  console.error('Error loading counter:', error);
});