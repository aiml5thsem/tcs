# Advanced Authenticated Web Scraper

A production-ready web scraper designed for extracting complete documentation from authenticated organizational websites. Perfect for creating LLM-ready documentation from internal wikis, knowledge bases, and private sites.

## 🎯 Key Features

### **Three Crawling Modes**
- **🤖 Auto Mode**: Fully automatic discovery and crawling
- **👤 Manual Mode**: You navigate, script extracts on demand
- **🔀 Hybrid Mode**: Auto-discovery with manual customization

### **Advanced Capabilities**
- ✅ **Sitemap.xml parsing** for efficient discovery
- ✅ **Breadth-first crawling** for proper page ordering
- ✅ **Multi-tab support** in manual mode
- ✅ **Existing Chrome/Chromium support** (no playwright browser needed!)
- ✅ **Smart URL deduplication** and normalization
- ✅ **Clean Markdown output** optimized for LLM ingestion
- ✅ **Session persistence** - maintains authentication state

---

## 📦 Installation

### Prerequisites
```bash
# Python 3.8+
python --version

# Install dependencies
pip install playwright html2text

# Install Playwright browsers (skip if using existing Chrome)
playwright install chromium
```

### Optional: Using Existing Chrome
If you want to use your system's Chrome/Chromium (recommended for org laptops):
```bash
# No need to install Playwright browsers
# Just use --use-chrome auto flag when running
```

---

## 🚀 Quick Start

### 1. Auto Mode (Fully Automatic)
Discovers all links via sitemap and page crawling, then extracts everything automatically.

```bash
python web_scraper.py \
  --url https://docs.mycompany.com \
  --mode auto \
  --max-pages 200 \
  --max-depth 8
```

**Best for:** Well-structured documentation sites with clear navigation

### 2. Manual Mode (You Control)
You manually navigate to pages, press Enter to extract each page.

```bash
python web_scraper.py \
  --url https://internal.mycompany.com \
  --mode manual
```

**Workflow:**
1. Script opens browser at base URL
2. You authenticate and navigate to pages you want
3. Press **ENTER** to extract current page(s) from all tabs
4. Continue navigating and pressing Enter for more pages
5. Type **'done'** when finished

**Best for:** Sites with complex navigation or when you want precise control

### 3. Hybrid Mode (Best of Both)
Auto-discovers links, shows you the list, lets you add/remove before crawling.

```bash
python web_scraper.py \
  --url https://wiki.mycompany.com \
  --mode hybrid \
  --max-pages 300
```

**Workflow:**
1. Script auto-discovers all links via sitemap and crawling
2. Shows you the discovered URLs
3. You can:
   - Press Enter to crawl all
   - Add additional URLs manually
   - Navigate to more pages in browser
4. Script crawls everything automatically

**Best for:** When you want automation but need to verify/customize the scope

---

## 🌐 Using Existing Chrome/Chromium

### Why Use Existing Chrome?
- **No browser download** needed (saves ~300MB)
- **Uses your Chrome profile** (extensions, settings)
- **Required on locked-down org laptops** where playwright install fails

### Finding Chrome Path

#### **macOS**
```bash
# Google Chrome
/Applications/Google Chrome.app/Contents/MacOS/Google Chrome

# Chromium
/Applications/Chromium.app/Contents/MacOS/Chromium

# Find automatically
ls -la "/Applications/Google Chrome.app/Contents/MacOS/"
```

#### **Windows**
```powershell
# Google Chrome (Common paths)
C:\Program Files\Google\Chrome\Application\chrome.exe
C:\Program Files (x86)\Google\Chrome\Application\chrome.exe

# Find automatically
where chrome
```

#### **Linux**
```bash
# Common paths
/usr/bin/google-chrome
/usr/bin/google-chrome-stable
/usr/bin/chromium-browser

# Find automatically
which google-chrome
```

### Usage Examples

```bash
# Auto-detect Chrome
python web_scraper.py --url https://myorg.com --mode auto --use-chrome auto

# Specify exact path (macOS)
python web_scraper.py --url https://myorg.com --mode auto \
  --use-chrome "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Specify exact path (Windows)
python web_scraper.py --url https://myorg.com --mode auto \
  --use-chrome "C:\Program Files\Google\Chrome\Application\chrome.exe"

# Specify exact path (Linux)
python web_scraper.py --url https://myorg.com --mode auto \
  --use-chrome /usr/bin/google-chrome
```

---

## 🎛️ Command-Line Options

```bash
python web_scraper.py [OPTIONS]

Required:
  --url URL              Base URL to scrape

Optional:
  --mode {auto|manual|hybrid}
                         Crawling mode (default: auto)
  
  --max-pages N         Maximum pages to scrape (default: 100)
  
  --max-depth N         Maximum crawl depth (default: 6)
  
  --output FILE         Output filename (auto-generated if not provided)
  
  --use-chrome PATH     Path to Chrome/Chromium, or 'auto' to auto-detect
  
  --no-sitemap          Don't try to parse sitemap.xml
  
  --delay SECONDS       Delay between requests (default: 0.5)
```

---

## 📖 Usage Examples

### Example 1: Complete Org Documentation
```bash
# Auto mode with existing Chrome, save to specific file
python web_scraper.py \
  --url https://docs.internal.company.com \
  --mode auto \
  --max-pages 500 \
  --max-depth 10 \
  --use-chrome auto \
  --output company_docs.md \
  --delay 1.0
```

### Example 2: Manual Selective Extraction
```bash
# Extract only specific pages you navigate to
python web_scraper.py \
  --url https://confluence.company.com \
  --mode manual \
  --use-chrome auto
```

**Then in browser:**
1. Authenticate with SSO
2. Navigate to "Engineering Handbook"
3. Press Enter → extracts current page
4. Open "API Guidelines" in new tab
5. Navigate to "Security Policies" in another tab
6. Press Enter → extracts all 3 tabs
7. Type 'done' when finished

### Example 3: Hybrid with Verification
```bash
# Let script discover, then review before crawling
python web_scraper.py \
  --url https://wiki.company.com \
  --mode hybrid \
  --max-pages 200 \
  --use-chrome "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
```

**Script will:**
1. Find sitemap.xml automatically
2. Show you all discovered URLs
3. Let you add missing URLs or remove unwanted ones
4. Crawl everything with proper ordering

---

## 🗂️ Output Format

The script generates a single Markdown file with:

```markdown
# Documentation: docs.mycompany.com
_Generated: 2024-12-04 15:30:45_
_Mode: AUTO_

**Base URL:** https://docs.mycompany.com
**Total Pages:** 156

---

## Table of Contents

• [Home Page](#home-page)
  • [Getting Started](#getting-started)
    • [Installation](#installation)
    • [Configuration](#configuration)
  • [User Guide](#user-guide)
...

---

## Home Page
_URL: [https://docs.mycompany.com](https://docs.mycompany.com)_
_Depth: 0 | Order: 1_

[Page content in clean Markdown...]

---

## Getting Started
_URL: [https://docs.mycompany.com/getting-started](...)_
_Parent: https://docs.mycompany.com_
_Depth: 1 | Order: 2_

[Page content...]

---

... [All pages follow]

---

## Metadata
- **Total Pages:** 156
- **Mode:** AUTO
- **Max Depth:** 6
- **Generated:** 2024-12-04T15:30:45.123456
```

---

## 🎯 Best Practices

### For Best Results

1. **Start with Hybrid Mode**
   - Verify what will be crawled before committing
   - Add any missing pages manually
   - Good for first-time scraping of a site

2. **Use Sitemap When Available**
   - Automatically discovered at `/sitemap.xml`
   - Provides complete, structured list of pages
   - Faster than recursive crawling

3. **Adjust Depth and Limits**
   - Start with `--max-depth 6 --max-pages 100`
   - Increase if needed: `--max-depth 10 --max-pages 500`
   - Monitor output to avoid scraping too much

4. **Add Delays for Rate Limiting**
   - Use `--delay 1.0` for sensitive sites
   - Prevents overwhelming the server
   - Reduces chances of being blocked

5. **Use Existing Chrome on Org Laptops**
   - `--use-chrome auto` is easiest
   - Avoids permission issues with browser downloads
   - Uses your existing profile and extensions

### For LLM Integration

After scraping, use with MCP Large File Server:

```bash
# 1. Generate documentation
python web_scraper.py --url https://docs.myorg.com --mode auto --output org_docs.md

# 2. Use with Claude via MCP Large File Server
# Add to your MCP config, then reference in prompts:
# "Using the org_docs.md file, explain our authentication flow..."
```

---

## 🔧 Troubleshooting

### "Playwright browser not found"
```bash
# Solution 1: Install playwright browsers
playwright install chromium

# Solution 2: Use existing Chrome (recommended)
python web_scraper.py --url URL --use-chrome auto
```

### "Chrome path not found with --use-chrome auto"
```bash
# Find Chrome manually, then specify exact path
# macOS:
python web_scraper.py --url URL --use-chrome "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Windows:
python web_scraper.py --url URL --use-chrome "C:\Program Files\Google\Chrome\Application\chrome.exe"
```

### "Too many pages, takes too long"
```bash
# Reduce limits
python web_scraper.py --url URL --max-pages 50 --max-depth 4

# Or use manual mode for precise control
python web_scraper.py --url URL --mode manual
```

### "Pages not in correct order"
- Script uses breadth-first crawling (best ordering)
- Pages sorted by depth, then discovery order
- Check sitemap.xml exists for optimal results
- Use `--mode hybrid` to verify order before crawling

### "Getting rate limited / blocked"
```bash
# Add delay between requests
python web_scraper.py --url URL --delay 2.0

# Reduce concurrent requests (in manual mode)
# Navigate and extract one page at a time
```

---

## 🔐 Authentication Support

The scraper supports all authentication methods:

- **✅ OAuth / SSO** (Google, Microsoft, Okta, etc.)
- **✅ Username/Password forms**
- **✅ Two-Factor Authentication (2FA)**
- **✅ SAML**
- **✅ Browser extensions** (when using existing Chrome)
- **✅ Session cookies** (maintained throughout crawl)

**How it works:**
1. Browser opens at base URL
2. You authenticate manually in the browser
3. Press Enter when ready
4. Script crawls while maintaining your session

---

## 📝 Advanced Configuration

### Custom Output Naming
```bash
# Auto-generated (default)
# Output: docs_mycompany_com_auto_20241204_153045.md

# Custom name
python web_scraper.py --url URL --output "Q4_Documentation.md"
```

### Multi-Site Documentation
```bash
# Scrape multiple sites separately
python web_scraper.py --url https://docs.myorg.com --output docs.md
python web_scraper.py --url https://wiki.myorg.com --output wiki.md
python web_scraper.py --url https://api.myorg.com --output api.md

# Then combine for LLM:
cat docs.md wiki.md api.md > complete_org_docs.md
```

### Selective Scraping
```bash
# Use manual mode with multiple tabs
python web_scraper.py --url https://huge-site.com --mode manual

# Then:
# 1. Navigate to sections you want
# 2. Open multiple tabs
# 3. Press Enter to extract all tabs at once
# 4. Type 'done' when finished
```

---

## 🆘 Support & Issues

### Common Issues Fixed
- ✅ URLs not in correct order → **Fixed with breadth-first crawling**
- ✅ Starting from wrong page → **Fixed with proper start URL handling**
- ✅ Duplicate pages → **Fixed with smart URL normalization**
- ✅ Missing pages → **Fixed with sitemap.xml support**
- ✅ Can't install browsers → **Fixed with existing Chrome support**

### Report Issues
If you encounter problems:
1. Check the troubleshooting section above
2. Try different modes (auto/manual/hybrid)
3. Use `--use-chrome auto` if browser issues
4. Reduce `--max-pages` and `--max-depth` if too slow

---

## 📚 Use Cases

### ✅ Perfect For:
- Internal company documentation
- Private wikis and knowledge bases
- Authenticated API documentation
- Confluence, SharePoint, GitBook sites
- Any site requiring login

### ❌ Not Suitable For:
- Sites explicitly blocking scraping (check robots.txt)
- Sites with aggressive anti-bot protection
- Dynamic single-page apps with heavy JavaScript
- Sites requiring specific user interactions (clicks, forms)

---

## ⚡ Performance Tips

1. **Use Sitemap** - Much faster than recursive crawling
2. **Increase Depth Carefully** - Depth 10+ can scrape thousands of pages
3. **Set Realistic Limits** - Start with 100 pages, increase if needed
4. **Use Manual Mode** - For large sites, extract only what you need
5. **Monitor Progress** - Watch console output for issues

---

## 🎉 Quick Reference

```bash
# Most Common Usage Patterns

# 1. Quick auto-scrape with existing Chrome
python web_scraper.py --url https://docs.myorg.com --mode auto --use-chrome auto

# 2. Manual selective extraction  
python web_scraper.py --url https://wiki.myorg.com --mode manual --use-chrome auto

# 3. Hybrid with verification
python web_scraper.py --url https://internal.myorg.com --mode hybrid --max-pages 200

# 4. Large-scale documentation
python web_scraper.py --url https://docs.myorg.com --mode auto --max-pages 500 --max-depth 10 --delay 1.0

# 5. Fast small scrape
python web_scraper.py --url https://api.myorg.com --mode auto --max-pages 50 --max-depth 4
```

---

**Happy Scraping! 🚀**

For LLM integration, feed the generated Markdown to Claude or other LLMs via MCP Large File Server for instant access to your org's complete documentation.