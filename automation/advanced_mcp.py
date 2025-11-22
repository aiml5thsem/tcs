#!/usr/bin/env python3
"""
Best-of-all-worlds MCP server for computer automation.
Cross-platform and supports advanced and basic agent tooling for Windows, Mac, Linux.

* Platform clever: runs/returns only on platforms/tools where safe/supported.
* Optional dependencies: try/except for window/ocr/clipboard features; always reply clearly.
"""
import base64
import os
import sys
import time
import platform
import pyautogui
from io import BytesIO
from fastmcp import FastMCP, Image

PY_VER = sys.version_info
OS = platform.system()
IS_WIN = OS == "Windows"
IS_MAC = OS == "Darwin"
IS_LINUX = OS == "Linux"

# --- Optional imports for advanced tools ---
try:
    import pygetwindow as gw
except ImportError:
    gw = None
try:
    import pyperclip
except ImportError:
    pyperclip = None
try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract, Image = None, None
try:
    import pyscreeze
except ImportError:
    pyscreeze = None

# macOS OCR via ocrmac (uncommon, but some use it)
try:
    import ocrmac
except ImportError:
    ocrmac = None

mcp = FastMCP("Super Computer Control MCP")

# ============================================================================
# Mouse & Keyboard
# ============================================================================

@mcp.tool()
def move_to(x: int, y: int, duration: float = 0.2) -> str:
    pyautogui.moveTo(x, y, duration=duration)
    return f"Moved mouse to ({x}, {y})"

@mcp.tool()
def click(x: int | None = None, y: int | None = None, clicks: int = 1, button: str = "left") -> str:
    if x is not None and y is not None:
        pyautogui.moveTo(x, y, duration=0.1)
    pyautogui.click(clicks=clicks, button=button)
    return f"Clicked ({clicks}x) button={button}"

@mcp.tool()
def right_click(x: int | None = None, y: int | None = None) -> str:
    if x is not None and y is not None:
        pyautogui.moveTo(x, y, duration=0.1)
    pyautogui.rightClick()
    return "Right-clicked"

@mcp.tool()
def double_click(x: int | None = None, y: int | None = None) -> str:
    if x is not None and y is not None:
        pyautogui.moveTo(x, y, duration=0.1)
    pyautogui.doubleClick()
    return "Double-clicked"

@mcp.tool()
def drag_to(to_x: int, to_y: int, from_x: int | None = None, from_y: int | None = None, duration: float = 0.5) -> str:
    if from_x is not None and from_y is not None:
        pyautogui.moveTo(from_x, from_y)
    pyautogui.dragTo(to_x, to_y, duration=duration, button="left")
    return f"Dragged to ({to_x},{to_y})"

@mcp.tool()
def scroll(clicks: int, x: int | None = None, y: int | None = None) -> str:
    if x is not None and y is not None:
        pyautogui.moveTo(x, y)
    pyautogui.scroll(clicks)
    return f"Scrolled {'up' if clicks>0 else 'down'} {abs(clicks)} clicks"

@mcp.tool()
def type_text(text: str, interval: float = 0.05) -> str:
    pyautogui.write(text, interval=interval)
    return f"Typed: {text}"

@mcp.tool()
def press_key(key: str) -> str:
    pyautogui.press(key)
    return f"Pressed key: {key}"

@mcp.tool()
def hotkey(keys: list[str]) -> str:
    pyautogui.hotkey(*keys)
    return f"Pressed hotkey: {'+'.join(keys)}"

@mcp.tool()
def get_mouse_position() -> dict:
    pos = pyautogui.position()
    return {"x": pos.x, "y": pos.y}

# ============================================================================
# Screenshot & Image Recognition
# ============================================================================

@mcp.tool()
def take_screenshot() -> Image:
    """
    Take a screenshot of the current screen and return as JPEG image
    
    Returns:
        Image object in JPEG format
    """
    screenshot = pyautogui.screenshot()
    if screenshot.mode == "RGBA":
        screenshot = screenshot.convert("RGB")
    buffer = BytesIO()
    screenshot.save(buffer, format="JPEG")
    return Image(data=buffer.getvalue(), format="jpeg")

@mcp.tool()
def screenshot_region(x: int, y: int, width: int, height: int) -> str:
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    if screenshot.mode == "RGBA":
        screenshot = screenshot.convert("RGB")
    buf = BytesIO()
    screenshot.save(buf, format="JPEG")
    return Image(data=buffer.getvalue(), format="jpeg")

@mcp.tool()
def get_screen_size() -> dict:
    size = pyautogui.size()
    return {"width": size.width, "height": size.height}

@mcp.tool()
def get_pixel_color(x: int, y: int) -> dict:
    screenshot = pyautogui.screenshot()
    color = screenshot.getpixel((x, y))
    return {"r": color[0], "g": color[1], "b": color[2]}

@mcp.tool()
def locate_on_screen(image_path: str, confidence: float = 0.8) -> dict | None:
    if pyscreeze is None:
        return {"error": "pyscreeze not installed"}
    try:
        result = pyautogui.locateOnScreen(image_path, confidence=confidence)  # pillow>=7.0 for confidence
        if result:
            return {
                "x": result.left,
                "y": result.top,
                "width": result.width,
                "height": result.height
            }
        return None
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def locate_all_on_screen(image_path: str, confidence: float = 0.8) -> list:
    if pyscreeze is None:
        return [{"error": "pyscreeze not installed"}]
    try:
        boxes = pyautogui.locateAllOnScreen(image_path, confidence=confidence)
        return [
            {"x": b.left, "y": b.top, "width": b.width, "height": b.height}
            for b in boxes
        ]
    except Exception as e:
        return [{"error": str(e)}]

# ============================================================================
# Application & Window/Process Control
# ============================================================================

@mcp.tool()
def open_app(app_name: str) -> str:
    sys_name = platform.system()
    if sys_name == "Darwin":
        os.system(f"open -a '{app_name}'")
    elif sys_name == "Windows":
        os.system(f'start "" "{app_name}"')
    else:
        os.system(f"{app_name} &")
    time.sleep(2)
    return f"Opened {app_name}"

@mcp.tool()
def close_app(app_name: str) -> str:
    sys_name = platform.system()
    if sys_name == "Windows":
        os.system(f'taskkill /im "{app_name}.exe" /f')
    elif sys_name == "Darwin":
        os.system(f"osascript -e 'quit app \"{app_name}\"'")
    else:
        os.system(f"pkill '{app_name}'")
    return f"Closed {app_name}"

@mcp.tool()
def list_windows() -> list:
    if gw is None:
        return [{"error": "pygetwindow not installed"}]
    try:
        wins = []
        for w in gw.getAllTitles():
            if w:
                wins.append(w)
        return wins
    except Exception as e:
        return [{"error": str(e)}]

@mcp.tool()
def focus_window(title: str) -> str:
    if gw is None:
        return "pygetwindow not installed"
    try:
        wins = gw.getWindowsWithTitle(title)
        if wins:
            wins[0].activate()
            return f"Focused window '{title}'"
        else:
            return f"Window '{title}' not found"
    except Exception as e:
        return str(e)

@mcp.tool()
def move_window(title: str, x: int, y: int) -> str:
    if gw is None:
        return "pygetwindow not installed"
    try:
        wins = gw.getWindowsWithTitle(title)
        if wins:
            wins[0].moveTo(x, y)
            return f"Moved window '{title}' to ({x},{y})"
        else:
            return f"Window '{title}' not found"
    except Exception as e:
        return str(e)

@mcp.tool()
def resize_window(title: str, width: int, height: int) -> str:
    if gw is None:
        return "pygetwindow not installed"
    try:
        wins = gw.getWindowsWithTitle(title)
        if wins:
            wins[0].resizeTo(width, height)
            return f"Resized window '{title}'"
        else:
            return f"Window '{title}' not found"
    except Exception as e:
        return str(e)

# ============================================================================
# Clipboard
# ============================================================================

@mcp.tool()
def clipboard_get() -> dict:
    """
    Get the current clipboard contents (text only).
    Returns:
        {"text": clipboard text}
    """
    if pyperclip is None:
        return {"error": "pyperclip not installed"}
    try:
        text = pyperclip.paste()
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def clipboard_set(text: str) -> str:
    """
    Set clipboard to the given text.
    Args:
        text: Text to be copied to clipboard
    """
    if pyperclip is None:
        return "pyperclip not installed"
    try:
        pyperclip.copy(text)
        return "Copied text to clipboard"
    except Exception as e:
        return f"Error: {e}"

# ============================================================================
# Dialogs
# ============================================================================

@mcp.tool()
def alert(text: str, title: str = "Alert", button: str = "OK") -> str:
    """
    Show an alert dialog box.
    Args:
        text: Message
        title: Dialog title
        button: Button text
    Returns:
        Button pressed
    """
    btn = pyautogui.alert(text=text, title=title, button=button)
    return btn

@mcp.tool()
def prompt(text: str, title: str = "Prompt", default: str = "") -> str:
    """
    Show a prompt dialog (input box).
    Args:
        text: Message
        title: Dialog title
        default: Default input value
    Returns:
        User input string
    """
    result = pyautogui.prompt(text=text, title=title, default=default)
    return result

@mcp.tool()
def confirm(text: str, title: str = "Confirm", buttons: list[str] = None) -> str:
    """
    Show a confirmation dialog box.
    Args:
        text: Message
        title: Dialog title
        buttons: List of button captions
    Returns:
        Button pressed
    """
    if buttons is None:
        buttons = ["OK", "Cancel"]
    result = pyautogui.confirm(text=text, title=title, buttons=buttons)
    return result

# ============================================================================
# OCR (Optical Character Recognition, Cross-Platform)
# ============================================================================

@mcp.tool()
def ocr_image(image_b64: str) -> dict:
    """
    Extract text from an image using OCR.
    Args:
        image_b64: Base64-encoded image string (JPG/PNG)
    Returns:
        {"text": recognized text or error}
    """
    try:
        img_bytes = base64.b64decode(image_b64)
        if IS_MAC and ocrmac:
            # Prefer ocrmac if installed
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                tmp.write(img_bytes)
                tmp.flush()
                text = ocrmac.ocr(tmp.name)
                return {"text": text}
        elif pytesseract and Image:
            img = Image.open(BytesIO(img_bytes))
            text = pytesseract.image_to_string(img)
            return {"text": text}
        else:
            return {"error": "No OCR engine installed (needs ocrmac or pytesseract+Pillow)"}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def ocr_screenshot_region(x: int, y: int, width: int, height: int) -> dict:
    """
    Take a screenshot of a region and OCR it.
    Args:
        x, y: top-left
        width, height: region size
    Returns:
        {"text": OCR result}
    """
    try:
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        buf = BytesIO()
        screenshot.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        if IS_MAC and ocrmac:
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                tmp.write(img_bytes)
                tmp.flush()
                text = ocrmac.ocr(tmp.name)
                return {"text": text}
        elif pytesseract and Image:
            img = Image.open(BytesIO(img_bytes))
            text = pytesseract.image_to_string(img)
            return {"text": text}
        else:
            return {"error": "No OCR engine installed (needs ocrmac or pytesseract+Pillow)"}
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# Utilities, Help and Graceful Error Pattern
# ============================================================================

@mcp.tool()
def about_tools() -> dict:
    """Return a summary of available tools/actions in this MCP server."""
    return {
        "tools": list(mcp._tools.keys()),
        "platform": OS,
        "python_version": f"{PY_VER.major}.{PY_VER.minor}.{PY_VER.micro}",
        "optional_deps": {
            "pygetwindow": gw is not None,
            "pyperclip": pyperclip is not None,
            "pytesseract": pytesseract is not None,
            "ocrmac": ocrmac is not None
        }
    }

# ============================================================================
# Start server
# ============================================================================

if __name__ == "__main__":
    mcp.run()
