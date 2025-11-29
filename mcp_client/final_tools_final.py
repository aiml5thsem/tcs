#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Server for Computer Automation using FastMCP.
Exposes all automation functions as MCP tools for LLM integration.
Includes window management, clipboard, dialog boxes, and AI vision tools.
"""

import base64
import os
import time
import platform
import pyautogui
import json
from io import BytesIO
from typing import Dict, Any, List, Optional, Union

from PIL import Image, ImageGrab

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pygetwindow as gw
    PYGETWINDOW_AVAILABLE = True
except ImportError:
    PYGETWINDOW_AVAILABLE = False

try:
    import pyperclip
    PYPERCLIP_AVAILABLE = True
except ImportError:
    PYPERCLIP_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from fastmcp import FastMCP
# Use new import path to avoid deprecation warning
try:
    from fastmcp.utilities.types import Image as MCPImage
except ImportError:
    # Fallback for older versions
    try:
        from fastmcp import Image as MCPImage
    except ImportError:
        MCPImage = None

# Pillow compatibility
Image.ANTIALIAS = Image.LANCZOS

# Create FastMCP server
mcp = FastMCP("Computer Automation Tools")

# Store tool names for logging
TOOL_NAMES = []


# ============================================================================
# SCREENSHOT & IMAGE CAPTURE TOOLS
# ============================================================================

@mcp.tool()
def take_screenshot(format: str = "base64") -> Union[str, Dict[str, Any]]:
    """
    Take full screenshot in specified format.
    
    Args:
        format: Return format - "array_info", "base64" (default), "file", or "mcp"
                - "array_info": Returns shape and size info (for array format)
                - "base64": Base64 encoded JPEG string (good for APIs)
                - "file": Filepath string (saved as screenshot.png)
                - "mcp": FastMCP Image type, so the LLM receives a proper MCP image (structured for MCP protocol)
    
    Returns:
        Base64 string, file path, info dict, or MCPImage dict for LLM
    
    Examples:
        take_screenshot()              # Returns base64 by default
        take_screenshot("file")        # Saves and returns filepath
        take_screenshot("mcp")         # Returns MCPImage format for LLM
    """
    screenshot = pyautogui.screenshot()
    
    if format.lower() == "array_info":
        arr_info = np.array(screenshot)
        return f"Array format available: shape={arr_info.shape}, dtype=uint8"
    
    elif format.lower() == "base64":
        if screenshot.mode == "RGBA":
            screenshot = screenshot.convert("RGB")
        buffer = BytesIO()
        screenshot.save(buffer, format="JPEG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode()
    
    elif format.lower() == "file":
        if screenshot.mode == "RGBA":
            screenshot = screenshot.convert("RGB")
        filepath = "screenshot.png"
        screenshot.save(filepath)
        return f"Screenshot saved to {filepath}"
    
    elif format.lower() == "mcp":
        # Return MCPImage format for LLM display via MCP
        if screenshot.mode == "RGBA":
            screenshot = screenshot.convert("RGB")
        buffer = BytesIO()
        screenshot.save(buffer, format="JPEG")
        buffer.seek(0)
        # screenshot_base64 = base64.b64encode(buffer.getvalue()).decode()
        return MCPImage(data=buffer.getvalue(), format="jpeg")
    
    else:
        return f"Error: Unknown format '{format}'. Use 'base64', 'array_info', 'file', or 'mcp'"


TOOL_NAMES.append("take_screenshot")


@mcp.tool()
def screenshot_region(x: int, y: int, width: int, height: int, format: str = "base64") -> Union[str, Dict[str, Any]]:
    """
    Capture specific region of screen in specified format.
    
    Args:
        x: X coordinate of top-left corner
        y: Y coordinate of top-left corner
        width: Width of region in pixels
        height: Height of region in pixels
        format: Return format - "array_info", "base64" (default), "file", or "mcp"
    
    Returns:
        Base64 string, file path, info dict, or MCPImage dict for LLM
    
    Examples:
        screenshot_region(400, 300, 800, 400)         # Returns base64
        screenshot_region(400, 300, 800, 400, "file") # Saves to file
        screenshot_region(400, 300, 800, 400, "mcp")  # Returns MCPImage format
    """
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    
    if format.lower() == "array_info":
        arr_info = np.array(screenshot)
        return f"Region array available: shape={arr_info.shape}"
    
    elif format.lower() == "base64":
        if screenshot.mode == "RGBA":
            screenshot = screenshot.convert("RGB")
        buffer = BytesIO()
        screenshot.save(buffer, format="JPEG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode()
    
    elif format.lower() == "file":
        if screenshot.mode == "RGBA":
            screenshot = screenshot.convert("RGB")
        filepath = f"screenshot_region_{x}_{y}_{width}_{height}.png"
        screenshot.save(filepath)
        return f"Region screenshot saved to {filepath}"
    
    elif format.lower() == "mcp":
        # Return MCPImage format for LLM display via MCP
        if screenshot.mode == "RGBA":
            screenshot = screenshot.convert("RGB")
        buffer = BytesIO()
        screenshot.save(buffer, format="JPEG")
        buffer.seek(0)
        return MCPImage(data=buffer.getvalue(), format="jpeg")
    
    else:
        return f"Error: Unknown format '{format}'. Use 'base64', 'array_info', 'file', or 'mcp'"


TOOL_NAMES.append("screenshot_region")


@mcp.tool()
def get_screen_size() -> Dict[str, int]:
    """Get screen resolution."""
    size = pyautogui.size()
    return {"width": size.width, "height": size.height}


TOOL_NAMES.append("get_screen_size")


@mcp.tool()
def get_pixel_color(x: int, y: int) -> Dict[str, int]:
    """Get RGB color of a specific pixel on screen."""
    screenshot = pyautogui.screenshot()
    color = screenshot.getpixel((x, y))
    return {"r": color[0], "g": color[1], "b": color[2]}


TOOL_NAMES.append("get_pixel_color")


# ============================================================================
# MOUSE CONTROL TOOLS
# ============================================================================

@mcp.tool()
def move_to(x: int, y: int, duration: float = 0.3) -> str:
    """Move mouse cursor to specified coordinates."""
    pyautogui.moveTo(x, y, duration=duration)
    return f"Moved mouse to ({x}, {y})"


TOOL_NAMES.append("move_to")


@mcp.tool()
def get_mouse_position() -> Dict[str, int]:
    """Get current mouse cursor position."""
    pos = pyautogui.position()
    return {"x": pos.x, "y": pos.y}


TOOL_NAMES.append("get_mouse_position")


# ============================================================================
# CLICK OPERATIONS TOOLS
# ============================================================================

@mcp.tool()
def click(x: Optional[int] = None, y: Optional[int] = None, clicks: int = 1, button: str = "left") -> str:
    """Perform left/right/middle click at coordinates."""
    if x is not None and y is not None:
        pyautogui.moveTo(x, y, duration=0.2)
    pyautogui.click(clicks=clicks, button=button)
    return f"Clicked ({clicks}x) {button} button at ({x}, {y})" if x and y else f"Clicked ({clicks}x) {button} button"


TOOL_NAMES.append("click")


@mcp.tool()
def right_click(x: Optional[int] = None, y: Optional[int] = None) -> str:
    """Right click at specified coordinates."""
    if x is not None and y is not None:
        pyautogui.moveTo(x, y, duration=0.2)
    pyautogui.rightClick()
    return f"Right clicked at ({x}, {y})" if x and y else "Right clicked"


TOOL_NAMES.append("right_click")


@mcp.tool()
def double_click(x: Optional[int] = None, y: Optional[int] = None) -> str:
    """Double click at specified coordinates."""
    if x is not None and y is not None:
        pyautogui.moveTo(x, y, duration=0.2)
    pyautogui.doubleClick()
    return f"Double clicked at ({x}, {y})" if x and y else "Double clicked"


TOOL_NAMES.append("double_click")


@mcp.tool()
def middle_click(x: Optional[int] = None, y: Optional[int] = None) -> str:
    """Middle click at specified coordinates."""
    if x is not None and y is not None:
        pyautogui.moveTo(x, y, duration=0.2)
    pyautogui.click(button='middle')
    return f"Middle clicked at ({x}, {y})" if x and y else "Middle clicked"


TOOL_NAMES.append("middle_click")


# ============================================================================
# DRAG & SCROLL TOOLS
# ============================================================================

@mcp.tool()
def drag_to(to_x: int, to_y: int, from_x: Optional[int] = None, from_y: Optional[int] = None, duration: float = 0.5) -> str:
    """Drag mouse from one position to another."""
    if from_x is not None and from_y is not None:
        pyautogui.moveTo(from_x, from_y)
    pyautogui.dragTo(to_x, to_y, duration=duration, button="left")
    return f"Dragged to ({to_x}, {to_y})"


TOOL_NAMES.append("drag_to")


@mcp.tool()
def scroll(clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> str:
    """Scroll at specified position."""
    if x is not None and y is not None:
        pyautogui.moveTo(x, y)
    pyautogui.scroll(clicks)
    direction = "up" if clicks > 0 else "down"
    return f"Scrolled {direction} {abs(clicks)} clicks"


TOOL_NAMES.append("scroll")


# ============================================================================
# KEYBOARD TOOLS
# ============================================================================

@mcp.tool()
def type_text(text: str, interval: float = 0.05) -> str:
    """Type text using keyboard."""
    pyautogui.write(text, interval=interval)
    return f"Typed: {text}"


TOOL_NAMES.append("type_text")


@mcp.tool()
def press_key(key: str) -> str:
    """Press a single key."""
    pyautogui.press(key)
    return f"Pressed key: {key}"


TOOL_NAMES.append("press_key")


@mcp.tool()
def hotkey(keys: List[str]) -> str:
    """Press multiple keys simultaneously as a keyboard shortcut."""
    pyautogui.hotkey(*keys)
    return f"Pressed hotkey: {'+'.join(keys)}"


TOOL_NAMES.append("hotkey")


# ============================================================================
# APPLICATION CONTROL TOOLS
# ============================================================================

@mcp.tool()
def open_app(app_name: str) -> str:
    """Open an application by name (cross-platform)."""
    sys_name = platform.system()
    try:
        if sys_name == "Darwin":
            os.system(f"open -a '{app_name}'")
        elif sys_name == "Windows":
            os.system(f'start "" "{app_name}"')
        else:
            os.system(f"{app_name} &")
        time.sleep(2)
        return f"Opened {app_name}"
    except Exception as e:
        return f"Error opening {app_name}: {str(e)}"


TOOL_NAMES.append("open_app")


@mcp.tool()
def close_app(app_name: str) -> str:
    """Close an application (cross-platform)."""
    sys_name = platform.system()
    try:
        if sys_name == "Windows":
            os.system(f'taskkill /im "{app_name}.exe" /f')
        elif sys_name == "Darwin":
            os.system(f'osascript -e \'quit app "{app_name}"\' ')
        else:
            os.system(f"pkill '{app_name}'")
        return f"Closed {app_name}"
    except Exception as e:
        return f"Error closing {app_name}: {str(e)}"


TOOL_NAMES.append("close_app")


# ============================================================================
# WINDOW MANAGEMENT TOOLS (PyGetWindow)
# ============================================================================

@mcp.tool()
def list_windows() -> Dict[str, Any]:
    """List all visible window titles on screen."""
    if not PYGETWINDOW_AVAILABLE:
        return {"found": False, "error": "pygetwindow not installed"}
    
    try:
        all_windows = gw.getAllTitles()
        windows = [w for w in all_windows if w]
        return {"found": True, "count": len(windows), "windows": windows}
    except Exception as e:
        return {"found": False, "error": str(e)}


TOOL_NAMES.append("list_windows")


@mcp.tool()
def focus_window(title: str) -> str:
    """Bring a window to the front by title."""
    if not PYGETWINDOW_AVAILABLE:
        return "Error: pygetwindow not installed"
    
    try:
        windows = gw.getWindowsWithTitle(title)
        if not windows:
            return f"Window '{title}' not found"
        
        window = windows[0]
        try:
            window.activate()
            return f"Focused window '{window.title}'"
        except:
            window.minimize()
            time.sleep(0.5)
            window.maximize()
            return f"Focused window '{window.title}'"
    except Exception as e:
        return f"Error: {str(e)}"


TOOL_NAMES.append("focus_window")


@mcp.tool()
def move_window(title: str, x: int, y: int) -> str:
    """Move a window to specified coordinates."""
    if not PYGETWINDOW_AVAILABLE:
        return "Error: pygetwindow not installed"
    
    try:
        windows = gw.getWindowsWithTitle(title)
        if not windows:
            return f"Window '{title}' not found"
        window = windows[0]
        window.moveTo(x, y)
        return f"Moved window to ({x}, {y})"
    except Exception as e:
        return f"Error: {str(e)}"


TOOL_NAMES.append("move_window")


@mcp.tool()
def resize_window(title: str, width: int, height: int) -> str:
    """Resize a window to specified dimensions."""
    if not PYGETWINDOW_AVAILABLE:
        return "Error: pygetwindow not installed"
    
    try:
        windows = gw.getWindowsWithTitle(title)
        if not windows:
            return f"Window '{title}' not found"
        window = windows[0]
        window.resizeTo(width, height)
        return f"Resized window to {width}x{height}"
    except Exception as e:
        return f"Error: {str(e)}"


TOOL_NAMES.append("resize_window")


# ============================================================================
# CLIPBOARD TOOLS (PyPerclip)
# ============================================================================

@mcp.tool()
def clipboard_get() -> Dict[str, Any]:
    """Get current clipboard contents (text only)."""
    if not PYPERCLIP_AVAILABLE:
        return {"found": False, "error": "pyperclip not installed"}
    
    try:
        text = pyperclip.paste()
        return {"found": True, "text": text}
    except Exception as e:
        return {"found": False, "error": str(e)}


TOOL_NAMES.append("clipboard_get")


@mcp.tool()
def clipboard_set(text: str) -> str:
    """Set clipboard to given text."""
    if not PYPERCLIP_AVAILABLE:
        return "Error: pyperclip not installed"
    
    try:
        pyperclip.copy(text)
        return f"Copied {len(text)} characters to clipboard"
    except Exception as e:
        return f"Error: {str(e)}"


TOOL_NAMES.append("clipboard_set")


# ============================================================================
# DIALOG BOX TOOLS (PyAutoGUI)
# ============================================================================

@mcp.tool()
def alert(text: str, title: str = "Alert", button: str = "OK") -> str:
    """Display an alert dialog box."""
    try:
        result = pyautogui.alert(text=text, title=title, button=button)
        return f"Alert closed. Button: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


TOOL_NAMES.append("alert")


@mcp.tool()
def confirm(text: str, title: str = "Confirm", buttons: List[str] = None) -> str:
    """Display a confirmation dialog with multiple buttons."""
    if buttons is None:
        buttons = ["OK", "Cancel"]
    
    try:
        result = pyautogui.confirm(text=text, title=title, buttons=buttons)
        return f"Dialog closed. Button: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


TOOL_NAMES.append("confirm")


@mcp.tool()
def prompt(text: str, title: str = "Input", default: str = "") -> str:
    """Display an input prompt dialog."""
    try:
        result = pyautogui.prompt(text=text, title=title, default=default)
        if result is None:
            return "Prompt cancelled"
        return f"User entered: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


TOOL_NAMES.append("prompt")


# ============================================================================
# TEXT DETECTION TOOLS (EasyOCR)
# ============================================================================

@mcp.tool()
def find_text_on_screen(target_text: str, confidence_threshold: float = 0.4) -> Dict[str, Any]:
    """Find text on screen using EasyOCR."""
    if not EASYOCR_AVAILABLE or not NUMPY_AVAILABLE:
        return {"found": False, "error": "EasyOCR not installed"}
    
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        img = ImageGrab.grab()
        img_np = np.array(img)
        results = reader.readtext(img_np, detail=1)
        
        target = target_text.lower()
        best = None
        
        for bbox, detected_text, conf in results:
            if conf < confidence_threshold:
                continue
            
            text_l = detected_text.lower()
            if target not in text_l and text_l not in target:
                continue
            
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            
            cand = {
                "text": detected_text,
                "confidence": conf,
                "center": (cx, cy),
                "bbox": {"top_left": (x_min, y_min), "bottom_right": (x_max, y_max)},
            }
            
            if best is None or conf > best["confidence"]:
                best = cand
        
        if best is None:
            return {"found": False, "text": target_text}
        
        return {"found": True, **best}
    except Exception as e:
        return {"found": False, "error": str(e)}


TOOL_NAMES.append("find_text_on_screen")


@mcp.tool()
def click_on_text(target_text: str, confidence_threshold: float = 0.4) -> str:
    """Find text on screen and click on it."""
    result = find_text_on_screen(target_text, confidence_threshold)
    
    if result.get("found"):
        cx, cy = result["center"]
        pyautogui.moveTo(cx, cy, duration=0.2)
        pyautogui.click(cx, cy)
        return f"Clicked '{result['text']}' at ({cx}, {cy})"
    else:
        return f"Text '{target_text}' not found"


TOOL_NAMES.append("click_on_text")


@mcp.tool()
def get_all_screen_text(confidence_threshold: float = 0.4) -> Dict[str, Any]:
    """Detect all text visible on screen."""
    if not EASYOCR_AVAILABLE or not NUMPY_AVAILABLE:
        return {"found": False, "error": "EasyOCR not installed"}
    
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        img = ImageGrab.grab()
        img_np = np.array(img)
        results = reader.readtext(img_np, detail=1)
        
        all_detections = []
        for bbox, detected_text, conf in results:
            if conf < confidence_threshold:
                continue
            
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            
            all_detections.append({
                "text": detected_text,
                "confidence": conf,
                "center": (cx, cy),
                "bbox": {"top_left": (x_min, y_min), "bottom_right": (x_max, y_max)},
            })
        
        return {"found": True, "count": len(all_detections), "texts": all_detections}
    except Exception as e:
        return {"found": False, "error": str(e)}


TOOL_NAMES.append("get_all_screen_text")


# ============================================================================
# IMAGE MATCHING TOOLS
# ============================================================================

@mcp.tool()
def locate_image_on_screen(image_path: str, confidence: float = 0.8) -> Dict[str, Any]:
    """Find an image on screen using PyAutoGUI."""
    try:
        location = pyautogui.locateOnScreen(image_path, confidence=confidence)
        if location:
            return {"found": True, "x": location.left, "y": location.top, "width": location.width, "height": location.height}
        return {"found": False, "image_path": image_path}
    except Exception as e:
        return {"found": False, "error": str(e)}


TOOL_NAMES.append("locate_image_on_screen")


@mcp.tool()
def click_on_image(image_path: str, confidence: float = 0.8) -> str:
    """Find image on screen and click on it."""
    result = locate_image_on_screen(image_path, confidence)
    if result.get("found"):
        x = result["x"] + result["width"] // 2
        y = result["y"] + result["height"] // 2
        pyautogui.moveTo(x, y, duration=0.2)
        pyautogui.click(x, y)
        return f"Clicked image at ({x}, {y})"
    else:
        return f"Image not found"


TOOL_NAMES.append("click_on_image")


# ============================================================================
# AI VISION TOOLS (Google Gemini - LLM-based UI Analysis)
# ============================================================================

@mcp.tool()
def analyze_screen_with_gemini(screenshot_base64: str = None, api_key: str = None) -> Dict[str, Any]:
    """
    Analyze entire screenshot using Google Gemini 3 Pro to identify ALL UI elements.
    
    Uses advanced vision AI to detect buttons, links, text, inputs, radio buttons, 
    checkboxes, etc. with precise pixel coordinates and confidence scores.
    """
    if not GEMINI_AVAILABLE:
        return {"found": False, "error": "google-genai not installed. Install: pip install google-genai"}
    
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"found": False, "error": "GEMINI_API_KEY not set"}
    
    try:
        if screenshot_base64 is None:
            screenshot = pyautogui.screenshot()
            if screenshot.mode == "RGBA":
                screenshot = screenshot.convert("RGB")
            buffer = BytesIO()
            screenshot.save(buffer, format="JPEG")
            buffer.seek(0)
            screenshot_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        client = genai.Client(api_key=api_key)
        model = "gemini-3-pro"
        
        prompt = """Analyze this screenshot carefully. Identify ALL clickable, interactive, or important UI elements visible.

For EACH element, extract: name, center_x, center_y, x1, y1, x2, y2, confidence (0-100), element_type, reason.

IMPORTANT: Find ALL elements. For radio/checkboxes, find the CLICKABLE area. Be precise.

Return ONLY valid JSON:
{
  "screen_resolution": "widthxheight",
  "elements": [
    {
      "name": "...",
      "center": {"x": 0, "y": 0},
      "bounds": {"x1": 0, "y1": 0, "x2": 0, "y2": 0},
      "confidence": 0-100,
      "element_type": "button/link/radio/checkbox/text/input/menu",
      "reason": "..."
    }
  ]
}"""
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(mime_type="image/jpeg", data=base64.b64decode(screenshot_base64)),
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        
        response_text = ""
        for chunk in client.models.generate_content_stream(model=model, contents=contents, config=types.GenerateContentConfig(temperature=0.2)):
            response_text += chunk.text
        
        result = json.loads(response_text)
        return {"found": True, "screen_resolution": result.get("screen_resolution", "Unknown"), "element_count": len(result.get("elements", [])), "elements": result.get("elements", [])}
    
    except json.JSONDecodeError as e:
        return {"found": False, "error": f"Invalid JSON: {str(e)}"}
    except Exception as e:
        return {"found": False, "error": f"Error: {str(e)}"}


TOOL_NAMES.append("analyze_screen_with_gemini")


@mcp.tool()
def find_element_with_gemini(element_description: str, screenshot_base64: str = None, api_key: str = None, multi_select: bool = False) -> Dict[str, Any]:
    """
    Find specific UI element(s) using Google Gemini 3 Pro vision AI.
    
    Perfect for finding radio buttons, checkboxes, buttons, etc.
    """
    if not GEMINI_AVAILABLE:
        return {"found": False, "error": "google-genai not installed"}
    
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"found": False, "error": "GEMINI_API_KEY not set"}
    
    try:
        if screenshot_base64 is None:
            screenshot = pyautogui.screenshot()
            if screenshot.mode == "RGBA":
                screenshot = screenshot.convert("RGB")
            buffer = BytesIO()
            screenshot.save(buffer, format="JPEG")
            buffer.seek(0)
            screenshot_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        client = genai.Client(api_key=api_key)
        model = "gemini-3-pro"
        
        if multi_select:
            prompt = f"""Find ALL elements matching: {element_description}

For EACH match: name, center_x/y, x1/y1/x2/y2, confidence, element_type, reason.

Return ONLY JSON:
{{
  "screen_resolution": "widthxheight",
  "found": true/false,
  "matches": [{{"name": "...", "center": {{"x": 0, "y": 0}}, "bounds": {{"x1": 0, "y1": 0, "x2": 0, "y2": 0}}, "confidence": 0, "element_type": "...", "reason": "..."}}],
  "match_count": 0
}}"""
        else:
            prompt = f"""Find the element: {element_description}

Extract: name, center_x/y, x1/y1/x2/y2, confidence, element_type, reason.

Return ONLY JSON:
{{
  "screen_resolution": "widthxheight",
  "found": true/false,
  "element": {{"name": "...", "center": {{"x": 0, "y": 0}}, "bounds": {{"x1": 0, "y1": 0, "x2": 0, "y2": 0}}, "confidence": 0, "element_type": "...", "reason": "..."}}
}}"""
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(mime_type="image/jpeg", data=base64.b64decode(screenshot_base64)),
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        
        response_text = ""
        for chunk in client.models.generate_content_stream(model=model, contents=contents, config=types.GenerateContentConfig(temperature=0.2)):
            response_text += chunk.text
        
        return json.loads(response_text)
    
    except json.JSONDecodeError as e:
        return {"found": False, "error": f"Invalid JSON: {str(e)}"}
    except Exception as e:
        return {"found": False, "error": f"Error: {str(e)}"}


TOOL_NAMES.append("find_element_with_gemini")


# ============================================================================
# UTILITY TOOLS
# ============================================================================

@mcp.tool()
def system_sleep(seconds: float) -> str:
    """Pause execution for specified seconds."""
    time.sleep(seconds)
    return f"Slept for {seconds} seconds"


TOOL_NAMES.append("system_sleep")


@mcp.tool()
def get_system_info() -> Dict[str, str]:
    """Get system information and available libraries."""
    return {
        "platform": platform.system(),
        "platform_version": platform.release(),
        "python_version": platform.python_version(),
        "easyocr_available": str(EASYOCR_AVAILABLE),
        "opencv_available": str(OPENCV_AVAILABLE),
        "pygetwindow_available": str(PYGETWINDOW_AVAILABLE),
        "pyperclip_available": str(PYPERCLIP_AVAILABLE),
        "gemini_available": str(GEMINI_AVAILABLE),
    }


TOOL_NAMES.append("get_system_info")


# ============================================================================
# START SERVER
# ============================================================================

if __name__ == "__main__":
    print("Starting Computer Automation MCP Server...")
    print("\nAvailable tools:")
    for tool_name in sorted(TOOL_NAMES):
        print(f"  ✓ {tool_name}")
    print(f"\nTotal tools: {len(TOOL_NAMES)}")
    print("\nServer running on stdio transport...")
    mcp.run()
