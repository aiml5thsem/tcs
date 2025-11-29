#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Server for Computer Automation using FastMCP.
Exposes all automation functions as MCP tools for LLM integration.
Includes window management, clipboard, and dialog box tools.
"""

import base64
import os
import time
import platform
import pyautogui
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

from fastmcp import FastMCP, Image as MCPImage

# Pillow compatibility
Image.ANTIALIAS = Image.LANCZOS

# Create FastMCP server
mcp = FastMCP("Computer Automation Tools")


# ============================================================================
# SCREENSHOT & IMAGE CAPTURE TOOLS
# ============================================================================

@mcp.tool()
def take_screenshot(format: str = "base64") -> str:
    """
    Take full screenshot in specified format.
    
    Args:
        format: Return format - "array_info", "base64" (default), or "file"
                - "array_info": Returns shape and size info (for array format)
                - "base64": Base64 encoded JPEG string (good for APIs)
                - "file": Filepath string (saved as screenshot.png)
    
    Returns:
        Base64 string, file path, or info about array format
    
    Examples:
        take_screenshot()  # Returns base64 by default
        take_screenshot("file")  # Saves and returns filepath
    """
    screenshot = pyautogui.screenshot()
    
    if format.lower() == "array_info":
        arr_info = np.array(screenshot)
        return f"Array format available: shape={arr_info.shape}, dtype=uint8, Can be used internally"
    
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
    
    else:
        return f"Error: Unknown format '{format}'. Use 'base64', 'array_info', or 'file'"


@mcp.tool()
def screenshot_region(x: int, y: int, width: int, height: int, format: str = "base64") -> str:
    """
    Capture specific region of screen in specified format.
    
    Args:
        x: X coordinate of top-left corner
        y: Y coordinate of top-left corner
        width: Width of region in pixels
        height: Height of region in pixels
        format: Return format - "array_info", "base64" (default), or "file"
    
    Returns:
        Base64 string, file path, or info about array format
    """
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    
    if format.lower() == "array_info":
        arr_info = np.array(screenshot)
        return f"Region array available: shape={arr_info.shape}, Can be used internally"
    
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
    
    else:
        return f"Error: Unknown format '{format}'. Use 'base64', 'array_info', or 'file'"


@mcp.tool()
def get_screen_size() -> Dict[str, int]:
    """Get screen resolution."""
    size = pyautogui.size()
    return {"width": size.width, "height": size.height}


@mcp.tool()
def get_pixel_color(x: int, y: int) -> Dict[str, int]:
    """Get RGB color of a specific pixel on screen."""
    screenshot = pyautogui.screenshot()
    color = screenshot.getpixel((x, y))
    return {"r": color[0], "g": color[1], "b": color[2]}


# ============================================================================
# MOUSE CONTROL TOOLS
# ============================================================================

@mcp.tool()
def move_to(x: int, y: int, duration: float = 0.3) -> str:
    """Move mouse cursor to specified coordinates."""
    pyautogui.moveTo(x, y, duration=duration)
    return f"Moved mouse to ({x}, {y})"


@mcp.tool()
def get_mouse_position() -> Dict[str, int]:
    """Get current mouse cursor position."""
    pos = pyautogui.position()
    return {"x": pos.x, "y": pos.y}


# ============================================================================
# CLICK OPERATIONS TOOLS
# ============================================================================

@mcp.tool()
def click(x: Optional[int] = None, y: Optional[int] = None, clicks: int = 1, button: str = "left") -> str:
    """
    Perform left/right/middle click at coordinates.
    
    Args:
        x: X coordinate (optional, uses current position if None)
        y: Y coordinate (optional, uses current position if None)
        clicks: Number of clicks (default 1)
        button: Mouse button - 'left', 'right', or 'middle' (default 'left')
    """
    if x is not None and y is not None:
        pyautogui.moveTo(x, y, duration=0.2)
    pyautogui.click(clicks=clicks, button=button)
    return f"Clicked ({clicks}x) {button} button at ({x}, {y})" if x and y else f"Clicked ({clicks}x) {button} button"


@mcp.tool()
def right_click(x: Optional[int] = None, y: Optional[int] = None) -> str:
    """Right click at specified coordinates."""
    if x is not None and y is not None:
        pyautogui.moveTo(x, y, duration=0.2)
    pyautogui.rightClick()
    return f"Right clicked at ({x}, {y})" if x and y else "Right clicked at current position"


@mcp.tool()
def double_click(x: Optional[int] = None, y: Optional[int] = None) -> str:
    """Double click at specified coordinates."""
    if x is not None and y is not None:
        pyautogui.moveTo(x, y, duration=0.2)
    pyautogui.doubleClick()
    return f"Double clicked at ({x}, {y})" if x and y else "Double clicked at current position"


@mcp.tool()
def middle_click(x: Optional[int] = None, y: Optional[int] = None) -> str:
    """Middle click at specified coordinates."""
    if x is not None and y is not None:
        pyautogui.moveTo(x, y, duration=0.2)
    pyautogui.click(button='middle')
    return f"Middle clicked at ({x}, {y})" if x and y else "Middle clicked at current position"


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


@mcp.tool()
def scroll(clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> str:
    """Scroll at specified position or current mouse position."""
    if x is not None and y is not None:
        pyautogui.moveTo(x, y)
    pyautogui.scroll(clicks)
    direction = "up" if clicks > 0 else "down"
    return f"Scrolled {direction} {abs(clicks)} clicks"


# ============================================================================
# KEYBOARD TOOLS
# ============================================================================

@mcp.tool()
def type_text(text: str, interval: float = 0.05) -> str:
    """Type text using keyboard."""
    pyautogui.write(text, interval=interval)
    return f"Typed: {text}"


@mcp.tool()
def press_key(key: str) -> str:
    """Press a single key."""
    pyautogui.press(key)
    return f"Pressed key: {key}"


@mcp.tool()
def hotkey(keys: List[str]) -> str:
    """Press multiple keys simultaneously as a keyboard shortcut."""
    pyautogui.hotkey(*keys)
    return f"Pressed hotkey: {'+'.join(keys)}"


# ============================================================================
# APPLICATION CONTROL TOOLS
# ============================================================================

@mcp.tool()
def open_app(app_name: str) -> str:
    """Open an application by name (cross-platform)."""
    sys_name = platform.system()
    try:
        if sys_name == "Darwin":  # macOS
            os.system(f"open -a '{app_name}'")
        elif sys_name == "Windows":
            os.system(f'start "" "{app_name}"')
        else:  # Linux
            os.system(f"{app_name} &")
        time.sleep(2)
        return f"Opened {app_name}"
    except Exception as e:
        return f"Error opening {app_name}: {str(e)}"


@mcp.tool()
def close_app(app_name: str) -> str:
    """Close an application (cross-platform)."""
    sys_name = platform.system()
    try:
        if sys_name == "Windows":
            os.system(f'taskkill /im "{app_name}.exe" /f')
        elif sys_name == "Darwin":  # macOS
            os.system(f'osascript -e \'quit app "{app_name}"\' ')
        else:  # Linux
            os.system(f"pkill '{app_name}'")
        return f"Closed {app_name}"
    except Exception as e:
        return f"Error closing {app_name}: {str(e)}"


# ============================================================================
# WINDOW MANAGEMENT TOOLS (PyGetWindow)
# ============================================================================

@mcp.tool()
def list_windows() -> Dict[str, Any]:
    """
    List all visible window titles on screen.
    
    Returns:
        Dictionary with found status and list of window titles
        {"found": true, "count": N, "windows": ["Window 1", "Window 2", ...]}
    """
    if not PYGETWINDOW_AVAILABLE:
        return {"found": False, "error": "pygetwindow not installed. Install with: pip install pygetwindow"}
    
    try:
        all_windows = gw.getAllTitles()
        windows = [w for w in all_windows if w]  # Filter empty titles
        return {
            "found": True,
            "count": len(windows),
            "windows": windows
        }
    except Exception as e:
        return {"found": False, "error": str(e)}


@mcp.tool()
def focus_window(title: str) -> str:
    """
    Bring a window to the front by title.
    
    Args:
        title: Window title to focus (partial match supported)
    
    Returns:
        Confirmation message
    """
    if not PYGETWINDOW_AVAILABLE:
        return "Error: pygetwindow not installed. Install with: pip install pygetwindow"
    
    try:
        windows = gw.getWindowsWithTitle(title)
        if not windows:
            return f"Window '{title}' not found"
        
        window = windows[0]
        try:
            window.activate()
            return f"Focused window '{window.title}'"
        except:
            # Fallback: minimize and maximize
            window.minimize()
            time.sleep(0.5)
            window.maximize()
            return f"Focused window '{window.title}' (using minimize/maximize)"
    
    except Exception as e:
        return f"Error focusing window: {str(e)}"


@mcp.tool()
def move_window(title: str, x: int, y: int) -> str:
    """
    Move a window to specified coordinates.
    
    Args:
        title: Window title to move
        x: Target X coordinate
        y: Target Y coordinate
    
    Returns:
        Confirmation message
    """
    if not PYGETWINDOW_AVAILABLE:
        return "Error: pygetwindow not installed"
    
    try:
        windows = gw.getWindowsWithTitle(title)
        if not windows:
            return f"Window '{title}' not found"
        
        window = windows[0]
        window.moveTo(x, y)
        return f"Moved window '{window.title}' to ({x}, {y})"
    
    except Exception as e:
        return f"Error moving window: {str(e)}"


@mcp.tool()
def resize_window(title: str, width: int, height: int) -> str:
    """
    Resize a window to specified dimensions.
    
    Args:
        title: Window title to resize
        width: Target width in pixels
        height: Target height in pixels
    
    Returns:
        Confirmation message
    """
    if not PYGETWINDOW_AVAILABLE:
        return "Error: pygetwindow not installed"
    
    try:
        windows = gw.getWindowsWithTitle(title)
        if not windows:
            return f"Window '{title}' not found"
        
        window = windows[0]
        window.resizeTo(width, height)
        return f"Resized window '{window.title}' to {width}x{height}"
    
    except Exception as e:
        return f"Error resizing window: {str(e)}"


# ============================================================================
# CLIPBOARD TOOLS (PyPerclip)
# ============================================================================

@mcp.tool()
def clipboard_get() -> Dict[str, Any]:
    """
    Get current clipboard contents (text only).
    
    Returns:
        Dictionary with clipboard text
        {"found": true, "text": "clipboard content"}
    """
    if not PYPERCLIP_AVAILABLE:
        return {"found": False, "error": "pyperclip not installed. Install with: pip install pyperclip"}
    
    try:
        text = pyperclip.paste()
        return {"found": True, "text": text}
    except Exception as e:
        return {"found": False, "error": str(e)}


@mcp.tool()
def clipboard_set(text: str) -> str:
    """
    Set clipboard to given text.
    
    Args:
        text: Text to copy to clipboard
    
    Returns:
        Confirmation message
    """
    if not PYPERCLIP_AVAILABLE:
        return "Error: pyperclip not installed. Install with: pip install pyperclip"
    
    try:
        pyperclip.copy(text)
        return f"Copied {len(text)} characters to clipboard"
    except Exception as e:
        return f"Error copying to clipboard: {str(e)}"


# ============================================================================
# DIALOG BOX TOOLS (PyAutoGUI - PyMsgBox)
# ============================================================================

@mcp.tool()
def alert(text: str, title: str = "Alert", button: str = "OK") -> str:
    """
    Display an alert dialog box with OK button.
    
    Args:
        text: Message to display
        title: Dialog title (default "Alert")
        button: Button text (default "OK")
    
    Returns:
        Text of button clicked
    """
    try:
        result = pyautogui.alert(text=text, title=title, button=button)
        return f"Alert closed. Button clicked: {result}"
    except Exception as e:
        return f"Error showing alert: {str(e)}"


@mcp.tool()
def confirm(text: str, title: str = "Confirm", buttons: List[str] = None) -> str:
    """
    Display a confirmation dialog with multiple buttons.
    
    Args:
        text: Message to display
        title: Dialog title (default "Confirm")
        buttons: List of button labels (default ["OK", "Cancel"])
    
    Returns:
        Text of button clicked
    
    Examples:
        confirm("Do you want to proceed?")
        confirm("Choose option", buttons=["Yes", "No", "Cancel"])
    """
    if buttons is None:
        buttons = ["OK", "Cancel"]
    
    try:
        result = pyautogui.confirm(text=text, title=title, buttons=buttons)
        return f"Confirm dialog closed. Button clicked: {result}"
    except Exception as e:
        return f"Error showing confirm dialog: {str(e)}"


@mcp.tool()
def prompt(text: str, title: str = "Input", default: str = "") -> str:
    """
    Display an input prompt dialog.
    
    Args:
        text: Message/prompt to display
        title: Dialog title (default "Input")
        default: Default text in input field (default "")
    
    Returns:
        Text entered by user, or None if cancelled
    
    Example:
        prompt("Enter your name:", default="Anonymous")
    """
    try:
        result = pyautogui.prompt(text=text, title=title, default=default)
        if result is None:
            return "Prompt cancelled"
        return f"User entered: {result}"
    except Exception as e:
        return f"Error showing prompt: {str(e)}"


# ============================================================================
# TEXT DETECTION & CLICKING TOOLS (EasyOCR)
# ============================================================================

@mcp.tool()
def find_text_on_screen(target_text: str, confidence_threshold: float = 0.4) -> Dict[str, Any]:
    """Find text on screen using EasyOCR and return its position."""
    if not EASYOCR_AVAILABLE or not NUMPY_AVAILABLE:
        return {"found": False, "error": "EasyOCR not installed. Install with: pip install easyocr"}
    
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
                "bbox": {
                    "top_left": (x_min, y_min),
                    "bottom_right": (x_max, y_max),
                },
            }
            
            if best is None or conf > best["confidence"]:
                best = cand
        
        if best is None:
            return {"found": False, "text": target_text}
        
        return {"found": True, **best}
    
    except Exception as e:
        return {"found": False, "error": str(e)}


@mcp.tool()
def click_on_text(target_text: str, confidence_threshold: float = 0.4) -> str:
    """Find text on screen and click on it."""
    result = find_text_on_screen(target_text, confidence_threshold)
    
    if result.get("found"):
        cx, cy = result["center"]
        pyautogui.moveTo(cx, cy, duration=0.2)
        pyautogui.click(cx, cy)
        return f"Found and clicked '{result['text']}' at ({cx}, {cy}) with confidence {result['confidence']:.2f}"
    else:
        return f"Text '{target_text}' not found on screen"


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
                "bbox": {
                    "top_left": (x_min, y_min),
                    "bottom_right": (x_max, y_max),
                },
            })
        
        return {
            "found": True,
            "count": len(all_detections),
            "texts": all_detections
        }
    
    except Exception as e:
        return {"found": False, "error": str(e)}


# ============================================================================
# IMAGE MATCHING TOOLS
# ============================================================================

@mcp.tool()
def locate_image_on_screen(image_path: str, confidence: float = 0.8) -> Dict[str, Any]:
    """Find an image on screen using PyAutoGUI."""
    try:
        location = pyautogui.locateOnScreen(image_path, confidence=confidence)
        if location:
            return {
                "found": True,
                "x": location.left,
                "y": location.top,
                "width": location.width,
                "height": location.height,
            }
        return {"found": False, "image_path": image_path}
    except Exception as e:
        return {"found": False, "error": str(e)}


@mcp.tool()
def click_on_image(image_path: str, confidence: float = 0.8) -> str:
    """Find image on screen and click on it."""
    result = locate_image_on_screen(image_path, confidence)
    if result.get("found"):
        x = result["x"] + result["width"] // 2
        y = result["y"] + result["height"] // 2
        pyautogui.moveTo(x, y, duration=0.2)
        pyautogui.click(x, y)
        return f"Found and clicked image at ({x}, {y})"
    else:
        return f"Image '{image_path}' not found on screen"


# ============================================================================
# UTILITY TOOLS
# ============================================================================

@mcp.tool()
def system_sleep(seconds: float) -> str:
    """Pause execution for specified seconds."""
    time.sleep(seconds)
    return f"Slept for {seconds} seconds"


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
    }


# ============================================================================
# START SERVER
# ============================================================================

if __name__ == "__main__":
    print("Starting Computer Automation MCP Server...")
    print("\nAvailable tools:")
    for tool_name in sorted(mcp._tools.keys()):
        print(f"  ✓ {tool_name}")
    print(f"\nTotal tools: {len(mcp._tools)}")
    print("\nServer running on stdio transport...")
    mcp.run()
