#!/usr/bin/env python3
"""
Universal Computer Control MCP Server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Complete cross-platform computer automation with proper coordinate scaling.

Features:
✅ Proper screenshot scaling (XGA/WXGA for LLM processing)
✅ Accurate coordinate transformation (LLM space ↔ Real screen)
✅ Cross-platform support (Windows, macOS, Linux)
✅ Mouse & keyboard control
✅ Window management
✅ OCR support
✅ Clipboard operations
✅ Image recognition
✅ Process control

Based on Anthropic's Computer Use implementation.

Author: FastMCP 2.0 Implementation
"""

import base64
import os
import sys
import time
import platform
import subprocess
from io import BytesIO
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

# Core libraries
import pyautogui
from PIL import Image as PILImage

# FastMCP - Fixed import
from fastmcp import FastMCP
from fastmcp.utilities.types import Image

# Platform detection
PLATFORM = platform.system()
IS_WINDOWS = PLATFORM == "Windows"
IS_MACOS = PLATFORM == "Darwin"
IS_LINUX = PLATFORM == "Linux"

# Optional dependencies with graceful degradation
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
except ImportError:
    pytesseract = None

try:
    import pyscreeze
except ImportError:
    pyscreeze = None

try:
    import psutil
except ImportError:
    psutil = None

# Initialize MCP server - Fixed: removed dependencies parameter
mcp = FastMCP("Computer Control")

# ============================================================================
# Configuration & Constants
# ============================================================================

# Anthropic Computer Use recommended max resolutions
MAX_SCALING_TARGETS = {
    "XGA": (1024, 768),      # 4:3 aspect ratio
    "WXGA": (1280, 800),     # 16:10 aspect ratio
    "WXGA_16_9": (1366, 768) # 16:9 aspect ratio
}

# PyAutoGUI safety settings
pyautogui.FAILSAFE = True  # Move mouse to corner to abort
pyautogui.PAUSE = 0.1      # 100ms pause between actions

# ============================================================================
# Coordinate Scaling System
# ============================================================================

@dataclass
class ScreenInfo:
    """Screen information and scaling parameters"""
    width: int
    height: int
    scale_width: int
    scale_height: int
    scale_ratio_x: float
    scale_ratio_y: float
    
    def to_dict(self) -> Dict:
        return {
            "actual_resolution": f"{self.width}x{self.height}",
            "scaled_resolution": f"{self.scale_width}x{self.scale_height}",
            "scale_ratio": f"{self.scale_ratio_x:.4f}x{self.scale_ratio_y:.4f}"
        }


class CoordinateScaler:
    """
    Handles coordinate transformation between LLM space and actual screen space.
    
    The LLM sees a downscaled screenshot (e.g., 1024x768) but the actual screen
    might be 1920x1080. This class transforms coordinates bidirectionally.
    """
    
    def __init__(self):
        self.screen_info = self._get_screen_info()
    
    def _get_screen_info(self) -> ScreenInfo:
        """Get actual screen size and calculate scaling parameters"""
        actual_width, actual_height = pyautogui.size()
        
        # Determine best scaling target based on aspect ratio
        aspect_ratio = actual_width / actual_height
        
        if abs(aspect_ratio - 4/3) < 0.1:
            target = MAX_SCALING_TARGETS["XGA"]
        elif abs(aspect_ratio - 16/10) < 0.1:
            target = MAX_SCALING_TARGETS["WXGA"]
        else:
            target = MAX_SCALING_TARGETS["WXGA_16_9"]
        
        # Scale down if actual resolution is larger than target
        if actual_width > target[0] or actual_height > target[1]:
            scale_x = target[0] / actual_width
            scale_y = target[1] / actual_height
            scale = min(scale_x, scale_y)
            
            scaled_width = int(actual_width * scale)
            scaled_height = int(actual_height * scale)
        else:
            scaled_width = actual_width
            scaled_height = actual_height
        
        # Calculate scaling ratios
        ratio_x = actual_width / scaled_width
        ratio_y = actual_height / scaled_height
        
        return ScreenInfo(
            width=actual_width,
            height=actual_height,
            scale_width=scaled_width,
            scale_height=scaled_height,
            scale_ratio_x=ratio_x,
            scale_ratio_y=ratio_y
        )
    
    def llm_to_screen(self, x: int, y: int) -> Tuple[int, int]:
        """Convert LLM coordinates to actual screen coordinates"""
        actual_x = int(x * self.screen_info.scale_ratio_x)
        actual_y = int(y * self.screen_info.scale_ratio_y)
        
        # Clamp to screen bounds
        actual_x = max(0, min(actual_x, self.screen_info.width - 1))
        actual_y = max(0, min(actual_y, self.screen_info.height - 1))
        
        return actual_x, actual_y
    
    def screen_to_llm(self, x: int, y: int) -> Tuple[int, int]:
        """Convert actual screen coordinates to LLM coordinates"""
        scaled_x = int(x / self.screen_info.scale_ratio_x)
        scaled_y = int(y / self.screen_info.scale_ratio_y)
        
        scaled_x = max(0, min(scaled_x, self.screen_info.scale_width - 1))
        scaled_y = max(0, min(scaled_y, self.screen_info.scale_height - 1))
        
        return scaled_x, scaled_y
    
    def scale_screenshot(self, screenshot: PILImage.Image) -> PILImage.Image:
        """Scale screenshot to LLM-friendly resolution"""
        if screenshot.size == (self.screen_info.scale_width, self.screen_info.scale_height):
            return screenshot
        
        scaled = screenshot.resize(
            (self.screen_info.scale_width, self.screen_info.scale_height),
            PILImage.Resampling.LANCZOS
        )
        
        return scaled
    
    def refresh(self):
        """Refresh screen info"""
        self.screen_info = self._get_screen_info()


# Global scaler instance
_scaler = CoordinateScaler()


def get_scaler() -> CoordinateScaler:
    """Get global coordinate scaler instance"""
    return _scaler


# ============================================================================
# Mouse & Keyboard Control
# ============================================================================

@mcp.tool()
def mouse_move(x: int, y: int, duration: float = 0.2) -> str:
    """
    Move mouse to coordinates (in LLM screenshot space).
    
    Args:
        x: X coordinate in screenshot space
        y: Y coordinate in screenshot space
        duration: Movement duration in seconds
    
    Returns:
        Success message
    """
    scaler = get_scaler()
    actual_x, actual_y = scaler.llm_to_screen(x, y)
    
    pyautogui.moveTo(actual_x, actual_y, duration=duration)
    
    return f"Moved mouse to ({x}, {y}) screenshot → ({actual_x}, {actual_y}) screen"


@mcp.tool()
def mouse_click(
    x: Optional[int] = None,
    y: Optional[int] = None,
    button: str = "left",
    clicks: int = 1,
    interval: float = 0.1
) -> str:
    """
    Click mouse at coordinates or current position.
    
    Args:
        x: X coordinate in screenshot space (None = current position)
        y: Y coordinate in screenshot space (None = current position)
        button: Mouse button ("left", "right", "middle")
        clicks: Number of clicks
        interval: Interval between clicks
    
    Returns:
        Success message
    """
    scaler = get_scaler()
    
    if x is not None and y is not None:
        actual_x, actual_y = scaler.llm_to_screen(x, y)
        pyautogui.moveTo(actual_x, actual_y, duration=0.1)
        location_str = f"at ({x}, {y}) screenshot"
    else:
        location_str = "at current position"
    
    pyautogui.click(clicks=clicks, interval=interval, button=button)
    
    return f"Clicked {button} button {clicks}x {location_str}"


@mcp.tool()
def mouse_double_click(x: Optional[int] = None, y: Optional[int] = None) -> str:
    """
    Double-click at coordinates or current position.
    
    Args:
        x: X coordinate in screenshot space
        y: Y coordinate in screenshot space
    
    Returns:
        Success message
    """
    return mouse_click(x, y, button="left", clicks=2, interval=0.1)


@mcp.tool()
def mouse_right_click(x: Optional[int] = None, y: Optional[int] = None) -> str:
    """
    Right-click at coordinates or current position.
    
    Args:
        x: X coordinate in screenshot space
        y: Y coordinate in screenshot space
    
    Returns:
        Success message
    """
    return mouse_click(x, y, button="right", clicks=1)


@mcp.tool()
def mouse_drag(
    from_x: int,
    from_y: int,
    to_x: int,
    to_y: int,
    duration: float = 0.5,
    button: str = "left"
) -> str:
    """
    Drag mouse from one position to another.
    
    Args:
        from_x: Start X in screenshot space
        from_y: Start Y in screenshot space
        to_x: End X in screenshot space
        to_y: End Y in screenshot space
        duration: Drag duration in seconds
        button: Mouse button
    
    Returns:
        Success message
    """
    scaler = get_scaler()
    
    actual_from_x, actual_from_y = scaler.llm_to_screen(from_x, from_y)
    actual_to_x, actual_to_y = scaler.llm_to_screen(to_x, to_y)
    
    pyautogui.moveTo(actual_from_x, actual_from_y)
    pyautogui.drag(
        actual_to_x - actual_from_x,
        actual_to_y - actual_from_y,
        duration=duration,
        button=button
    )
    
    return f"Dragged from ({from_x}, {from_y}) to ({to_x}, {to_y})"


@mcp.tool()
def mouse_scroll(clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> str:
    """
    Scroll mouse wheel.
    
    Args:
        clicks: Number of clicks (positive=up, negative=down)
        x: Optional X position in screenshot space
        y: Optional Y position in screenshot space
    
    Returns:
        Success message
    """
    scaler = get_scaler()
    
    if x is not None and y is not None:
        actual_x, actual_y = scaler.llm_to_screen(x, y)
        pyautogui.moveTo(actual_x, actual_y)
    
    pyautogui.scroll(clicks)
    
    direction = "up" if clicks > 0 else "down"
    return f"Scrolled {direction} {abs(clicks)} clicks"


@mcp.tool()
def get_mouse_position() -> Dict:
    """
    Get current mouse position.
    
    Returns:
        Mouse position in both screen and screenshot coordinates
    """
    scaler = get_scaler()
    pos = pyautogui.position()
    
    scaled_x, scaled_y = scaler.screen_to_llm(pos.x, pos.y)
    
    return {
        "screen_x": pos.x,
        "screen_y": pos.y,
        "screenshot_x": scaled_x,
        "screenshot_y": scaled_y,
        "note": "Use screenshot_x/y for mouse commands"
    }


@mcp.tool()
def type_text(text: str, interval: float = 0.05) -> str:
    """
    Type text at current cursor position.
    
    Args:
        text: Text to type
        interval: Interval between keystrokes
    
    Returns:
        Success message
    """
    pyautogui.write(text, interval=interval)
    return f"Typed {len(text)} characters"


@mcp.tool()
def press_key(key: str, presses: int = 1, interval: float = 0.1) -> str:
    """
    Press a keyboard key.
    
    Args:
        key: Key name (e.g., "enter", "esc", "tab", "a", "ctrl", "shift")
        presses: Number of times to press
        interval: Interval between presses
    
    Returns:
        Success message
    """
    for _ in range(presses):
        pyautogui.press(key)
        if presses > 1:
            time.sleep(interval)
    
    return f"Pressed '{key}' {presses}x"


@mcp.tool()
def press_hotkey(key1: str, key2: str, key3: str = "", key4: str = "") -> str:
    """
    Press keyboard hotkey combination.
    
    Args:
        key1: First key (e.g., "ctrl", "cmd", "alt")
        key2: Second key (e.g., "c", "v", "tab")
        key3: Optional third key
        key4: Optional fourth key
    
    Returns:
        Success message
    
    Examples:
        press_hotkey("ctrl", "c") → Ctrl+C
        press_hotkey("ctrl", "shift", "esc") → Ctrl+Shift+Esc
    """
    # Fixed: No *args, explicit parameters
    keys = [key1, key2]
    if key3:
        keys.append(key3)
    if key4:
        keys.append(key4)
    
    pyautogui.hotkey(*keys)
    return f"Pressed: {'+'.join(keys)}"


# ============================================================================
# Screenshot & Screen Information
# ============================================================================

@mcp.tool()
def take_screenshot() -> Image:
    """
    Take a screenshot scaled for LLM processing.
    
    Returns scaled screenshot (max 1366x768). Coordinates from analyzing
    this image should be used directly in mouse commands.
    
    Returns:
        Screenshot as Image
    """
    scaler = get_scaler()
    
    screenshot = pyautogui.screenshot()
    
    if screenshot.mode == "RGBA":
        screenshot = screenshot.convert("RGB")
    
    scaled = scaler.scale_screenshot(screenshot)
    
    buffer = BytesIO()
    scaled.save(buffer, format="JPEG", quality=95)
    
    return Image(data=buffer.getvalue(), format="jpeg")


@mcp.tool()
def take_screenshot_region(x: int, y: int, width: int, height: int) -> Image:
    """
    Take screenshot of a region.
    
    Args:
        x: Top-left X in screenshot space
        y: Top-left Y in screenshot space
        width: Width in screenshot space
        height: Height in screenshot space
    
    Returns:
        Screenshot of region
    """
    scaler = get_scaler()
    
    actual_x, actual_y = scaler.llm_to_screen(x, y)
    actual_width = int(width * scaler.screen_info.scale_ratio_x)
    actual_height = int(height * scaler.screen_info.scale_ratio_y)
    
    screenshot = pyautogui.screenshot(region=(actual_x, actual_y, actual_width, actual_height))
    
    if screenshot.mode == "RGBA":
        screenshot = screenshot.convert("RGB")
    
    buffer = BytesIO()
    screenshot.save(buffer, format="JPEG", quality=95)
    
    return Image(data=buffer.getvalue(), format="jpeg")


@mcp.tool()
def get_screen_info() -> Dict:
    """
    Get screen resolution and scaling information.
    
    Returns:
        Screen dimensions and scaling details
    """
    scaler = get_scaler()
    info = scaler.screen_info.to_dict()
    
    info["explanation"] = (
        "Screenshots are scaled for LLM. Use scaled_resolution coordinates "
        "in mouse commands - they auto-scale to actual_resolution."
    )
    
    return info


@mcp.tool()
def get_pixel_color(x: int, y: int) -> Dict:
    """
    Get RGB color of pixel.
    
    Args:
        x: X in screenshot space
        y: Y in screenshot space
    
    Returns:
        RGB color
    """
    scaler = get_scaler()
    actual_x, actual_y = scaler.llm_to_screen(x, y)
    
    screenshot = pyautogui.screenshot()
    color = screenshot.getpixel((actual_x, actual_y))
    
    return {
        "r": color[0],
        "g": color[1],
        "b": color[2],
        "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
    }


# ============================================================================
# Window Management
# ============================================================================

@mcp.tool()
def list_windows() -> List[str]:
    """
    List all open window titles.
    
    Returns:
        List of window titles
    """
    if gw is None:
        return ["Error: pygetwindow not installed"]
    
    try:
        titles = []
        for title in gw.getAllTitles():
            if title.strip():
                titles.append(title)
        return titles or ["No windows found"]
    except Exception as e:
        return [f"Error: {str(e)}"]


@mcp.tool()
def get_active_window() -> Dict:
    """
    Get information about active window.
    
    Returns:
        Window info
    """
    if gw is None:
        return {"error": "pygetwindow not installed"}
    
    try:
        window = gw.getActiveWindow()
        if window:
            return {
                "title": window.title,
                "left": window.left,
                "top": window.top,
                "width": window.width,
                "height": window.height,
                "isMaximized": window.isMaximized,
                "isMinimized": window.isMinimized
            }
        return {"error": "No active window"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def focus_window(title: str) -> str:
    """
    Focus window by title (partial match).
    
    Args:
        title: Window title or partial title
    
    Returns:
        Success or error message
    """
    if gw is None:
        return "Error: pygetwindow not installed"
    
    try:
        windows = gw.getWindowsWithTitle(title)
        if windows:
            windows[0].activate()
            return f"Focused: {windows[0].title}"
        return f"Window not found: {title}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def maximize_window(title: str) -> str:
    """
    Maximize window.
    
    Args:
        title: Window title (partial match)
    
    Returns:
        Success message
    """
    if gw is None:
        return "Error: pygetwindow not installed"
    
    try:
        windows = gw.getWindowsWithTitle(title)
        if windows:
            windows[0].maximize()
            return f"Maximized: {windows[0].title}"
        return f"Window not found: {title}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def minimize_window(title: str) -> str:
    """
    Minimize window.
    
    Args:
        title: Window title (partial match)
    
    Returns:
        Success message
    """
    if gw is None:
        return "Error: pygetwindow not installed"
    
    try:
        windows = gw.getWindowsWithTitle(title)
        if windows:
            windows[0].minimize()
            return f"Minimized: {windows[0].title}"
        return f"Window not found: {title}"
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# Application Control
# ============================================================================

@mcp.tool()
def open_application(app_name: str, wait_time: float = 2.0) -> str:
    """
    Open application.
    
    Args:
        app_name: Application name or path
        wait_time: Wait time in seconds
    
    Returns:
        Success message
    """
    try:
        if IS_MACOS:
            subprocess.Popen(["open", "-a", app_name])
        elif IS_WINDOWS:
            subprocess.Popen(["start", "", app_name], shell=True)
        else:
            subprocess.Popen([app_name])
        
        time.sleep(wait_time)
        return f"Opened: {app_name}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def close_application(app_name: str) -> str:
    """
    Close application by name.
    
    Args:
        app_name: Application/process name
    
    Returns:
        Success message
    """
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/im", f"{app_name}.exe", "/f"], 
                         capture_output=True)
            return f"Closed: {app_name}"
        elif IS_MACOS:
            subprocess.run(["osascript", "-e", f'quit app "{app_name}"'],
                         capture_output=True)
            return f"Closed: {app_name}"
        else:
            subprocess.run(["pkill", app_name], capture_output=True)
            return f"Closed: {app_name}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def list_processes() -> List[Dict]:
    """
    List running processes.
    
    Returns:
        List of processes (top 50 by memory)
    """
    if psutil is None:
        return [{"error": "psutil not installed"}]
    
    try:
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
            try:
                info = proc.info
                processes.append({
                    "pid": info['pid'],
                    "name": info['name'],
                    "memory_percent": round(info['memory_percent'], 2)
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        processes.sort(key=lambda x: x['memory_percent'], reverse=True)
        return processes[:50]
    except Exception as e:
        return [{"error": str(e)}]


# ============================================================================
# Clipboard Operations
# ============================================================================

@mcp.tool()
def clipboard_get() -> Dict:
    """
    Get clipboard contents.
    
    Returns:
        Clipboard text
    """
    if pyperclip is None:
        return {"error": "pyperclip not installed"}
    
    try:
        text = pyperclip.paste()
        return {"text": text, "length": len(text)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def clipboard_set(text: str) -> str:
    """
    Set clipboard contents.
    
    Args:
        text: Text to copy
    
    Returns:
        Success message
    """
    if pyperclip is None:
        return "Error: pyperclip not installed"
    
    try:
        pyperclip.copy(text)
        return f"Copied {len(text)} characters to clipboard"
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# OCR
# ============================================================================

@mcp.tool()
def ocr_screenshot(x: Optional[int] = None, y: Optional[int] = None,
                   width: Optional[int] = None, height: Optional[int] = None) -> Dict:
    """
    Extract text from screenshot using OCR.
    
    Args:
        x: Optional region X in screenshot space
        y: Optional region Y in screenshot space
        width: Optional region width
        height: Optional region height
    
    Returns:
        Extracted text
    """
    if pytesseract is None:
        return {"error": "pytesseract not installed"}
    
    try:
        scaler = get_scaler()
        
        if all(v is not None for v in [x, y, width, height]):
            actual_x, actual_y = scaler.llm_to_screen(x, y)
            actual_width = int(width * scaler.screen_info.scale_ratio_x)
            actual_height = int(height * scaler.screen_info.scale_ratio_y)
            screenshot = pyautogui.screenshot(region=(actual_x, actual_y, actual_width, actual_height))
        else:
            screenshot = pyautogui.screenshot()
        
        text = pytesseract.image_to_string(screenshot)
        
        return {
            "text": text.strip(),
            "length": len(text.strip()),
            "lines": text.strip().count('\n') + 1 if text.strip() else 0
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# Image Recognition
# ============================================================================

@mcp.tool()
def find_image_on_screen(image_path: str, confidence: float = 0.8) -> Dict:
    """
    Find image on screen.
    
    Args:
        image_path: Path to template image
        confidence: Match confidence (0.0-1.0)
    
    Returns:
        Location in screenshot space
    """
    if pyscreeze is None:
        return {"error": "pyscreeze not installed"}
    
    try:
        scaler = get_scaler()
        result = pyautogui.locateOnScreen(image_path, confidence=confidence)
        
        if result:
            center_x = result.left + result.width // 2
            center_y = result.top + result.height // 2
            
            screenshot_x, screenshot_y = scaler.screen_to_llm(center_x, center_y)
            
            return {
                "found": True,
                "screenshot_x": screenshot_x,
                "screenshot_y": screenshot_y,
                "width": int(result.width / scaler.screen_info.scale_ratio_x),
                "height": int(result.height / scaler.screen_info.scale_ratio_y)
            }
        else:
            return {"found": False}
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# Utility
# ============================================================================

@mcp.tool()
def get_system_info() -> Dict:
    """
    Get system information.
    
    Returns:
        Platform and capability info
    """
    return {
        "platform": PLATFORM,
        "platform_version": platform.version(),
        "python_version": sys.version,
        "screen_info": get_scaler().screen_info.to_dict(),
        "capabilities": {
            "window_management": gw is not None,
            "clipboard": pyperclip is not None,
            "ocr": pytesseract is not None,
            "image_recognition": pyscreeze is not None,
            "process_management": psutil is not None
        }
    }


@mcp.tool()
def wait(seconds: float) -> str:
    """
    Wait for duration.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Success message
    """
    time.sleep(seconds)
    return f"Waited {seconds} seconds"


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Computer Control MCP Server")
    print("=" * 70)
    print(f"Platform: {PLATFORM}")
    
    scaler = get_scaler()
    print(f"Screen: {scaler.screen_info.width}x{scaler.screen_info.height}")
    print(f"LLM sees: {scaler.screen_info.scale_width}x{scaler.screen_info.scale_height}")
    print(f"Scale: {scaler.screen_info.scale_ratio_x:.2f}x, {scaler.screen_info.scale_ratio_y:.2f}y")
    print("=" * 70)
    print("\nServer ready. Coordinates auto-scale.\n")
    
    mcp.run()
