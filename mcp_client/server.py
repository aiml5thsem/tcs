#!/usr/bin/env python3
"""
Universal Computer Control MCP Server - FIXED COORDINATE MAPPING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL FIX: Proper aspect-ratio-preserving screenshot scaling with letterboxing
to ensure LLM coordinates map correctly to screen coordinates.

Based on Anthropic's Computer Use with corrected coordinate transformation.
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
from PIL import Image as PILImage, ImageDraw

# FastMCP
from fastmcp import FastMCP
from fastmcp.utilities.types import Image

# Platform detection
PLATFORM = platform.system()
IS_WINDOWS = PLATFORM == "Windows"
IS_MACOS = PLATFORM == "Darwin"
IS_LINUX = PLATFORM == "Linux"

# Optional dependencies
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

# Initialize MCP server
mcp = FastMCP("Computer Control")

# PyAutoGUI safety
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

# Anthropic Computer Use max resolution
MAX_SCALING_TARGET = (1366, 768)  # WXGA 16:9


# ============================================================================
# FIXED: Coordinate Scaling with Aspect Ratio Preservation
# ============================================================================

@dataclass
class ScreenInfo:
    """Screen information with proper coordinate transformation"""
    # Actual screen
    screen_width: int
    screen_height: int
    
    # Scaled image dimensions (what LLM sees)
    scaled_width: int
    scaled_height: int
    
    # Coordinate transformation offsets (for letterboxing)
    offset_x: int
    offset_y: int
    
    # Active region scale (the actual content, not black bars)
    content_scale: float
    
    def to_dict(self) -> Dict:
        return {
            "actual_screen": f"{self.screen_width}x{self.screen_height}",
            "llm_sees": f"{self.scaled_width}x{self.scaled_height}",
            "content_scale": f"{self.content_scale:.4f}",
            "offset": f"({self.offset_x}, {self.offset_y})",
            "note": "LLM coordinates map directly to screenshot, auto-scaled to screen"
        }


class CoordinateScaler:
    """
    FIXED: Proper coordinate transformation with aspect ratio preservation.
    
    Key Concept:
    1. Screenshot is scaled to fit MAX_SCALING_TARGET while preserving aspect ratio
    2. This may create letterboxing (black bars on sides/top/bottom)
    3. Coordinates must account for letterboxing offset
    4. Scale factor is uniform (same for X and Y)
    """
    
    def __init__(self):
        self.screen_info = self._calculate_screen_info()
    
    def _calculate_screen_info(self) -> ScreenInfo:
        """Calculate proper scaling with aspect ratio preservation"""
        # Get actual screen size
        screen_width, screen_height = pyautogui.size()
        
        # Calculate scale to fit within MAX_SCALING_TARGET while preserving aspect ratio
        scale_x = MAX_SCALING_TARGET[0] / screen_width
        scale_y = MAX_SCALING_TARGET[1] / screen_height
        
        # Use the SMALLER scale to ensure image fits (letterboxing if needed)
        scale = min(scale_x, scale_y)
        
        # Calculate content dimensions (actual screenshot size after scaling)
        content_width = int(screen_width * scale)
        content_height = int(screen_height * scale)
        
        # Calculate letterbox offsets (to center the content)
        offset_x = (MAX_SCALING_TARGET[0] - content_width) // 2
        offset_y = (MAX_SCALING_TARGET[1] - content_height) // 2
        
        return ScreenInfo(
            screen_width=screen_width,
            screen_height=screen_height,
            scaled_width=MAX_SCALING_TARGET[0],
            scaled_height=MAX_SCALING_TARGET[1],
            offset_x=offset_x,
            offset_y=offset_y,
            content_scale=scale
        )
    
    def llm_to_screen(self, x: int, y: int) -> Tuple[int, int]:
        """
        Convert LLM coordinates (from screenshot) to actual screen coordinates.
        
        Process:
        1. Subtract letterbox offset to get content coordinates
        2. Scale up by content_scale to get actual screen coordinates
        3. Clamp to screen bounds
        """
        # Remove letterbox offset
        content_x = x - self.screen_info.offset_x
        content_y = y - self.screen_info.offset_y
        
        # Scale to actual screen coordinates
        screen_x = content_x / self.screen_info.content_scale
        screen_y = content_y / self.screen_info.content_scale
        
        # Round and clamp
        screen_x = int(round(screen_x))
        screen_y = int(round(screen_y))
        
        screen_x = max(0, min(screen_x, self.screen_info.screen_width - 1))
        screen_y = max(0, min(screen_y, self.screen_info.screen_height - 1))
        
        return screen_x, screen_y
    
    def screen_to_llm(self, x: int, y: int) -> Tuple[int, int]:
        """
        Convert actual screen coordinates to LLM coordinates.
        
        Process:
        1. Scale down by content_scale
        2. Add letterbox offset
        3. Clamp to scaled image bounds
        """
        # Scale to content coordinates
        content_x = x * self.screen_info.content_scale
        content_y = y * self.screen_info.content_scale
        
        # Add letterbox offset
        llm_x = int(round(content_x + self.screen_info.offset_x))
        llm_y = int(round(content_y + self.screen_info.offset_y))
        
        # Clamp to scaled image bounds
        llm_x = max(0, min(llm_x, self.screen_info.scaled_width - 1))
        llm_y = max(0, min(llm_y, self.screen_info.scaled_height - 1))
        
        return llm_x, llm_y
    
    def scale_screenshot(self, screenshot: PILImage.Image) -> PILImage.Image:
        """
        Scale screenshot with aspect ratio preservation and letterboxing.
        
        Returns image of size MAX_SCALING_TARGET with letterboxing if needed.
        """
        # Calculate content size (scaled screenshot)
        content_width = int(screenshot.width * self.screen_info.content_scale)
        content_height = int(screenshot.height * self.screen_info.content_scale)
        
        # Resize screenshot (preserving aspect ratio)
        scaled_content = screenshot.resize(
            (content_width, content_height),
            PILImage.Resampling.LANCZOS
        )
        
        # Create canvas with black letterboxing
        canvas = PILImage.new(
            'RGB',
            MAX_SCALING_TARGET,
            (0, 0, 0)  # Black background
        )
        
        # Paste scaled content onto canvas (centered with letterboxing)
        canvas.paste(
            scaled_content,
            (self.screen_info.offset_x, self.screen_info.offset_y)
        )
        
        return canvas
    
    def refresh(self):
        """Refresh screen info (if resolution changes)"""
        self.screen_info = self._calculate_screen_info()


# Global scaler instance
_scaler = CoordinateScaler()

def get_scaler() -> CoordinateScaler:
    """Get global coordinate scaler"""
    return _scaler


# ============================================================================
# Mouse & Keyboard Control (Updated with fixed coordinates)
# ============================================================================

@mcp.tool()
def mouse_move(x: int, y: int, duration: float = 0.2) -> str:
    """
    Move mouse to coordinates (in LLM screenshot space).
    
    Args:
        x: X coordinate from screenshot (0 to 1366)
        y: Y coordinate from screenshot (0 to 768)
        duration: Movement duration in seconds
    
    Returns:
        Success message with coordinate transformation info
    """
    scaler = get_scaler()
    actual_x, actual_y = scaler.llm_to_screen(x, y)
    
    pyautogui.moveTo(actual_x, actual_y, duration=duration)
    
    return f"Moved mouse: LLM({x}, {y}) → Screen({actual_x}, {actual_y})"


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
        x: X coordinate from screenshot (None = current position)
        y: Y coordinate from screenshot (None = current position)
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
        location_str = f"LLM({x}, {y}) → Screen({actual_x}, {actual_y})"
    else:
        pos = pyautogui.position()
        location_str = f"current Screen({pos.x}, {pos.y})"
    
    pyautogui.click(clicks=clicks, interval=interval, button=button)
    
    return f"Clicked {button} {clicks}x at {location_str}"


@mcp.tool()
def mouse_double_click(x: Optional[int] = None, y: Optional[int] = None) -> str:
    """Double-click at coordinates or current position."""
    return mouse_click(x, y, button="left", clicks=2, interval=0.1)


@mcp.tool()
def mouse_right_click(x: Optional[int] = None, y: Optional[int] = None) -> str:
    """Right-click at coordinates or current position."""
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
        duration: Drag duration
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
    
    return f"Dragged LLM({from_x}, {from_y}) → LLM({to_x}, {to_y})"


@mcp.tool()
def mouse_scroll(clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> str:
    """
    Scroll mouse wheel.
    
    Args:
        clicks: Number of clicks (positive=up, negative=down)
        x: Optional X position in screenshot space
        y: Optional Y position in screenshot space
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
    Get current mouse position in both coordinate systems.
    
    Returns:
        Position in screen and screenshot coordinates
    """
    scaler = get_scaler()
    pos = pyautogui.position()
    
    llm_x, llm_y = scaler.screen_to_llm(pos.x, pos.y)
    
    return {
        "screen_coordinates": {
            "x": pos.x,
            "y": pos.y
        },
        "llm_coordinates": {
            "x": llm_x,
            "y": llm_y,
            "note": "Use these coordinates for mouse commands"
        },
        "transformation": {
            "screen_to_llm": f"({pos.x}, {pos.y}) → ({llm_x}, {llm_y})",
            "scale_factor": scaler.screen_info.content_scale
        }
    }


@mcp.tool()
def type_text(text: str, interval: float = 0.05) -> str:
    """Type text at current cursor position."""
    pyautogui.write(text, interval=interval)
    return f"Typed {len(text)} characters"


@mcp.tool()
def press_key(key: str, presses: int = 1, interval: float = 0.1) -> str:
    """Press a keyboard key."""
    for _ in range(presses):
        pyautogui.press(key)
        if presses > 1:
            time.sleep(interval)
    return f"Pressed '{key}' {presses}x"


@mcp.tool()
def press_hotkey(key1: str, key2: str, key3: str = "", key4: str = "") -> str:
    """Press keyboard hotkey combination."""
    keys = [key1, key2]
    if key3:
        keys.append(key3)
    if key4:
        keys.append(key4)
    
    pyautogui.hotkey(*keys)
    return f"Pressed: {'+'.join(keys)}"


# ============================================================================
# Screenshot & Screen Information (FIXED)
# ============================================================================

@mcp.tool()
def take_screenshot() -> Image:
    """
    Take a screenshot scaled for LLM processing.
    
    Returns screenshot with proper aspect ratio preservation and letterboxing.
    Coordinates from analyzing this image map directly to screen via auto-scaling.
    
    Returns:
        Screenshot as Image (1366x768 with letterboxing if needed)
    """
    scaler = get_scaler()
    
    # Capture full screen
    screenshot = pyautogui.screenshot()
    
    # Convert RGBA to RGB if needed
    if screenshot.mode == "RGBA":
        screenshot = screenshot.convert("RGB")
    
    # Scale with aspect ratio preservation (adds letterboxing)
    scaled = scaler.scale_screenshot(screenshot)
    
    # Convert to JPEG
    buffer = BytesIO()
    scaled.save(buffer, format="JPEG", quality=95)
    
    return Image(data=buffer.getvalue(), format="jpeg")


@mcp.tool()
def take_screenshot_region(x: int, y: int, width: int, height: int) -> Image:
    """
    Take screenshot of a region (coordinates in screenshot space).
    
    Args:
        x: Top-left X in screenshot space
        y: Top-left Y in screenshot space
        width: Width in screenshot space
        height: Height in screenshot space
    """
    scaler = get_scaler()
    
    # Convert to screen coordinates
    screen_x1, screen_y1 = scaler.llm_to_screen(x, y)
    screen_x2, screen_y2 = scaler.llm_to_screen(x + width, y + height)
    
    actual_width = screen_x2 - screen_x1
    actual_height = screen_y2 - screen_y1
    
    screenshot = pyautogui.screenshot(region=(screen_x1, screen_y1, actual_width, actual_height))
    
    if screenshot.mode == "RGBA":
        screenshot = screenshot.convert("RGB")
    
    buffer = BytesIO()
    screenshot.save(buffer, format="JPEG", quality=95)
    
    return Image(data=buffer.getvalue(), format="jpeg")


@mcp.tool()
def get_screen_info() -> Dict:
    """
    Get screen resolution and coordinate transformation info.
    
    Returns:
        Detailed screen and scaling information
    """
    scaler = get_scaler()
    info = scaler.screen_info.to_dict()
    
    info["explanation"] = (
        "Screenshots preserve aspect ratio with letterboxing. "
        "LLM coordinates (0-1366, 0-768) auto-scale to actual screen. "
        "Scale factor is uniform for accurate coordinate mapping."
    )
    
    return info


@mcp.tool()
def get_pixel_color(x: int, y: int) -> Dict:
    """
    Get RGB color of pixel (coordinates in screenshot space).
    
    Args:
        x: X in screenshot space
        y: Y in screenshot space
    """
    scaler = get_scaler()
    actual_x, actual_y = scaler.llm_to_screen(x, y)
    
    screenshot = pyautogui.screenshot()
    color = screenshot.getpixel((actual_x, actual_y))
    
    return {
        "rgb": {"r": color[0], "g": color[1], "b": color[2]},
        "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
        "coordinates": {
            "llm": f"({x}, {y})",
            "screen": f"({actual_x}, {actual_y})"
        }
    }


# ============================================================================
# Window Management (Keep existing implementations)
# ============================================================================

@mcp.tool()
def list_windows() -> List[str]:
    """List all open window titles."""
    if gw is None:
        return ["Error: pygetwindow not installed"]
    
    try:
        titles = [title for title in gw.getAllTitles() if title.strip()]
        return titles or ["No windows found"]
    except Exception as e:
        return [f"Error: {str(e)}"]


@mcp.tool()
def get_active_window() -> Dict:
    """Get information about active window."""
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
    """Focus window by title (partial match)."""
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
    """Maximize window."""
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
    """Minimize window."""
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
# Application Control (Keep existing)
# ============================================================================

@mcp.tool()
def open_application(app_name: str, wait_time: float = 2.0) -> str:
    """Open application."""
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
    """Close application by name."""
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/im", f"{app_name}.exe", "/f"], capture_output=True)
            return f"Closed: {app_name}"
        elif IS_MACOS:
            subprocess.run(["osascript", "-e", f'quit app "{app_name}"'], capture_output=True)
            return f"Closed: {app_name}"
        else:
            subprocess.run(["pkill", app_name], capture_output=True)
            return f"Closed: {app_name}"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def list_processes() -> List[Dict]:
    """List running processes (top 50 by memory)."""
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
# Clipboard Operations (Keep existing)
# ============================================================================

@mcp.tool()
def clipboard_get() -> Dict:
    """Get clipboard contents."""
    if pyperclip is None:
        return {"error": "pyperclip not installed"}
    
    try:
        text = pyperclip.paste()
        return {"text": text, "length": len(text)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def clipboard_set(text: str) -> str:
    """Set clipboard contents."""
    if pyperclip is None:
        return "Error: pyperclip not installed"
    
    try:
        pyperclip.copy(text)
        return f"Copied {len(text)} characters to clipboard"
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# OCR (Keep existing)
# ============================================================================

@mcp.tool()
def ocr_screenshot(x: Optional[int] = None, y: Optional[int] = None,
                   width: Optional[int] = None, height: Optional[int] = None) -> Dict:
    """Extract text from screenshot using OCR."""
    if pytesseract is None:
        return {"error": "pytesseract not installed"}
    
    try:
        scaler = get_scaler()
        
        if all(v is not None for v in [x, y, width, height]):
            screen_x1, screen_y1 = scaler.llm_to_screen(x, y)
            screen_x2, screen_y2 = scaler.llm_to_screen(x + width, y + height)
            actual_width = screen_x2 - screen_x1
            actual_height = screen_y2 - screen_y1
            screenshot = pyautogui.screenshot(region=(screen_x1, screen_y1, actual_width, actual_height))
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
# Image Recognition (Keep existing)
# ============================================================================

@mcp.tool()
def find_image_on_screen(image_path: str, confidence: float = 0.8) -> Dict:
    """Find image on screen."""
    if pyscreeze is None:
        return {"error": "pyscreeze not installed"}
    
    try:
        scaler = get_scaler()
        result = pyautogui.locateOnScreen(image_path, confidence=confidence)
        
        if result:
            center_x = result.left + result.width // 2
            center_y = result.top + result.height // 2
            
            llm_x, llm_y = scaler.screen_to_llm(center_x, center_y)
            
            return {
                "found": True,
                "llm_coordinates": {"x": llm_x, "y": llm_y},
                "screen_coordinates": {"x": center_x, "y": center_y},
                "width": result.width,
                "height": result.height
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
    """Get system information."""
    scaler = get_scaler()
    
    return {
        "platform": PLATFORM,
        "platform_version": platform.version(),
        "python_version": sys.version,
        "screen_info": scaler.screen_info.to_dict(),
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
    """Wait for duration."""
    time.sleep(seconds)
    return f"Waited {seconds} seconds"


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Computer Control MCP Server - FIXED COORDINATE MAPPING")
    print("=" * 70)
    print(f"Platform: {PLATFORM}")
    
    scaler = get_scaler()
    print(f"Actual Screen: {scaler.screen_info.screen_width}x{scaler.screen_info.screen_height}")
    print(f"LLM Sees: {scaler.screen_info.scaled_width}x{scaler.screen_info.scaled_height}")
    print(f"Content Scale: {scaler.screen_info.content_scale:.4f}x (uniform)")
    print(f"Letterbox Offset: ({scaler.screen_info.offset_x}, {scaler.screen_info.offset_y})")
    print("=" * 70)
    print("\nCoordinate mapping: Aspect ratio preserved, letterboxing applied")
    print("LLM coordinates auto-scale accurately to screen\n")
    
    mcp.run()
