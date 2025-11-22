#!/usr/bin/env python3
"""
FastMCP Server for Computer Automation
Standalone implementation with no external dependencies
Uses fastmcp library (https://gofastmcp.com)
"""
import base64
import os
import time
import platform
import pyautogui
from io import BytesIO
from fastmcp import FastMCP, Image

# Create FastMCP server
mcp = FastMCP("Computer Control Server")

# ============================================================================
# TOOL IMPLEMENTATIONS
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
    
    # Return FastMCP Image type (NOT base64 string)
    return Image(data=buffer.getvalue(), format="jpeg")

@mcp.tool()
def open_app(app_name: str) -> str:
    """
    Open an application by name (cross-platform: Windows, macOS, Linux)
    
    Args:
        app_name: Name of the application (e.g., 'Excel', 'Calculator', 'Chrome', 'notepad')
    
    Returns:
        Confirmation message
    
    Examples:
        - Windows: 'notepad', 'calc', 'excel'
        - macOS: 'Calculator', 'TextEdit', 'Safari'
        - Linux: 'gedit', 'firefox', 'libreoffice'
    """
    sys_name = platform.system()
    
    if sys_name == "Darwin":  # macOS
        os.system(f"open -a '{app_name}'")
    elif sys_name == "Windows":
        os.system(f'start "" "{app_name}"')
    else:  # Linux
        os.system(f"{app_name} &")
    
    time.sleep(2)  # Wait for app to open
    return f"Opened {app_name}"

@mcp.tool()
def click(x: int | None = None, y: int | None = None, clicks: int = 1, button: str = "left") -> str:
    """
    Click at specified coordinates
    
    Args:
        x: X coordinate (optional, uses current mouse position if None)
        y: Y coordinate (optional, uses current mouse position if None)
        clicks: Number of clicks (default 1)
        button: Mouse button - 'left', 'right', or 'middle' (default 'left')
    
    Returns:
        Confirmation message
    """
    if x is not None and y is not None:
        pyautogui.moveTo(x, y, duration=0.3)
    pyautogui.click(clicks=clicks, button=button)
    return f"Clicked at ({x}, {y})" if x and y else "Clicked at current position"

@mcp.tool()
def right_click(x: int | None = None, y: int | None = None) -> str:
    """
    Right click at specified coordinates
    
    Args:
        x: X coordinate
        y: Y coordinate
    
    Returns:
        Confirmation message
    """
    if x is not None and y is not None:
        pyautogui.moveTo(x, y, duration=0.3)
    pyautogui.rightClick()
    return f"Right clicked at ({x}, {y})" if x and y else "Right clicked at current position"

@mcp.tool()
def double_click(x: int | None = None, y: int | None = None) -> str:
    """
    Double click at specified coordinates
    
    Args:
        x: X coordinate
        y: Y coordinate
    
    Returns:
        Confirmation message
    """
    if x is not None and y is not None:
        pyautogui.moveTo(x, y, duration=0.3)
    pyautogui.doubleClick()
    return f"Double clicked at ({x}, {y})" if x and y else "Double clicked at current position"

@mcp.tool()
def move_to(x: int, y: int, duration: float = 0.3) -> str:
    """
    Move mouse cursor to specified coordinates
    
    Args:
        x: Target X coordinate
        y: Target Y coordinate
        duration: Time in seconds to move (default 0.3)
    
    Returns:
        Confirmation message
    """
    pyautogui.moveTo(x, y, duration=duration)
    return f"Moved mouse to ({x}, {y})"

@mcp.tool()
def drag_to(to_x: int, to_y: int, from_x: int | None = None, from_y: int | None = None, duration: float = 0.5) -> str:
    """
    Drag mouse from one position to another
    
    Args:
        to_x: Target X coordinate
        to_y: Target Y coordinate
        from_x: Starting X coordinate (optional, uses current position if None)
        from_y: Starting Y coordinate (optional, uses current position if None)
        duration: Time in seconds for drag operation (default 0.5)
    
    Returns:
        Confirmation message
    """
    if from_x is not None and from_y is not None:
        pyautogui.moveTo(from_x, from_y)
    pyautogui.dragTo(to_x, to_y, duration=duration, button="left")
    return f"Dragged to ({to_x}, {to_y})"

@mcp.tool()
def scroll(clicks: int, x: int | None = None, y: int | None = None) -> str:
    """
    Scroll at specified position or current mouse position
    
    Args:
        clicks: Amount to scroll (positive values scroll up, negative values scroll down)
        x: X coordinate to scroll at (optional)
        y: Y coordinate to scroll at (optional)
    
    Returns:
        Confirmation message
    
    Examples:
        scroll(5) - scroll up 5 clicks at current position
        scroll(-3, 500, 300) - scroll down 3 clicks at (500, 300)
    """
    if x is not None and y is not None:
        pyautogui.moveTo(x, y)
    pyautogui.scroll(clicks)
    direction = "up" if clicks > 0 else "down"
    return f"Scrolled {direction} {abs(clicks)} clicks"

@mcp.tool()
def type_text(text: str, interval: float = 0.05) -> str:
    """
    Type text using keyboard
    
    Args:
        text: Text string to type
        interval: Time between keystrokes in seconds (default 0.05)
    
    Returns:
        Confirmation message
    """
    pyautogui.write(text, interval=interval)
    return f"Typed: {text}"

@mcp.tool()
def press_key(key: str) -> str:
    """
    Press a single key
    
    Args:
        key: Key name to press
    
    Returns:
        Confirmation message
    
    Common keys:
        - 'enter', 'return'
        - 'tab'
        - 'escape', 'esc'
        - 'space'
        - 'backspace'
        - 'delete', 'del'
        - 'up', 'down', 'left', 'right'
        - 'home', 'end'
        - 'pageup', 'pagedown'
        - 'f1' through 'f12'
    """
    pyautogui.press(key)
    return f"Pressed key: {key}"

@mcp.tool()
def hotkey(keys: list[str]) -> str:
    """
    Press multiple keys simultaneously (keyboard shortcuts)
    Args:
        keys: List of keys to press together (e.g., ['ctrl', 'c'])
    """
    pyautogui.hotkey(*keys)
    combo = '+'.join(keys)
    return f"Pressed hotkey: {combo}"

@mcp.tool()
def get_screen_size() -> dict:
    """
    Get the current screen resolution
    
    Returns:
        Dictionary with 'width' and 'height' in pixels
    
    Example return:
        {"width": 1920, "height": 1080}
    """
    size = pyautogui.size()
    return {"width": size.width, "height": size.height}

@mcp.tool()
def get_mouse_position() -> dict:
    """
    Get the current mouse cursor position
    
    Returns:
        Dictionary with 'x' and 'y' coordinates
    
    Example return:
        {"x": 500, "y": 300}
    """
    pos = pyautogui.position()
    return {"x": pos.x, "y": pos.y}

@mcp.tool()
def screenshot_region(x: int, y: int, width: int, height: int) -> Image:
    """
    Take a screenshot of a specific region and return as JPEG image
    
    Args:
        x: X coordinate of top-left corner
        y: Y coordinate of top-left corner
        width: Width of region in pixels
        height: Height of region in pixels
    
    Returns:
        Image object of the region in JPEG format
    """
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    if screenshot.mode == "RGBA":
        screenshot = screenshot.convert("RGB")
    buffer = BytesIO()
    screenshot.save(buffer, format="JPEG")
    
    # Return FastMCP Image type
    return Image(data=buffer.getvalue(), format="jpeg")

@mcp.tool()
def locate_on_screen(image_path: str, confidence: float = 0.8) -> dict | None:
    """
    Find an image on the screen and return its position
    
    Args:
        image_path: Path to the image file to locate
        confidence: Match confidence threshold 0.0-1.0 (default 0.8)
    
    Returns:
        Dictionary with position and size if found, None if not found
        {"x": int, "y": int, "width": int, "height": int}
    """
    try:
        location = pyautogui.locateOnScreen(image_path, confidence=confidence)
        if location:
            return {
                "x": location.left,
                "y": location.top,
                "width": location.width,
                "height": location.height
            }
        return None
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def alert(text: str, title: str = "Alert", button: str = "OK") -> str:
    """
    Display an alert dialog box
    
    Args:
        text: Message to display
        title: Dialog title (default "Alert")
        button: Button text (default "OK")
    
    Returns:
        The button text that was clicked
    """
    result = pyautogui.alert(text=text, title=title, button=button)
    return result

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    # Run FastMCP server with stdio transport (default)
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)

