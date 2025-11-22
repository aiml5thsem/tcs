import base64
import os
import time
import platform
import pyautogui
from io import BytesIO
from typing import Dict, Any, List

# Action history for logging
action_history: List[Dict[str, Any]] = []

def log_action(action_name: str, parameters: Dict[str, Any], result: Any = None):
    """Log every action for final JSON output"""
    action_history.append({
        "action": action_name,
        "parameters": parameters,
        "result": str(result) if result else "completed"
    })

def take_screenshot() -> str:
    """Take screenshot and return as base64 JPEG"""
    screenshot = pyautogui.screenshot()
    if screenshot.mode == "RGBA":
        screenshot = screenshot.convert("RGB")
    buffer = BytesIO()
    screenshot.save(buffer, format="JPEG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()
    log_action("take_screenshot", {})
    return img_b64

def open_app(app_name: str) -> str:
    """Open application (cross-platform)"""
    sys_name = platform.system()
    if sys_name == "Darwin":
        os.system(f"open -a '{app_name}'")
    elif sys_name == "Windows":
        os.system(f'start "" "{app_name}"')
    else:
        os.system(f"{app_name} &")
    time.sleep(2)
    log_action("open_app", {"app_name": app_name})
    return f"Opened {app_name}"

def click(x: int = None, y: int = None, clicks: int = 1, button: str = "left") -> str:
    """Click at coordinates"""
    if x is not None and y is not None:
        pyautogui.moveTo(x, y, duration=0.3)
    pyautogui.click(clicks=clicks, button=button)
    log_action("click", {"x": x, "y": y, "clicks": clicks})
    return f"Clicked at ({x}, {y})"

def right_click(x: int = None, y: int = None) -> str:
    """Right click"""
    if x is not None and y is not None:
        pyautogui.moveTo(x, y, duration=0.3)
    pyautogui.rightClick()
    log_action("right_click", {"x": x, "y": y})
    return "Right clicked"

def double_click(x: int = None, y: int = None) -> str:
    """Double click"""
    if x is not None and y is not None:
        pyautogui.moveTo(x, y, duration=0.3)
    pyautogui.doubleClick()
    log_action("double_click", {"x": x, "y": y})
    return "Double clicked"

def move_to(x: int, y: int, duration: float = 0.3) -> str:
    """Move mouse"""
    pyautogui.moveTo(x, y, duration=duration)
    log_action("move_to", {"x": x, "y": y})
    return f"Moved to ({x}, {y})"

def drag_to(to_x: int, to_y: int, from_x: int = None, from_y: int = None) -> str:
    """Drag operation"""
    if from_x and from_y:
        pyautogui.moveTo(from_x, from_y)
    pyautogui.dragTo(to_x, to_y, duration=0.5, button="left")
    log_action("drag_to", {"from_x": from_x, "from_y": from_y, "to_x": to_x, "to_y": to_y})
    return f"Dragged to ({to_x}, {to_y})"

def scroll(clicks: int, x: int = None, y: int = None) -> str:
    """Scroll (positive=up, negative=down)"""
    if x and y:
        pyautogui.moveTo(x, y)
    pyautogui.scroll(clicks)
    log_action("scroll", {"clicks": clicks})
    return f"Scrolled {abs(clicks)} clicks"

def type_text(text: str) -> str:
    """Type text"""
    pyautogui.write(text, interval=0.05)
    log_action("type_text", {"text": text})
    return f"Typed: {text}"

def press_key(key: str) -> str:
    """Press single key"""
    pyautogui.press(key)
    log_action("press_key", {"key": key})
    return f"Pressed {key}"

def hotkey(*keys: str) -> str:
    """Press key combination"""
    pyautogui.hotkey(*keys)
    log_action("hotkey", {"keys": list(keys)})
    return f"Pressed {'+'.join(keys)}"

def get_screen_size() -> dict:
    """Get screen resolution"""
    size = pyautogui.size()
    return {"width": size.width, "height": size.height}

def get_mouse_position() -> dict:
    """Get mouse position"""
    pos = pyautogui.position()
    return {"x": pos.x, "y": pos.y}

def reset_action_history():
    """Clear action history"""
    global action_history
    action_history = []

def get_action_history() -> List[Dict[str, Any]]:
    """Get all logged actions"""
    return action_history
