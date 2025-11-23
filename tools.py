import base64
import os
import json
import time
import platform
from typing import Any, List
import pyautogui
from io import BytesIO

from typing import TypedDict, Any, Optional, List

class ActionInput(TypedDict):
    action_name: str
    parameters: dict

class State(TypedDict):
    input_query: str
    messages: List[Any]
    final_result: Optional[str]
    captured_actions: List[Any]


def convert_coordinates(coord, from_resolution=(1372, 887), to_resolution=None):
    x, y = coord
    from_width, from_height = from_resolution
    if to_resolution is None:
        to_resolution = pyautogui.size()
    to_width, to_height = to_resolution
    x_new = (x / from_width) * to_width
    y_new = (y / from_height) * to_height
    return round(x_new), round(y_new)

class Tools:
    _instance = None

    def __init__(self, virtual: bool = False):
        if virtual:
            raise Exception("Virtual Machine not available.")
        self._system = pyautogui
        self.os_name = platform.system().lower()
        self.executed_actions: List[dict] = []

    @classmethod
    def get_instance(cls, virtual: bool = False):
        if cls._instance is None:
            cls._instance = Tools(virtual)
            cls._instance._system.scroll(0)
        return cls._instance

    def get_available_actions(self):
        supported_actions = [
            {
                "action_name": "click",
                "description": "Perform left click operation at (x, y) coordinate",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "x coordinate to perform operation at. (default: None)"},
                        "y": {"type": "number", "description": "y coordinate to perform operation at. (default: None)"},
                        "clicks": {"type": "number", "description": "number of clicks to perform. (default: 1)"}
                    },
                    "required": []
                }
            },
            {
                "action_name": "move_to",
                "description": "Move pointer to (x, y) coordinate",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "x coordinate to move to."},
                        "y": {"type": "number", "description": "y coordinate to move to."},
                        "duration": {"type": "number", "description": "Time in second taken to move to. (default: 0)"}
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "action_name": "right_click",
                "description": "Perform right click operation at (x, y) coordinate",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "x coordinate"},
                        "y": {"type": "number", "description": "y coordinate"}
                    },
                    "required": []
                }
            },
            {
                "action_name": "double_click",
                "description": "Perform double click operation at (x, y) coordinate",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "x coordinate"},
                        "y": {"type": "number", "description": "y coordinate"}
                    },
                    "required": []
                }
            },
            {
                "action_name": "scroll",
                "description": "Perform scroll operation at (x, y) coordinate",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "clicks": {"type": "number", "description": "The amount of scrolling to perform. Positive for scroll up, negative for scroll down."},
                        "x": {"type": "number", "description": "x coordinate"},
                        "y": {"type": "number", "description": "y coordinate"},
                    },
                    "required": []
                }
            },
            {
                "action_name": "drag_to",
                "description": "Perform drag operation from one coordinate to another",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "from_x": {"type": "number", "description": "x coordinate to start dragging from. (default: None)"},
                        "from_y": {"type": "number", "description": "y coordinate to start dragging from. (default: None)"},
                        "to_x": {"type": "number", "description": "x coordinate to drag to."},
                        "to_y": {"type": "number", "description": "y coordinate to drag to."}
                    },
                    "required": ["to_x", "to_y"]
                }
            },
            {
                "action_name": "type",
                "description": "Type the characters in the string passed",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "The characters to type"}
                    },
                    "required": ["message"]
                }
            },
            {
                "action_name": "hotkey",
                "description": "Press the hotkeys in sequence",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "keys": {
                            "type": "array",
                            "description": "The characters to type",
                            "items": {"type": "string", "description": "key name to press. e.g., space, ctrl, command."}
                        }
                    },
                    "required": ["keys"]
                }
            },
            {
                "action_name": "open_app",
                "description": "Open an application by name using the system command.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "app_name": {
                            "type": "string",
                            "description": "The name of the application to open. e.g., 'Numbers', 'Pycharm', 'Visual Studio Code'."
                        }
                    },
                    "required": ["app_name"]
                }
            }
        ]
        return json.dumps(supported_actions, indent=2)

    def perform_action(self, action: ActionInput):
        action_name = action.get("action_name")
        parameters = action.get("parameters", {})
        func = getattr(self, action_name, None)
        if func:
            try:
                return func(**parameters)
            except Exception as e:
                return f"Error executing {action_name}: {str(e)}"
        return "Action not found"

    def screenshot(self):
        screenshot = pyautogui.screenshot()
        if screenshot.mode == "RGBA":
            screenshot = screenshot.convert("RGB")
        buffer = BytesIO()
        screenshot.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_str

    def open_app(self, app_name: str):
        # Cross-platform launcher for Mac and Windows
        if self.os_name == "darwin":
            os.system(f"open -a '{app_name}'")
            time.sleep(2)
            # Optional: bring app to front (macOS only)
            os.system(
                '''osascript -e 'tell application "System Events" to keystroke "f" using {command down, control down}' '''
            )
        elif self.os_name == "windows":
            os.system(f'start "" "{app_name}"')
            time.sleep(2)
        else:
            os.system(f"{app_name} &")
            time.sleep(2)
        return "Done"

    def click(self, x: Any = None, y: Any = None, clicks: int = 1):
        if x is not None and y is not None:
            self.move_to(x, y)
        self._system.click(clicks=clicks)

    def move_to(self, x: Any, y: Any, duration: int = 0.5):
        self._system.moveTo(x, y, duration=duration)

    def right_click(self, x: Any = None, y: Any = None):
        if x is not None and y is not None:
            self.move_to(x, y)
        self._system.rightClick()

    def double_click(self, x: Any = None, y: Any = None):
        if x is not None and y is not None:
            self.move_to(x, y)
        self._system.doubleClick(interval=0.1)

    def scroll(self, clicks: Any = None, x: Any = None, y: Any = None):
        if x is not None and y is not None:
            self.move_to(x, y)
        self._system.scroll(clicks)

    def drag_to(self, to_x: Any, to_y: Any, from_x: Any = None, from_y: Any = None):
        if from_x is not None and from_y is not None:
            self.move_to(from_x, from_y)
        self._system.dragTo(to_x, to_y, duration=1, button="left")

    def type(self, message: Any = None):
        self._system.write(message)

    def hotkey(self, keys: List[str]):
        self._system.hotkey(*keys)

    def has_done(self, action: ActionInput):
        action_name = action.get("action_name")
        return action_name and action_name.lower() == "done"

    def reset_actions(self):
        self.executed_actions = []

    def has_failed(self, action: ActionInput):
        action_name = action.get("action_name")
        return action_name and action_name.lower() == "failed"
