"""CDP browser audit for the branded sidebar action. Requires Chrome on port 9222."""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import requests
from websockets.sync.client import connect


DEBUG_URL = os.getenv("ZUBE_CHROME_DEBUG_URL", "http://127.0.0.1:9222")
APP_URL = os.getenv("ZUBE_AUDIT_URL", "http://localhost:8501")
SAMPLE = str((Path(__file__).parent / "assets" / "browser_sample.csv").resolve())
OUTPUT = Path("C:/tmp")


class Cdp:
    def __init__(self, websocket_url: str):
        self.socket = connect(websocket_url, origin=DEBUG_URL, max_size=None)
        self.identifier = 0

    def call(self, method: str, params: dict | None = None):
        self.identifier += 1
        current = self.identifier
        self.socket.send(json.dumps({"id": current, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("id") == current:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})

    def evaluate(self, expression: str):
        return self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})["result"].get("value")

    def screenshot(self, path: Path):
        result = self.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        path.write_bytes(base64.b64decode(result["data"]))


def wait_for(cdp: Cdp, expression: str, timeout: int = 30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cdp.evaluate(expression):
            return
        time.sleep(.5)
    raise TimeoutError(f"Browser condition not met: {expression}")


def main():
    targets = requests.get(f"{DEBUG_URL}/json", timeout=5).json()
    page = next(target for target in targets if target["type"] == "page")
    cdp = Cdp(page["webSocketDebuggerUrl"])
    cdp.call("Page.enable")
    cdp.call("Runtime.enable")
    cdp.call("DOM.enable")
    cdp.call("Page.navigate", {"url": APP_URL})
    wait_for(cdp, "document.querySelector('input[type=file]') !== null")

    document = cdp.call("DOM.getDocument")
    node = cdp.call("DOM.querySelector", {"nodeId": document["root"]["nodeId"], "selector": "input[type=file]"})
    cdp.call("DOM.setFileInputFiles", {"nodeId": node["nodeId"], "files": [SAMPLE]})
    wait_for(cdp, "Array.from(document.querySelectorAll('button')).some(b => b.innerText.includes('Analyze another file'))", 45)
    time.sleep(1)

    button_expression = "Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Analyze another file'))"
    normal = cdp.evaluate(
        f"""(() => {{ const b={button_expression}; const s=getComputedStyle(b); const r=b.getBoundingClientRect();
        return {{text:b.innerText,color:s.color,background:s.backgroundImage,disabled:b.disabled,x:r.x+r.width/2,y:r.y+r.height/2}}; }})()"""
    )
    assert normal["text"].strip(), "Sidebar button has no visible label"
    assert normal["color"] == "rgb(255, 255, 255)", normal
    assert "gradient" in normal["background"], normal
    assert not normal["disabled"], normal
    cdp.screenshot(OUTPUT / "zube-sidebar-desktop.png")

    cdp.call("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": normal["x"], "y": normal["y"]})
    time.sleep(.4)
    hover = cdp.evaluate(f"(() => {{ const s=getComputedStyle({button_expression}); return {{background:s.backgroundImage,transform:s.transform}}; }})()")
    assert hover["background"] != normal["background"], {"normal": normal, "hover": hover}

    focus = cdp.evaluate(f"(() => {{ const b={button_expression}; b.focus(); const s=getComputedStyle(b); return {{outline:s.outline,outlineColor:s.outlineColor}}; }})()")
    assert focus["outline"] != "none", focus
    assert focus["outlineColor"] != "rgba(0, 0, 0, 0)", focus

    cdp.call("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True})
    time.sleep(1)
    mobile = cdp.evaluate(f"(() => {{ const b={button_expression}; const r=b.getBoundingClientRect(); return {{text:b.innerText,width:r.width,height:r.height}}; }})()")
    assert mobile["text"].strip() and mobile["width"] > 100 and mobile["height"] >= 40, mobile
    cdp.screenshot(OUTPUT / "zube-sidebar-mobile.png")
    print(json.dumps({"normal": normal, "hover": hover, "focus": focus, "mobile": mobile}, indent=2))


if __name__ == "__main__":
    main()
