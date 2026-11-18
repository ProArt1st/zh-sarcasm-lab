"""本地 Web 服务入口：提供接口并托管前端静态页面。"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import AppConfig, ensure_venv, load_config

if TYPE_CHECKING:
    from backend.analyzer import SarcasmAnalyzer

MAX_BODY_BYTES = 1024 * 1024

FALLBACK_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>zh-sarcasm-lab 后端已就绪</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 680px; margin: 60px auto; padding: 0 20px; line-height: 1.7; color: #222; }
    code { background: #f3f4f6; padding: 2px 6px; border-radius: 4px; }
  </style>
</head>
<body>
  <h2>zh-sarcasm-lab 后端服务已启动</h2>
  <p>当前未检测到 <code>frontend/index.html</code>，页面合并后刷新即可显示正式界面。</p>
  <p>接口已就绪：<code>GET /api/examples</code>、<code>POST /api/analyze</code>。</p>
</body>
</html>
"""


def make_handler(config: AppConfig, analyzer: SarcasmAnalyzer) -> type[BaseHTTPRequestHandler]:
    """构建绑定了配置和推理实例的 HTTP 请求处理器。"""

    class DemoRequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_frontend(self, request_path: str) -> None:
            rel_path = "index.html" if request_path in ("/", "") else request_path.lstrip("/")
            frontend_root = config.frontend_dir.resolve()
            target = (frontend_root / unquote(rel_path)).resolve()

            if not target.is_relative_to(frontend_root):
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "非法访问路径"})
                return

            if not target.is_file():
                if rel_path == "index.html":
                    self._send_bytes(
                        HTTPStatus.OK,
                        FALLBACK_HTML.encode("utf-8"),
                        "text/html; charset=utf-8",
                    )
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "未找到请求的资源"})
                return

            mime_type, _ = mimetypes.guess_type(str(target))
            content_type = mime_type or "application/octet-stream"
            if content_type.startswith("text/") or content_type == "application/javascript":
                content_type = f"{content_type}; charset=utf-8"
            self._send_bytes(HTTPStatus.OK, target.read_bytes(), content_type)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/examples":
                self._send_json(HTTPStatus.OK, {"examples": config.examples})
                return
            self._serve_frontend(path)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != "/api/analyze":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "接口不存在"})
                return

            raw_len = self.headers.get("Content-Length", "0")
            try:
                content_length = int(raw_len)
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "请求头 Content-Length 非法"})
                return

            if content_length <= 0 or content_length > MAX_BODY_BYTES:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "请输入待检测的中文文本"})
                return

            try:
                raw_body = self.rfile.read(content_length).decode("utf-8")
                payload = json.loads(raw_body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "请输入合法的 JSON 数据"})
                return

            if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "请输入待检测的中文文本"})
                return

            try:
                result = analyzer.analyze(payload["text"])
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            self._send_json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[HTTP] {self.address_string()} - {format % args}")

    return DemoRequestHandler


def create_server(config: AppConfig, analyzer: SarcasmAnalyzer) -> ThreadingHTTPServer:
    """创建本地多线程 HTTP 服务实例。"""
    handler_cls = make_handler(config, analyzer)
    return ThreadingHTTPServer((config.host, config.port), handler_cls)


def main() -> None:
    ensure_venv(Path(__file__))
    from backend.analyzer import SarcasmAnalyzer

    parser = argparse.ArgumentParser(description="zh-sarcasm-lab 后端本地服务")
    parser.add_argument("--no-browser", action="store_true", help="启动时不自动打开浏览器")
    args = parser.parse_args()

    try:
        config = load_config()
        print("正在加载离线模型，请稍候...")
        analyzer = SarcasmAnalyzer(config)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"启动失败：{exc}")
        sys.exit(1)

    try:
        server = create_server(config, analyzer)
    except OSError as exc:
        print(f"端口 {config.host}:{config.port} 启动失败：{exc}")
        print("可在根目录 config.json 中修改 server.port 后重试。")
        sys.exit(1)

    url = f"http://{config.host}:{config.port}"
    print(f"服务已启动，请访问：{url}")
    print("按 Ctrl+C 停止服务。")

    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=[url]).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
