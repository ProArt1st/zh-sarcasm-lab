"""后端自测脚本：验证配置读取、模型推理与 HTTP 接口。"""

from __future__ import annotations

import json
import sys
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import AppConfig, ensure_venv, load_config


class BackendTestCase(unittest.TestCase):
    """覆盖核心推理与 HTTP 接口的自测用例。"""

    @classmethod
    def setUpClass(cls) -> None:
        from backend.analyzer import SarcasmAnalyzer
        from backend.server import create_server

        cls.config = load_config()
        cls.analyzer = SarcasmAnalyzer(cls.config)
        test_cfg = AppConfig(
            host="127.0.0.1",
            port=0,
            model_dir=cls.config.model_dir,
            onnx_path=cls.config.onnx_path,
            tokenizer_path=cls.config.tokenizer_path,
            max_length=cls.config.max_length,
            positive_threshold=cls.config.positive_threshold,
            negative_threshold=cls.config.negative_threshold,
            examples=cls.config.examples,
            frontend_dir=cls.config.frontend_dir,
        )
        cls.server = create_server(test_cfg, cls.analyzer)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def test_analyze_preset_examples(self) -> None:
        for item in self.config.examples:
            result = self.analyzer.analyze(item["text"])
            self.assertEqual(result["text"], item["text"])
            self.assertIn(result["snownlp"]["label"], ("正面", "负面", "中性"))
            self.assertIn(result["roberta"]["label"], ("正面", "负面"))
            self.assertIsInstance(result["disagree"], bool)
            self.assertGreaterEqual(result["elapsed_ms"], 0.0)

    def test_analyze_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.analyzer.analyze("   ")

    def test_http_examples(self) -> None:
        with urlopen(f"http://127.0.0.1:{self.port}/api/examples", timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(len(data["examples"]), len(self.config.examples))

    def test_http_analyze_success(self) -> None:
        body = json.dumps({"text": "快递速度真快，整整走了一个星期"}).encode("utf-8")
        req = Request(
            f"http://127.0.0.1:{self.port}/api/analyze",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(data["roberta"]["label"], "负面")
        self.assertEqual(data["snownlp"]["label"], "中性")
        self.assertTrue(data["disagree"])

    def test_http_analyze_empty_returns_400(self) -> None:
        body = json.dumps({"text": "   "}).encode("utf-8")
        req = Request(
            f"http://127.0.0.1:{self.port}/api/analyze",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as ctx:
            urlopen(req, timeout=5)
        try:
            self.assertEqual(ctx.exception.code, 400)
            err = json.loads(ctx.exception.read().decode("utf-8"))
            self.assertIn("error", err)
        finally:
            ctx.exception.close()

    def test_http_root_serves_page(self) -> None:
        with urlopen(f"http://127.0.0.1:{self.port}/", timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            html = resp.read().decode("utf-8")
        self.assertIn("zh-sarcasm-lab", html)


if __name__ == "__main__":
    ensure_venv(Path(__file__))
    unittest.main(verbosity=2)
