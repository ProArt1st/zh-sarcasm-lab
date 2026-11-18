"""后端自测脚本：验证配置读取、模型推理与 HTTP 接口。"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
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

        base_cfg = load_config()
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.config = AppConfig(
            host="127.0.0.1",
            port=0,
            model_dir=base_cfg.model_dir,
            onnx_path=base_cfg.onnx_path,
            tokenizer_path=base_cfg.tokenizer_path,
            max_length=base_cfg.max_length,
            positive_threshold=base_cfg.positive_threshold,
            negative_threshold=base_cfg.negative_threshold,
            examples=base_cfg.examples,
            frontend_dir=base_cfg.frontend_dir,
            cache_enabled=True,
            cache_max_entries=100,
            cache_path=Path(cls.tmp_dir.name) / "results.json",
        )
        cls.analyzer = SarcasmAnalyzer(cls.config)
        cls.server = create_server(cls.config, cls.analyzer)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp_dir.cleanup()

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

    def test_cache_persistence_eviction_and_invalidation(self) -> None:
        from backend.analyzer import SarcasmAnalyzer

        with tempfile.TemporaryDirectory() as tmp:
            cache_file = Path(tmp) / "cache.json"
            cfg = AppConfig(
                host=self.config.host,
                port=self.config.port,
                model_dir=self.config.model_dir,
                onnx_path=self.config.onnx_path,
                tokenizer_path=self.config.tokenizer_path,
                max_length=self.config.max_length,
                positive_threshold=self.config.positive_threshold,
                negative_threshold=self.config.negative_threshold,
                examples=self.config.examples,
                frontend_dir=self.config.frontend_dir,
                cache_enabled=True,
                cache_max_entries=2,
                cache_path=cache_file,
            )
            a1 = SarcasmAnalyzer(cfg)
            r1 = a1.analyze("第一句测试")
            self.assertEqual(set(r1.keys()), {"text", "snownlp", "roberta", "disagree", "elapsed_ms"})
            a1.analyze("第二句测试")
            a1.analyze("第三句测试")
            self.assertTrue(cache_file.is_file())

            # 重建实例应从磁盘读取缓存且仅保留最近 2 条（不触发模型推理）
            a2 = SarcasmAnalyzer(cfg)
            with patch.object(a2, "_classify_roberta", side_effect=AssertionError("不应重新推理")):
                cached_hit = a2.analyze("第三句测试")
                self.assertEqual(cached_hit["text"], "第三句测试")

            # 阈值配置变化时旧缓存应失效
            changed_cfg = AppConfig(
                host=cfg.host,
                port=cfg.port,
                model_dir=cfg.model_dir,
                onnx_path=cfg.onnx_path,
                tokenizer_path=cfg.tokenizer_path,
                max_length=cfg.max_length,
                positive_threshold=0.75,
                negative_threshold=cfg.negative_threshold,
                examples=cfg.examples,
                frontend_dir=cfg.frontend_dir,
                cache_enabled=True,
                cache_max_entries=2,
                cache_path=cache_file,
            )
            a3 = SarcasmAnalyzer(changed_cfg)
            self.assertEqual(a3.cache_size, 0)


if __name__ == "__main__":
    ensure_venv(Path(__file__))
    unittest.main(verbosity=2)
