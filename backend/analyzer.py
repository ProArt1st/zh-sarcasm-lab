"""纯离线情感对比推理模块：SnowNLP + INT8 量化 ONNX 模型（带本地结果缓存）。"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Any

import numpy as np
import onnxruntime as ort
from snownlp import SnowNLP
from tokenizers import Tokenizer

from backend.config import AppConfig


class SarcasmAnalyzer:
    """封装 SnowNLP 与 ONNX RoBERTa 双模型推理，并透明缓存最近分析结果。"""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._validate_model_files()

        self._tokenizer = Tokenizer.from_file(str(config.tokenizer_path))
        self._tokenizer.enable_truncation(max_length=config.max_length)
        self._session = ort.InferenceSession(
            str(config.onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {item.name for item in self._session.get_inputs()}

        self._signature = self._build_signature()
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        if self._cache_active():
            self._load_cache()

    @property
    def cache_size(self) -> int:
        """返回当前内存中缓存的条数。"""
        with self._lock:
            return len(self._cache)

    def _cache_active(self) -> bool:
        return self._config.cache_enabled and self._config.cache_max_entries > 0

    def _build_signature(self) -> str:
        stat = self._config.onnx_path.stat()
        return (
            f"{self._config.onnx_path.name}|{stat.st_size}|{stat.st_mtime_ns}|"
            f"{self._config.max_length}|{self._config.positive_threshold}|"
            f"{self._config.negative_threshold}"
        )

    def _load_cache(self) -> None:
        cache_path = self._config.cache_path
        if not cache_path.is_file():
            return
        try:
            with cache_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return

        if not isinstance(data, dict) or data.get("signature") != self._signature:
            return
        items = data.get("items")
        if not isinstance(items, dict):
            return

        for key, val in items.items():
            if isinstance(key, str) and self._is_valid_entry(val):
                self._cache[key] = val
                if len(self._cache) >= self._config.cache_max_entries:
                    break

    @staticmethod
    def _is_valid_entry(entry: Any) -> bool:
        return (
            isinstance(entry, dict)
            and isinstance(entry.get("snownlp"), dict)
            and isinstance(entry.get("roberta"), dict)
            and isinstance(entry.get("disagree"), bool)
        )

    def _save_cache(self) -> None:
        cache_path = self._config.cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp")
        payload = {"signature": self._signature, "items": dict(self._cache)}
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        tmp_path.replace(cache_path)

    def _validate_model_files(self) -> None:
        if not self._config.onnx_path.is_file():
            raise FileNotFoundError(
                f"未找到离线模型文件：{self._config.onnx_path}\n"
                "请从仓库 Releases 下载 model_int8.onnx 并放入 model/ 目录。"
            )
        if not self._config.tokenizer_path.is_file():
            raise FileNotFoundError(
                f"未找到分词器文件：{self._config.tokenizer_path}\n"
                "请检查仓库 model/ 目录是否完整。"
            )

    def _classify_snownlp(self, text: str) -> dict[str, Any]:
        score = float(SnowNLP(text).sentiments)
        if score >= self._config.positive_threshold:
            label = "正面"
        elif score <= self._config.negative_threshold:
            label = "负面"
        else:
            label = "中性"
        return {
            "label": label,
            "positive_prob": round(score, 3),
        }

    def _classify_roberta(self, text: str) -> dict[str, Any]:
        encoded = self._tokenizer.encode(text)
        feed: dict[str, np.ndarray] = {
            "input_ids": np.array([encoded.ids], dtype=np.int64),
            "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array([encoded.type_ids], dtype=np.int64)

        raw_logits = self._session.run(["logits"], feed)[0]
        if not isinstance(raw_logits, np.ndarray):
            raise TypeError(f"模型输出类型异常：{type(raw_logits)}")
        logits = raw_logits[0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # id2label: 0 -> Negative, 1 -> Positive
        neg_prob = float(probs[0])
        pos_prob = float(probs[1])
        if pos_prob >= neg_prob:
            label = "正面"
            confidence = pos_prob
        else:
            label = "负面"
            confidence = neg_prob

        return {
            "label": label,
            "confidence": round(confidence, 3),
            "positive_prob": round(pos_prob, 3),
            "negative_prob": round(neg_prob, 3),
        }

    def analyze(self, text: str) -> dict[str, Any]:
        """对单条中文文本同时运行两个模型并返回对比结果。"""
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("请输入待检测的中文文本")

        start = time.perf_counter()
        with self._lock:
            if self._cache_active() and cleaned in self._cache:
                self._cache.move_to_end(cleaned)
                entry = self._cache[cleaned]
                elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
                return {
                    "text": cleaned,
                    "snownlp": dict(entry["snownlp"]),
                    "roberta": dict(entry["roberta"]),
                    "disagree": bool(entry["disagree"]),
                    "elapsed_ms": elapsed_ms,
                }

            snow_res = self._classify_snownlp(cleaned)
            roberta_res = self._classify_roberta(cleaned)
            disagree = snow_res["label"] != roberta_res["label"]

            if self._cache_active():
                self._cache[cleaned] = {
                    "snownlp": dict(snow_res),
                    "roberta": dict(roberta_res),
                    "disagree": disagree,
                }
                while len(self._cache) > self._config.cache_max_entries:
                    self._cache.popitem(last=False)
                self._save_cache()

        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "text": cleaned,
            "snownlp": snow_res,
            "roberta": roberta_res,
            "disagree": disagree,
            "elapsed_ms": elapsed_ms,
        }
