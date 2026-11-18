"""纯离线情感对比推理模块：SnowNLP + INT8 量化 ONNX 模型。"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import onnxruntime as ort
from snownlp import SnowNLP
from tokenizers import Tokenizer

from backend.config import AppConfig


class SarcasmAnalyzer:
    """封装 SnowNLP 与 ONNX RoBERTa 双模型推理。"""

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
            snow_res = self._classify_snownlp(cleaned)
            roberta_res = self._classify_roberta(cleaned)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        return {
            "text": cleaned,
            "snownlp": snow_res,
            "roberta": roberta_res,
            "disagree": snow_res["label"] != roberta_res["label"],
            "elapsed_ms": elapsed_ms,
        }
