"""配置加载、跨平台路径管理与虚拟环境检查模块。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.json"


def ensure_venv(script_path: Path) -> None:
    """确保脚本运行在虚拟环境中；若未激活且存在 .venv 则自动切换拉起。"""
    if sys.prefix != sys.base_prefix:
        return
    if os.environ.get("ZH_SARCASM_VENV_CHECKED") == "1":
        return

    venv_dir = ROOT_DIR / ".venv"
    venv_python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not venv_python.is_file():
        print("错误：未进入 Python 虚拟环境，且未在项目根目录找到 .venv。")
        print("请先在项目根目录创建虚拟环境并安装依赖：")
        if os.name == "nt":
            print("  python -m venv .venv")
            print("  .venv\\Scripts\\pip install -r requirements.txt")
        else:
            print("  python3 -m venv .venv")
            print("  .venv/bin/pip install -r requirements.txt")
        sys.exit(1)

    env = os.environ.copy()
    env["ZH_SARCASM_VENV_CHECKED"] = "1"
    result = subprocess.run(
        [str(venv_python), str(script_path.resolve()), *sys.argv[1:]],
        env=env,
        check=False,
    )
    sys.exit(result.returncode)


@dataclass(frozen=True)
class AppConfig:
    """从根目录 config.json 解析出的运行配置。"""

    host: str
    port: int
    model_dir: Path
    onnx_path: Path
    tokenizer_path: Path
    max_length: int
    positive_threshold: float
    negative_threshold: float
    examples: list[dict[str, str]]
    frontend_dir: Path
    cache_enabled: bool = True
    cache_max_entries: int = 100
    cache_path: Path = ROOT_DIR / "backend" / "cache" / "results.json"


def load_config(config_path: Path = CONFIG_PATH) -> AppConfig:
    """读取并校验全局配置文件。"""
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到配置文件：{config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = json.load(file)

    server_cfg = raw["server"]
    model_cfg = raw["model"]
    snow_cfg = raw["snownlp"]
    cache_cfg = raw.get("cache", {})

    model_dir = ROOT_DIR / model_cfg["dir"]
    cache_file = str(cache_cfg.get("file", "backend/cache/results.json"))
    return AppConfig(
        host=str(server_cfg["host"]),
        port=int(server_cfg["port"]),
        model_dir=model_dir,
        onnx_path=model_dir / model_cfg["onnx_file"],
        tokenizer_path=model_dir / model_cfg["tokenizer_file"],
        max_length=int(model_cfg["max_length"]),
        positive_threshold=float(snow_cfg["positive_threshold"]),
        negative_threshold=float(snow_cfg["negative_threshold"]),
        examples=[
            {"tag": str(item["tag"]), "text": str(item["text"])}
            for item in raw.get("examples", [])
        ],
        frontend_dir=ROOT_DIR / "frontend",
        cache_enabled=bool(cache_cfg.get("enabled", True)),
        cache_max_entries=max(0, int(cache_cfg.get("max_entries", 100))),
        cache_path=ROOT_DIR / cache_file,
    )
