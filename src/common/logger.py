"""공통 로거 팩토리.

콘솔(INFO 이상) + 파일(WARN 이상, 로테이션) 이중 출력.
중복 핸들러 부착을 막아 Jupyter/Colab 재실행 시 로그가 N배 찍히는 문제를 방지한다.
"""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_MAX_BYTES = 5 * 1024 * 1024  # 5MB
_BACKUP_COUNT = 3


def get_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 동일 이름 로거 재요청 시 핸들러를 또 붙이지 않는다.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, f"{name}.log")
    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 루트 로거로의 전파 차단 — 다른 라이브러리 핸들러에 의해 중복 출력되는 것 방지
    logger.propagate = False
    return logger
