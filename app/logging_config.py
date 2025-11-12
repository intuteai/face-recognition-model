# app/logging_config.py
import logging, sys, os, time
from contextlib import contextmanager

class KVFormatter(logging.Formatter):
    def format(self, record):
        base = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # include extra structured keys if present
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            base.update(record.extra)
        return " | ".join(f"{k}={v}" for k, v in base.items())

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(KVFormatter())

def setup_logging():
    """Initialize root logging with env LOG_LEVEL (default INFO)."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.root.handlers.clear()
    logging.root.setLevel(level)
    logging.root.addHandler(_handler)
    # quiet noisy libs a bit
    for noisy in ["uvicorn", "uvicorn.error"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

@contextmanager
def timed(log: logging.Logger, event: str, **fields):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt_ms = int((time.perf_counter() - t0) * 1000)
        log.info(event, extra={**fields, "ms": dt_ms})
