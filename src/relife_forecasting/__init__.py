import os

import uvicorn


def get_worker_count() -> int:
    raw_value = os.getenv("API_WORKERS", "1")

    try:
        return max(1, int(raw_value))
    except ValueError as exc:
        raise ValueError("API_WORKERS must be a positive integer.") from exc


def main() -> None:
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 9090))
    workers = get_worker_count()
    # Dynamically determine the module path
    module_name = __name__.split(".")[0]
    uvicorn.run(f"{module_name}.main:app", host=host, port=port, workers=workers)
