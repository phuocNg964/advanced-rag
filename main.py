import asyncio
import os

import uvicorn

from src.core.logger import setup_logging


if __name__ == "__main__":
    if os.name == "nt" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    setup_logging()

    is_dev = os.getenv("ENVIRONMENT", "dev").lower() != "production"

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=is_dev,
        reload_dirs=["src", "static"] if is_dev else None,
    )
