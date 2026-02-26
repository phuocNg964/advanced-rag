# To run this application, execute `python main.py` in your terminal.
# Do not run `uvicorn main:app --reload` directly.
import uvicorn
from src.logging_config import setup_logging

if __name__ == "__main__":
    setup_logging()
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=["src", "static"])
