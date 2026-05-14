# To run this application, execute `python main.py` in your terminal.
# Do not run `uvicorn main:app --reload` directly.
import uvicorn
from src.core.logger import setup_logging

import os

if __name__ == "__main__":
    setup_logging()
    # Disable reload in production to save memory and CPU
    is_dev = os.getenv("ENVIRONMENT", "dev").lower() != "production"
    
    uvicorn.run(
        "src.api.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=is_dev, 
        reload_dirs=["src", "static"] if is_dev else None
    )
