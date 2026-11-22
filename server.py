"""
MarineMine Pro — Server Entrypoint
Imports and exposes the FastAPI app from backend.app.main
"""

import sys
from pathlib import Path

# Ensure root and backend directory are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"

for p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
