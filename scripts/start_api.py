#!/usr/bin/env python3
"""Start the FastAPI server."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from src.config import get_settings

if __name__ == '__main__':
    settings = get_settings()
    print(f'Starting ATS Multi-Agent Ops Assistant API on port {settings.fastapi_port}...')
    uvicorn.run('src.api.main:app', host='0.0.0.0', port=settings.fastapi_port, reload=True)
