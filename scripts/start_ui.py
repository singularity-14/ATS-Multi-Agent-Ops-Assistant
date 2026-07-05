#!/usr/bin/env python3
"""Start the Streamlit UI."""
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    ui_path = Path(__file__).parent.parent / 'src' / 'ui' / 'app.py'
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', str(ui_path),
        '--server.port=8501',
        '--server.address=0.0.0.0',
        '--theme.base=dark'
    ])
