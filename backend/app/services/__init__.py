"""
MarineMine Pro — Application Services Package
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path so services can cleanly import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

