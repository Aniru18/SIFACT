import sys
import os

# Add ROOT to path so graph/, tools/, agents/, utils/, config/ are all importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.main import app  # noqa: E402