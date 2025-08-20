"""
Deployment package for Phase 4.
Contains FastAPI app, CLI tools, and monitoring.
"""

from .api import app
from .cli import cli_predict

__all__ = [
    'app',
    'cli_predict'
]