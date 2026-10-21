"""
Make ``import nawah`` work under pytest without an install step.

The repository root goes on ``sys.path`` because the package lives beside
the tests rather than inside a ``src/`` tree. Doing it explicitly means the
suite runs identically from a checkout, from a zip, and from CI, whether
or not the user has run ``pip install -e .`` first.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
