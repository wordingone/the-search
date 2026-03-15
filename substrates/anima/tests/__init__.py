"""
Anima Tests Module
==================

Test suites for Anima and variants.
"""

from .unit_tests import run_all_tests
from .variant_comparison import run_variant_comparison

__all__ = ['run_all_tests', 'run_variant_comparison']
