"""
tests/conftest.py
-----------------
Pytest configuration and shared fixtures.
Mocks the model loading so tests are fast and don't require GPU/network.
"""

"""Shared pytest configuration.

Model loading is mocked only by the HTTP tests that need a started app. Model
unit tests intentionally exercise loading and failure reporting.
"""
