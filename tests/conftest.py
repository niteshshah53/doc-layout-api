"""
tests/conftest.py
-----------------
Pytest configuration and shared fixtures.
Mocks the model loading so tests are fast and don't require GPU/network.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_model_for_tests():
    """
    Autouse fixture that mocks model loading for every test.
    
    This prevents:
    - Downloading real Detectron2 models from the internet
    - Attempting to load CUDA
    - YAML parsing errors from corrupted downloads
    """
    with patch("app.model.load_model") as mock_load:
        mock_load.return_value = MagicMock()
        yield


