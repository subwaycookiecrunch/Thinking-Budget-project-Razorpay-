"""
tests/test_server_api.py
========================
Integration tests for the OpenEnv HTTP server endpoints in server/app.py.
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient
from server.app import app


class TestServerAPI(unittest.TestCase):
    """Test FastAPI application endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")

    def test_schema_endpoint(self):
        res = self.client.get("/schema")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsInstance(data, dict)

    def test_openapi_endpoint(self):
        res = self.client.get("/openapi.json")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("openapi", "").split(".")[0], "3")

    def test_docs_endpoint(self):
        res = self.client.get("/docs")
        self.assertEqual(res.status_code, 200)
        self.assertIn("swagger", res.text.lower())

    def test_404_not_found(self):
        res = self.client.get("/non_existent_endpoint")
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
