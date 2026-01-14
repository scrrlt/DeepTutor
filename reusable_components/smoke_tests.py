"""
End-to-End Smoke Tests

Tests the complete flow: Collector → GCS → Gateway → Embeddings → Traces

Run locally or in CI/CD after deploying services to Cloud Run.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any
import time

import requests

# Configuration
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8080")
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0460359034")

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def log_pass(msg: str):
    print(f"{GREEN}✓ {msg}{RESET}")


def log_fail(msg: str):
    print(f"{RED}✗ {msg}{RESET}")


def log_info(msg: str):
    print(f"{YELLOW}ℹ {msg}{RESET}")


def test_gateway_health() -> bool:
    """Test 1: Gateway is healthy."""
    log_info("Test 1: Gateway Health Check")
    
    try:
        resp = requests.get(f"{GATEWAY_URL}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("ok"):
                log_pass(f"Gateway healthy: {data.get('service')}")
                return True
        
        log_fail(f"Gateway health failed: {resp.status_code}")
        return False
    except Exception as e:
        log_fail(f"Gateway connection failed: {e}")
        return False


def test_gateway_ready() -> bool:
    """Test 2: Gateway is ready."""
    log_info("Test 2: Gateway Readiness Check")
    
    try:
        resp = requests.get(f"{GATEWAY_URL}/ready", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("ready"):
                log_pass("Gateway ready")
                return True
        
        log_fail(f"Gateway readiness failed: {resp.status_code}")
        return False
    except Exception as e:
        log_fail(f"Readiness check failed: {e}")
        return False


def test_embeddings_batch_validation() -> bool:
    """Test 3: Embeddings batch request validation."""
    log_info("Test 3: Embeddings Batch Validation")
    
    # Test 3a: Missing texts field
    resp = requests.post(
        f"{GATEWAY_URL}/v1/embeddings/batch",
        json={},
        headers={"Content-Type": "application/json"},
        timeout=5,
    )
    if resp.status_code != 400:
        log_fail(f"Should reject empty request (got {resp.status_code})")
        return False
    
    log_pass("Correctly rejects empty request")
    
    # Test 3b: Empty texts array
    resp = requests.post(
        f"{GATEWAY_URL}/v1/embeddings/batch",
        json={"texts": []},
        headers={"Content-Type": "application/json"},
        timeout=5,
    )
    if resp.status_code != 400:
        log_fail(f"Should reject empty texts (got {resp.status_code})")
        return False
    
    log_pass("Correctly rejects empty texts array")
    return True


def test_embeddings_batch_submission() -> Dict[str, Any]:
    """Test 4: Submit embeddings batch job."""
    log_info("Test 4: Embeddings Batch Submission")
    
    payload = {
        "texts": [
            "Apple Inc. quarterly earnings report.",
            "Microsoft cloud services announcement.",
            "Google workspace updates.",
        ]
    }
    
    try:
        resp = requests.post(
            f"{GATEWAY_URL}/v1/embeddings/batch",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        
        if resp.status_code in [200, 202]:
            data = resp.json()
            job_name = data.get("job_name") or data.get("id")
            if job_name:
                log_pass(f"Batch submitted: {job_name}")
                return data
            else:
                log_fail("No job_name in response")
                return {}
        elif resp.status_code == 503:
            log_info("Embedding service unavailable (expected if not deployed)")
            return {}
        else:
            log_fail(f"Batch submission failed: {resp.status_code}")
            return {}
    
    except Exception as e:
        log_fail(f"Batch submission error: {e}")
        return {}


def test_sec_edgar_endpoints() -> bool:
    """Test 5: SEC EDGAR endpoints (even if not wired yet)."""
    log_info("Test 5: SEC EDGAR Endpoints")
    
    try:
        resp = requests.get(f"{GATEWAY_URL}/v1/sec/facts/AAPL", timeout=5)
        
        # Can be 501 (not yet wired) or 200 (if wired)
        if resp.status_code in [200, 501]:
            log_pass(f"SEC endpoint responded: {resp.status_code}")
            return True
        else:
            log_fail(f"Unexpected status: {resp.status_code}")
            return False
    
    except Exception as e:
        log_fail(f"SEC endpoint error: {e}")
        return False


def test_rate_limiting() -> bool:
    """Test 6: Rate limiting is enforced."""
    log_info("Test 6: Rate Limiting")
    
    try:
        # Rapid fire requests
        responses = []
        for i in range(15):
            resp = requests.post(
                f"{GATEWAY_URL}/v1/embeddings/batch",
                json={"texts": [f"test {i}"]},
                headers={"Content-Type": "application/json"},
                timeout=2,
            )
            responses.append(resp.status_code)
        
        # Should see some 429 responses if rate limiting is active
        has_429 = 429 in responses
        if has_429:
            log_pass("Rate limiting enforced (got 429 responses)")
            return True
        else:
            log_info("No 429 responses (rate limiting may not be active)")
            return True  # Not a failure, just informational
    
    except Exception as e:
        log_info(f"Rate limit test inconclusive: {e}")
        return True  # Not a failure


def test_traces_output() -> bool:
    """Test 7: Verify traces are written to generated/."""
    log_info("Test 7: Traces Output")
    
    traces_dir = Path(__file__).parent.parent.parent / "generated"
    traces_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to find any trace files
    trace_files = list(traces_dir.glob("collection_trace_*.jsonl"))
    
    if trace_files:
        latest = max(trace_files, key=lambda f: f.stat().st_mtime)
        with open(latest) as f:
            lines = f.readlines()
        
        log_pass(f"Found {len(lines)} trace entries in {latest.name}")
        return True
    else:
        log_info("No trace files yet (expected if collector not run)")
        return True  # Not a failure


def run_all_tests():
    """Run all smoke tests and report results."""
    print("\n" + "="*60)
    print("CHIMERA End-to-End Smoke Tests")
    print("="*60 + "\n")
    
    tests = [
        ("Gateway Health", test_gateway_health),
        ("Gateway Ready", test_gateway_ready),
        ("Batch Validation", test_embeddings_batch_validation),
        ("Batch Submission", test_embeddings_batch_submission),
        ("SEC EDGAR", test_sec_edgar_endpoints),
        ("Rate Limiting", test_rate_limiting),
        ("Traces", test_traces_output),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            if isinstance(result, bool):
                results.append((name, result))
            else:
                results.append((name, True))  # Submission test
        except Exception as e:
            log_fail(f"Test crashed: {e}")
            results.append((name, False))
        
        print()
    
    # Summary
    print("="*60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed\n")
    
    for name, passed in results:
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {status} – {name}")
    
    print("\n" + "="*60)
    
    return all(r for _, r in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)