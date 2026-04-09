#!/usr/bin/env python3
"""
Test script for Document Layout Detection API
Verifies the API is working correctly with various test cases
"""

import requests
import json
from PIL import Image, ImageDraw
import time
import sys

# Configuration
API_URL = "http://localhost:8000/api/v1"
HEALTH_ENDPOINT = f"{API_URL}/health"
PREDICT_ENDPOINT = f"{API_URL}/predict"

def test_health_check():
    """Test 1: Check if API is online and model is loaded"""
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            
            if data.get("model_loaded"):
                print("✅ PASS: Model is loaded and ready")
                return True
            else:
                print("❌ FAIL: Model is not loaded")
                print(f"Error: {data.get('model_info', {}).get('error')}")
                return False
        else:
            print(f"❌ FAIL: Unexpected status code {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ FAIL: Cannot connect to API. Is the server running on port 8000?")
        print("   Start with: conda activate doc-layout && python -m uvicorn app.main:app --port 8000")
        return False
    except Exception as e:
        print(f"❌ FAIL: {type(e).__name__}: {e}")
        return False


def create_test_image_simple():
    """Create a simple white test image"""
    print("\nCreating simple white test image...")
    img = Image.new('RGB', (200, 200), color='white')
    path = '/tmp/test_simple.png'
    img.save(path)
    print(f"✓ Created: {path}")
    return path


def create_test_image_with_objects():
    """Create test image with rectangles (simulating text blocks)"""
    print("\nCreating test image with objects...")
    img = Image.new('RGB', (400, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw rectangles of different sizes
    colors = ['red', 'blue', 'green', 'black']
    positions = [
        (20, 20, 150, 80),      # small
        (170, 20, 380, 80),     # small
        (20, 100, 380, 200),    # large - should be detected
        (20, 220, 150, 300),    # small
        (170, 220, 380, 350),   # medium
    ]
    
    for i, (x1, y1, x2, y2) in enumerate(positions):
        draw.rectangle([x1, y1, x2, y2], fill=colors[i % len(colors)], width=2)
        draw.text((x1+5, y1+5), f"Block {i+1}", fill='white')
    
    path = '/tmp/test_objects.png'
    img.save(path)
    print(f"✓ Created: {path}")
    return path


def test_prediction(image_path, test_name):
    """Test prediction with an image"""
    print(f"\n--- Predicting on {test_name} ---")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.split('/')[-1], f, 'image/png')}
            start = time.time()
            response = requests.post(PREDICT_ENDPOINT, files=files, timeout=30)
            elapsed = time.time() - start
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Inference time: {data['inference_time_ms']:.2f}ms")
            print(f"Image size: {data['image_size']['width']}x{data['image_size']['height']}")
            print(f"Blocks detected: {data['num_blocks']}")
            
            if data['num_blocks'] > 0:
                print(f"✅ PASS: Detected {data['num_blocks']} blocks")
                print("Sample blocks:")
                for i, block in enumerate(data['blocks'][:3]):
                    print(f"  Block {i+1}: {block['label']} (confidence: {block['score']:.2f})")
                    print(f"    BBox: ({block['bbox']['x1']}, {block['bbox']['y1']}) → "
                          f"({block['bbox']['x2']}, {block['bbox']['y2']})")
            else:
                print("⚠️  WARNING: No blocks detected (might be normal for simple images)")
            return True
        
        elif response.status_code == 400:
            print(f"❌ FAIL: Invalid request")
            print(f"Detail: {response.json().get('detail', 'Unknown error')}")
            return False
        
        elif response.status_code == 413:
            print(f"❌ FAIL: File too large")
            print(f"Detail: {response.json().get('detail', 'Unknown error')}")
            return False
        
        elif response.status_code == 500:
            print(f"❌ FAIL: Server error")
            print(f"Detail: {response.json().get('detail', 'Unknown error')}")
            print("Check server logs for more information")
            return False
        
        else:
            print(f"❌ FAIL: Unexpected status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
    
    except requests.exceptions.Timeout:
        print(f"❌ FAIL: Request timed out (inference took > 30 seconds)")
        return False
    except requests.exceptions.ConnectionError:
        print(f"❌ FAIL: Cannot connect to API")
        return False
    except Exception as e:
        print(f"❌ FAIL: {type(e).__name__}: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("DOCUMENT LAYOUT DETECTION API - TEST SUITE")
    print("="*70)
    
    results = []
    
    # Test 1: Health check
    health_ok = test_health_check()
    results.append(("Health Check", health_ok))
    
    if not health_ok:
        print("\n" + "="*70)
        print("⚠️  API is not ready. Cannot proceed with prediction tests.")
        print("="*70)
        return 1
    
    # Test 2: Simple white image
    print("\n" + "="*70)
    print("TEST 2: Prediction on Simple Image")
    print("="*70)
    img_simple = create_test_image_simple()
    pred_simple = test_prediction(img_simple, "simple white image")
    results.append(("Simple Image Prediction", pred_simple))
    
    # Test 3: Image with objects
    print("\n" + "="*70)
    print("TEST 3: Prediction on Image with Objects")
    print("="*70)
    img_objects = create_test_image_with_objects()
    pred_objects = test_prediction(img_objects, "image with colored rectangles")
    results.append(("Objects Image Prediction", pred_objects))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! API is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
