import json

import requests


# Create a small test image (1x1 pixel) in base64
def create_test_image():
    # This is a 1x1 pixel PNG image in base64
    test_image_b64 = """iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="""
    return test_image_b64


# Test the add_faces endpoint
def test_add_faces():
    url = "http://localhost:5000/api/faces/add_faces"
    image_data = create_test_image()

    payload = {"group_id": "test_group", "image": image_data}

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_add_faces()
