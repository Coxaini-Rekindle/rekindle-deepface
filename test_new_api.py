#!/usr/bin/env python3
"""
Test script for the new Face Recognition API endpoints.
"""

import base64
import os

import requests

# Configuration
API_BASE_URL = "http://localhost:5000/api"
TEST_GROUP_ID = "test_group_redesign"


def encode_image_to_base64(image_path):
    """Convert an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def test_add_faces():
    """Test the new add_faces endpoint."""
    print("🔍 Testing POST /api/add_faces...")

    # Create a simple test image if none exists
    test_image_path = "test_image.jpg"
    if not os.path.exists(test_image_path):
        print(
            "ℹ️  No test image found. Please add a test_image.jpg file to test the API."
        )
        return False

    image_base64 = encode_image_to_base64(test_image_path)

    payload = {"group_id": TEST_GROUP_ID, "image": image_base64}

    try:
        response = requests.post(f"{API_BASE_URL}/add_faces", json=payload)
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success! Found {len(result.get('faces', []))} faces")
            for face in result.get("faces", []):
                temp_status = "🟡 TEMP" if face.get("is_temp_user") else "🟢 PERMANENT"
                print(
                    f"      - Face {face['face_index']}: {face['person_id']} ({temp_status})"
                )
            return result
        else:
            print(f"   ❌ Error: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(
            "   ❌ Connection error. Make sure the Flask app is running on localhost:5000"
        )
        return False
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False


def test_list_users():
    """Test the list users endpoint."""
    print(f"👥 Testing GET /api/groups/{TEST_GROUP_ID}/users...")

    try:
        response = requests.get(f"{API_BASE_URL}/groups/{TEST_GROUP_ID}/users")
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            users = result.get("users", {})
            summary = result.get("summary", {})

            print(f"   ✅ Success! Found {summary.get('total_users', 0)} total users")
            print(f"      - Permanent: {summary.get('permanent_users', 0)}")
            print(f"      - Temporary: {summary.get('temporary_users', 0)}")

            # Show some user details
            for user_type in ["permanent", "temporary"]:
                user_list = users.get(user_type, [])
                for user in user_list[:3]:  # Show first 3 users
                    print(
                        f"      - {user_type.title()}: {user['person_id']} ({user['face_count']} faces)"
                    )

            return result
        else:
            print(f"   ❌ Error: {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False


def test_merge_users(users_data):
    """Test the merge users endpoint."""
    print("🔄 Testing POST /api/merge_users...")

    if not users_data:
        print("   ⚠️  Skipping merge test - no users data available")
        return False

    users = users_data.get("users", {})
    temp_users = users.get("temporary", [])
    permanent_users = users.get("permanent", [])

    if len(temp_users) == 0:
        print("   ⚠️  No temporary users found to merge")
        return False

    if len(permanent_users) == 0:
        print("   ⚠️  No permanent users found to merge into")
        return False

    # Try to merge first temp user into first permanent user
    source_id = temp_users[0]["person_id"]
    target_id = permanent_users[0]["person_id"]

    payload = {
        "group_id": TEST_GROUP_ID,
        "source_person_ids": [source_id],
        "target_person_id": target_id,
    }

    try:
        response = requests.post(f"{API_BASE_URL}/merge_users", json=payload)
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success! Merged {result.get('merged_count', 0)} users")
            print(f"      - Source: {source_id}")
            print(f"      - Target: {target_id}")
            return result
        else:
            print(f"   ❌ Error: {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False


def test_recognize_faces():
    """Test the recognize faces endpoint."""
    print("🔍 Testing POST /api/recognize...")

    test_image_path = "test_image.jpg"
    if not os.path.exists(test_image_path):
        print("   ⚠️  No test image found for recognition test")
        return False

    image_base64 = encode_image_to_base64(test_image_path)

    payload = {"group_id": TEST_GROUP_ID, "images": [image_base64]}

    try:
        response = requests.post(f"{API_BASE_URL}/recognize", json=payload)
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            results = result.get("results", [])
            print(f"   ✅ Success! Processed {len(results)} images")

            for img_result in results:
                if "faces" in img_result:
                    faces = img_result["faces"]
                    print(
                        f"      - Image {img_result['image_index']}: {len(faces)} faces recognized"
                    )
                else:
                    print(
                        f"      - Image {img_result['image_index']}: {img_result.get('error', 'Unknown error')}"
                    )

            return result
        else:
            print(f"   ❌ Error: {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False


def test_performance_setting():
    """Test the set performance endpoint."""
    print("⚙️  Testing POST /api/set_performance...")

    payload = {"mode": "balanced"}

    try:
        response = requests.post(f"{API_BASE_URL}/set_performance", json=payload)
        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(
                f"   ✅ Success! Performance mode set to: {result.get('message', 'unknown')}"
            )
            return result
        else:
            print(f"   ❌ Error: {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False


def main():
    """Run all API tests."""
    print("🚀 Starting Face Recognition API Tests")
    print("=" * 50)

    # Test 1: Add faces
    add_result = test_add_faces()
    print()

    # Test 2: List users
    users_result = test_list_users()
    print()

    # Test 3: Merge users (if data available)
    merge_result = test_merge_users(users_result)
    print()

    # Test 4: Recognize faces
    recognize_result = test_recognize_faces()
    print()

    # Test 5: Set performance
    performance_result = test_performance_setting()
    print()

    print("=" * 50)
    print("🏁 API Tests Complete")

    # Summary
    tests_passed = sum(
        [
            bool(add_result),
            bool(users_result),
            bool(merge_result),
            bool(recognize_result),
            bool(performance_result),
        ]
    )

    print(f"✅ {tests_passed}/5 tests passed")

    if tests_passed == 5:
        print("🎉 All API endpoints are working correctly!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
