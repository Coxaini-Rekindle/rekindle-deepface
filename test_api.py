import argparse
import base64
import json
import os

import requests

# Base URL for the API
BASE_URL = "http://localhost:5000"


def encode_image(image_path):
    """Convert an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def test_healthcheck():
    """Test the health check endpoint."""
    response = requests.get(f"{BASE_URL}/healthcheck")
    print("Health Check Response:", response.json())
    return response.status_code == 200


def test_train_model(group_id, person_image_map):
    """
    Test training a model with sample faces.

    Args:
        group_id (str): Group identifier
        person_image_map (dict): Dict mapping person_ids to list of image paths
    """
    faces = []

    for person_id, image_paths in person_image_map.items():
        for image_path in image_paths:
            faces.append({"person_id": person_id, "image": encode_image(image_path)})

    data = {"group_id": group_id, "faces": faces}

    response = requests.post(f"{BASE_URL}/api/train", json=data)

    print("Train Model Response:", json.dumps(response.json(), indent=2))
    return response.status_code == 200


def test_recognize_faces(group_id, image_paths):
    """
    Test recognizing faces in images.

    Args:
        group_id (str): Group identifier
        image_paths (list): List of image paths to recognize
    """
    images = [encode_image(path) for path in image_paths]

    data = {"group_id": group_id, "images": images}

    response = requests.post(f"{BASE_URL}/api/recognize", json=data)

    print("Recognize Faces Response:", json.dumps(response.json(), indent=2))
    return response.status_code == 200


def test_add_person(group_id, person_id, image_paths):
    """
    Test adding a new person to an existing group.

    Args:
        group_id (str): Group identifier
        person_id (str): Person identifier
        image_paths (list): List of image paths for the person
    """
    images = [encode_image(path) for path in image_paths]

    data = {"group_id": group_id, "person_id": person_id, "images": images}

    response = requests.post(f"{BASE_URL}/api/add_person", json=data)

    print("Add Person Response:", json.dumps(response.json(), indent=2))
    return response.status_code == 200


def test_delete_group(group_id):
    """
    Test deleting a group.

    Args:
        group_id (str): Group identifier to delete
    """
    data = {"group_id": group_id}

    response = requests.delete(f"{BASE_URL}/api/delete_group", json=data)

    print("Delete Group Response:", json.dumps(response.json(), indent=2))
    return response.status_code == 200


def main():
    parser = argparse.ArgumentParser(description="Test Face Recognition API")

    parser.add_argument(
        "--action",
        choices=["healthcheck", "train", "recognize", "add", "delete", "all"],
        required=True,
        help="Action to perform",
    )

    parser.add_argument("--group_id", help="Group ID for the operation")
    parser.add_argument("--person_id", help="Person ID for add operation")
    parser.add_argument("--image_dir", help="Directory containing test images")
    parser.add_argument(
        "--image", action="append", help="Specific image path(s) to use"
    )

    args = parser.parse_args()

    if args.action == "healthcheck" or args.action == "all":
        test_healthcheck()

    # For train, recognize, add, and delete, we need group_id
    if (
        args.action in ["train", "recognize", "add", "delete", "all"]
        and not args.group_id
    ):
        print("Error: group_id is required for this action")
        return

    if args.action == "train" or (args.action == "all" and args.image_dir):
        # For training, organize images by subdirectory name (person_id)
        if not args.image_dir:
            print("Error: image_dir is required for training")
            return

        person_image_map = {}
        for person_id in os.listdir(args.image_dir):
            person_dir = os.path.join(args.image_dir, person_id)
            if os.path.isdir(person_dir):
                person_image_map[person_id] = []
                for img_file in os.listdir(person_dir):
                    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        person_image_map[person_id].append(
                            os.path.join(person_dir, img_file)
                        )

        test_train_model(args.group_id, person_image_map)

    if args.action == "recognize" or (args.action == "all" and args.image):
        # For recognition, use the specified images
        if not args.image:
            print("Error: at least one --image is required for recognition")
            return

        test_recognize_faces(args.group_id, args.image)

    if args.action == "add" or (args.action == "all" and args.person_id and args.image):
        # For adding a person, need person_id and images
        if not args.person_id:
            print("Error: person_id is required for adding a person")
            return

        if not args.image:
            print("Error: at least one --image is required for adding a person")
            return

        test_add_person(args.group_id, args.person_id, args.image)

    if args.action == "delete" or args.action == "all":
        # For deleting a group, just need group_id
        test_delete_group(args.group_id)


if __name__ == "__main__":
    main()
