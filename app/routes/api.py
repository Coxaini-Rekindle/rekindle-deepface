from flask import Blueprint, jsonify, request

from app.services.face_recognition_service import FaceRecognitionService
from app.utils.image_utils import save_image_from_base64, save_temp_image

# Create a blueprint for our API routes
api_bp = Blueprint("api", __name__, url_prefix="/api")

# Initialize service with configuration
DATA_DIR = "data"
MODELS_DIR = "models"
face_service = FaceRecognitionService(DATA_DIR, MODELS_DIR)


@api_bp.route("/train", methods=["POST"])
def train_model():
    """
    Train a face recognition model for a group.

    Request JSON format:
    {
        "group_id": "unique_group_identifier",
        "faces": [
            {
                "person_id": "unique_person_id1",
                "image": "base64_encoded_image"
            },
            ...
        ]
    }
    """
    try:
        data = request.json

        if not data or "group_id" not in data or "faces" not in data:
            return jsonify({"error": "Missing required parameters"}), 400

        group_id = data["group_id"]
        faces = data["faces"]

        # Create group directory
        group_dir = face_service.get_group_dir(group_id)

        # Process and save faces
        saved_image_paths = []
        for face in faces:
            person_id = face.get("person_id")
            image_data = face.get("image")

            if not person_id or not image_data:
                continue

            # Create directory for this person
            person_dir = face_service.get_person_dir(group_id, person_id)

            # Save the image
            success, result = save_image_from_base64(image_data, person_dir)
            if success:
                saved_image_paths.append(result)
            else:
                print(f"Error processing image for {person_id}: {result}")

        # Train the model with saved images
        success, result = face_service.train_model(saved_image_paths, group_dir)

        if success:
            person_ids = result
            return jsonify(
                {
                    "status": "success",
                    "message": "Model trained successfully",
                    "group_id": group_id,
                    "person_ids": person_ids,
                }
            )
        else:
            return jsonify({"error": result}), 500

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        return jsonify({"error": str(e), "trace": error_trace}), 500


@api_bp.route("/recognize", methods=["POST"])
def recognize_faces():
    """
    Recognize faces in the provided images against a trained model.

    Request JSON format:
    {
        "group_id": "unique_group_identifier",
        "images": [
            "base64_encoded_image1",
            "base64_encoded_image2",
            ...
        ]
    }
    """
    try:
        data = request.json

        if not data or "group_id" not in data or "images" not in data:
            return jsonify({"error": "Missing required parameters"}), 400

        group_id = data["group_id"]
        images = data["images"]

        # Check if model exists
        group_dir = face_service.get_group_dir(group_id)
        import os

        if not os.path.exists(group_dir) or not os.listdir(group_dir):
            return (
                jsonify({"error": f"No trained model found for group_id: {group_id}"}),
                404,
            )

        results = []
        for idx, image_data in enumerate(images):
            # Save temporary image
            temp_img_path = save_temp_image(image_data, DATA_DIR)

            try:
                # Recognize faces in the image
                success, face_results = face_service.recognize_faces(
                    temp_img_path, group_dir
                )

                if success:
                    results.append({"image_index": idx, "faces": face_results})
                else:
                    results.append({"image_index": idx, "error": face_results})

            except Exception as e:
                import traceback

                error_trace = traceback.format_exc()
                results.append(
                    {"image_index": idx, "error": str(e), "trace": error_trace}
                )

            finally:
                # Clean up temp file
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)

        return jsonify({"status": "success", "group_id": group_id, "results": results})

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        return jsonify({"error": str(e), "trace": error_trace}), 500


@api_bp.route("/add_person", methods=["POST"])
def add_person():
    """
    Add a new person to an existing group.

    Request JSON format:
    {
        "group_id": "unique_group_identifier",
        "person_id": "unique_person_id",
        "images": [
            "base64_encoded_image1",
            "base64_encoded_image2",
            ...
        ]
    }
    """
    try:
        data = request.json

        if (
            not data
            or "group_id" not in data
            or "person_id" not in data
            or "images" not in data
        ):
            return jsonify({"error": "Missing required parameters"}), 400

        group_id = data["group_id"]
        person_id = data["person_id"]
        images = data["images"]

        # Create directories
        group_dir = face_service.get_group_dir(group_id)
        person_dir = face_service.get_person_dir(group_id, person_id)

        # Save images
        saved_image_paths = []
        for image_data in images:
            success, result = save_image_from_base64(image_data, person_dir)
            if success:
                saved_image_paths.append(result)
            else:
                print(f"Error processing image: {result}")

        # Add person to the model
        success, result = face_service.add_person(saved_image_paths, group_dir)

        if success:
            return jsonify(
                {
                    "status": "success",
                    "message": "Person added successfully",
                    "group_id": group_id,
                    "person_id": person_id,
                    "saved_images": result,
                }
            )
        else:
            return jsonify({"error": result}), 500

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        return jsonify({"error": str(e), "trace": error_trace}), 500


@api_bp.route("/delete_group", methods=["DELETE"])
def delete_group():
    """Delete a group and all its associated data."""
    try:
        data = request.json

        if not data or "group_id" not in data:
            return jsonify({"error": "Missing group_id parameter"}), 400

        group_id = data["group_id"]

        # Delete the group
        success, message = face_service.delete_group(group_id)

        if success:
            return jsonify({"status": "success", "message": message})
        else:
            return jsonify({"error": message}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500
