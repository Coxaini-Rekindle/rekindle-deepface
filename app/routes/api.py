"""
Face Recognition API Routes

This module provides REST API endpoints for face recognition operations with support for
temporary user management and merging workflows.

API Workflow:
1. Use POST /api/add_faces to add faces to a group
   - Recognized faces are associated with existing users
   - Unrecognized faces get temporary user IDs (prefixed with "temp_")
2. Use GET /api/groups/{group_id}/users to list all users (permanent and temporary)
3. Use POST /api/merge_users to merge temporary users with existing ones
4. Use POST /api/recognize to recognize faces in new images
5. Use DELETE /api/delete_group to remove groups and all associated data
6. Use POST /api/set_performance to adjust performance settings

All endpoints accept and return JSON data.
"""

from flask import Blueprint, jsonify, request

from app.services.face_recognition_service import FaceRecognitionService
from app.utils.image_utils import save_temp_image

# Create a blueprint for our API routes
api_bp = Blueprint("api", __name__, url_prefix="/api")

# Initialize service with configuration
DATA_DIR = "data"
MODELS_DIR = "models"
face_service = FaceRecognitionService(DATA_DIR, MODELS_DIR)

# Set the GPU-optimized mode by default
face_service.set_performance_mode("gpu_optimized")


@api_bp.route("/add_faces", methods=["POST"])
def add_faces():
    """
    Add faces from an image to a group. Try to recognize existing faces,
    create temporary users for unrecognized faces.

    Request JSON format:
    {
        "group_id": "unique_group_identifier",
        "image": "base64_encoded_image"
    }    Response JSON format (success):
    {
        "status": "success",
        "group_id": "example_group",
        "faces": [
            {
                "face_index": 0,
                "person_id": "person_12345678-1234-1234-1234-123456789abc",
                "is_temp_user": false,
                "is_new_person": false,
                "confidence": 0.85,
                "recognition_type": "recognized",
                "saved_to": "/path/to/saved/image.jpg",
                "face_image_base64": "/9j/4AAQSkZJRgABAQEAYABgAAD..."
            },
            {
                "face_index": 1,
                "person_id": "temp_98765432-4321-4321-4321-cba987654321",
                "is_temp_user": true,
                "is_new_person": true,
                "confidence": 0.0,
                "recognition_type": "temp_user",
                "saved_to": "/path/to/saved/temp_image.jpg",
                "face_image_base64": "/9j/4AAQSkZJRgABAQEAYABgAAD..."
            }
        ],
        "timing": {
            "total_processing_time": 2.45,
            "face_detection_time": 0.32
        }
    }

    Response JSON format (error):
    {
        "error": "Error message describing what went wrong",
        "trace": "Optional stack trace for debugging"
    }
    """
    try:
        data = request.json

        if not data or "group_id" not in data or "image" not in data:
            return jsonify({"error": "Missing required parameters"}), 400

        group_id = data["group_id"]
        image_data = data["image"]

        # Save temporary image
        temp_img_path = save_temp_image(image_data, DATA_DIR)

        try:
            # Process faces in the image
            success, results = face_service.add_faces_to_group(temp_img_path, group_id)

            if success:
                return jsonify({"status": "success", "group_id": group_id, **results})
            else:
                return jsonify({"error": results}), 500

        except Exception as e:
            import traceback

            error_trace = traceback.format_exc()
            return jsonify({"error": str(e), "trace": error_trace}), 500

        finally:
            # Clean up temp file
            import os

            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        return jsonify({"error": str(e), "trace": error_trace}), 500


@api_bp.route("/merge_users", methods=["POST"])
def merge_users():
    """
    Merge multiple users into a target user.

    Request JSON format:
    {
        "group_id": "unique_group_identifier",
        "source_person_ids": ["temp_12345", "temp_67890"],
        "target_person_id": "person_abcdef123456"
    }

    Response JSON format (success):
    {
        "status": "success",
        "group_id": "example_group",
        "merged_count": 2,
        "target_person_id": "person_abcdef123456",
        "source_person_ids": ["temp_12345", "temp_67890"],
        "message": "Successfully merged 2 users into person_abcdef123456"
    }

    Response JSON format (error):
    {
        "error": "Error message describing what went wrong"
    }
    """
    try:
        data = request.json

        if (
            not data
            or "group_id" not in data
            or "source_person_ids" not in data
            or "target_person_id" not in data
        ):
            return (
                jsonify(
                    {
                        "error": "Missing required parameters: group_id, source_person_ids, target_person_id"
                    }
                ),
                400,
            )

        group_id = data["group_id"]
        source_person_ids = data["source_person_ids"]
        target_person_id = data["target_person_id"]

        # Validate inputs
        if not isinstance(source_person_ids, list) or len(source_person_ids) == 0:
            return jsonify({"error": "source_person_ids must be a non-empty list"}), 400

        # Merge users
        success, message, merged_info = face_service.merge_users(
            group_id, source_person_ids, target_person_id
        )

        if success:
            return jsonify(
                {
                    "status": "success",
                    "group_id": group_id,
                    "merged_count": len(source_person_ids),
                    "target_person_id": target_person_id,
                    "source_person_ids": source_person_ids,
                    "message": message,
                    **merged_info,
                }
            )
        else:
            return jsonify({"error": message}), 400

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        return jsonify({"error": str(e), "trace": error_trace}), 500


@api_bp.route("/groups/<group_id>/users", methods=["GET"])
def list_users_in_group(group_id):
    """
    List all users in a group.

    Response JSON format (success):
    {
        "status": "success",
        "group_id": "example_group",
        "users": {
            "permanent": [
                {
                    "person_id": "person_12345678-1234-1234-1234-123456789abc",
                    "face_count": 5,
                    "metadata": {
                        "created_at": "2025-06-08T10:30:00",
                        "recognition_type": "recognized",
                        "confidence": 0.85
                    }
                }
            ],
            "temporary": [
                {
                    "person_id": "temp_98765432-4321-4321-4321-cba987654321",
                    "face_count": 2,
                    "metadata": {
                        "created_at": "2025-06-08T11:15:00",
                        "recognition_type": "temp_user",
                        "confidence": 0.0
                    }
                }
            ]
        },
        "summary": {
            "total_users": 2,
            "permanent_users": 1,
            "temporary_users": 1
        }
    }

    Response JSON format (error):
    {
        "error": "Error message describing what went wrong"
    }
    """
    try:
        # Check if group exists
        if not face_service.storage_manager.group_exists(group_id):
            return jsonify({"error": f"Group '{group_id}' does not exist"}), 404

        # Get users in group
        users_info = face_service.list_users_in_group(group_id)

        return jsonify({"status": "success", "group_id": group_id, **users_info})

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        return jsonify({"error": str(e), "trace": error_trace}), 500


@api_bp.route("/groups/<group_id>/users/<person_id>/last_image", methods=["GET"])
def get_last_user_image(group_id, person_id):
    """
    Get the latest (last) image for a specific user.

    Response JSON format (success):
    {
        "status": "success",
        "group_id": "example_group",
        "person_id": "person_12345678-1234-1234-1234-123456789abc",
        "image": {
            "image_base64": "base64_encoded_image_data",
            "filename": "abc123def456.jpg",
            "created_at": "2025-06-08T15:30:45.123456",
            "file_size": 25648
        }
    }

    Response JSON format (error):
    {
        "error": "Error message describing what went wrong"
    }
    """
    try:
        # Check if group exists
        if not face_service.storage_manager.group_exists(group_id):
            return jsonify({"error": f"Group '{group_id}' does not exist"}), 404

        # Get the last image for the user
        success, result = face_service.get_last_user_image(group_id, person_id)

        if success:
            return jsonify(
                {
                    "status": "success",
                    "group_id": group_id,
                    "person_id": person_id,
                    "image": result,
                }
            )
        else:
            return jsonify({"error": result}), 404

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
            "base64_encoded_image2"
        ]
    }

    Response JSON format (success):
    {
        "status": "success",
        "group_id": "example_group",
        "results": [
            {
                "image_index": 0,
                "faces": [
                    {
                        "person_id": "person_12345678-1234-1234-1234-123456789abc",
                        "confidence": 0.85,
                        "facial_area": {
                            "x": 100,
                            "y": 50,
                            "w": 200,
                            "h": 250
                        }
                    }
                ]
            },
            {
                "image_index": 1,
                "error": "No faces detected in the image"
            }
        ]
    }

    Response JSON format (error):
    {
        "error": "Error message describing what went wrong"    }
    """
    try:
        data = request.json

        if not data or "group_id" not in data or "images" not in data:
            return jsonify({"error": "Missing required parameters"}), 400

        group_id = data["group_id"]
        images = data["images"]

        # Check if model exists
        if not face_service.storage_manager.group_exists(group_id):
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
                    temp_img_path, group_id
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
                import os

                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)

        return jsonify({"status": "success", "group_id": group_id, "results": results})

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        return jsonify({"error": str(e), "trace": error_trace}), 500


@api_bp.route("/delete_group", methods=["DELETE"])
def delete_group():
    """
    Delete a group and all its associated data.

    Request JSON format:
    {
        "group_id": "unique_group_identifier"
    }

    Response JSON format (success):
    {
        "status": "success",
        "message": "Group 'example_group' and all associated data deleted successfully"
    }

    Response JSON format (error):
    {
        "error": "Error message describing what went wrong"
    }
    """
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


@api_bp.route("/set_performance", methods=["POST"])
def set_performance():
    """
    Set the performance mode for the face recognition service.

    Request JSON format:
    {
        "mode": "balanced"
    }

    Valid modes:
    - "speed": Prioritize speed over accuracy
    - "accuracy": Prioritize accuracy over speed
    - "balanced": Balance between speed and accuracy
    - "gpu_optimized": Settings optimized for GPU processing

    Response JSON format (success):
    {
        "status": "success",
        "message": "Performance mode set to: balanced",
        "settings": {
            "detector_backend": "retinaface",
            "recognition_model": "VGG-Face",
            "distance_metric": "cosine",
            "confidence_threshold": 0.7
        }
    }

    Response JSON format (error):
    {
        "error": "Error message describing what went wrong"
    }
    """
    try:
        data = request.json

        if not data or "mode" not in data:
            return jsonify({"error": "Missing mode parameter"}), 400

        mode = data["mode"]
        valid_modes = ["speed", "accuracy", "balanced", "gpu_optimized"]

        if mode not in valid_modes:
            return (
                jsonify(
                    {"error": f"Invalid mode. Must be one of: {', '.join(valid_modes)}"}
                ),
                400,
            )

        # Apply the new performance mode
        settings = face_service.set_performance_mode(mode)

        return jsonify(
            {
                "status": "success",
                "message": f"Performance mode set to: {mode}",
                "settings": settings,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
