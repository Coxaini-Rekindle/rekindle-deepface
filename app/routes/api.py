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


@api_bp.route("/process_faces", methods=["POST"])
def process_faces():
    """
    Extract faces from an image, recognize them, and save to appropriate folders.
    If a face is recognized with high confidence, save it to that person's folder.
    If not recognized, create a new unique ID and save to a new folder.

    Request JSON format:
    {
        "group_id": "unique_group_identifier",
        "image": "base64_encoded_image"
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
            success, results = face_service.extract_and_handle_faces(
                temp_img_path, group_id
            )

            if success:
                return jsonify(
                    {"status": "success", "group_id": group_id, "faces": results}
                )
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

        # Get group directory through storage_manager
        group_dir = face_service.storage_manager.get_group_dir(group_id)
        
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


@api_bp.route("/set_performance", methods=["POST"])
def set_performance():
    """
    Set the performance mode for the face recognition service.

    Request JSON format:
    {
        "mode": "one of: speed, accuracy, balanced, gpu_optimized"
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
