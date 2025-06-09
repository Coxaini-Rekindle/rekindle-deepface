"""Face recognition service implementation."""

import base64
import os
import time
import traceback

import cv2

from app.config.face_recognition_config import FaceRecognitionConfig
from app.core.gpu_manager import GPUManager
from app.detectors.face_detector import FaceDetector
from app.recognizers.face_recognizer import FaceRecognizer
from app.storage.storage_manager import StorageManager
from app.utils.image_processing.processor import ImageProcessor


class FaceRecognitionService:
    """Service for handling face recognition operations."""

    def __init__(self, data_dir, models_dir):
        """
        Initialize the face recognition service with its dependencies.

        Args:
            data_dir (str): Directory for storing face data
            models_dir (str): Directory for storing trained models
        """
        # Initialize configuration
        self.config = FaceRecognitionConfig()

        # Configure GPU if available
        GPUManager.configure_gpu()

        # Initialize components with dependencies
        self.storage_manager = StorageManager(data_dir, models_dir)
        self.image_processor = ImageProcessor()
        self.face_detector = FaceDetector(detector_backend=self.config.detector_backend)
        self.face_recognizer = FaceRecognizer(
            recognition_model=self.config.recognition_model,
            detector_backend=self.config.detector_backend,
            distance_metric=self.config.distance_metric,
            confidence_threshold=self.config.confidence_threshold,
        )

        # Pre-load models for faster inference
        self._preload_models()

    def _preload_models(self):
        """Preload DeepFace models to improve subsequent inference speed."""
        print("Preloading face recognition models...")
        try:
            # Create a small dummy image to warm up the models
            dummy_img = self.image_processor.create_dummy_image()
            dummy_path = os.path.join(self.storage_manager.data_dir, "dummy.jpg")
            cv2.imwrite(dummy_path, dummy_img)

            # Warm up the models
            self.face_recognizer.preload_models(dummy_path)

            # Clean up
            if os.path.exists(dummy_path):
                os.remove(dummy_path)
        except Exception as e:
            print(f"Error preloading models: {str(e)}")

    def extract_and_handle_faces(self, temp_img_path, group_id):
        """
        Extract faces from an image, recognize each face, and save to appropriate folders.
        If a face is recognized with high confidence, save it to that person's folder.
        If not recognized, create a new unique ID and save to a new folder.

        Args:
            temp_img_path (str): Path to the temporary image file
            group_id (str): Group identifier

        Returns:
            tuple: (success, result_data or error_info)
        """
        try:
            start_time = time.time()
            print(f"Starting face extraction for image: {temp_img_path}")

            # Get the group directory
            group_dir = self.storage_manager.get_group_dir(group_id)

            # Ensure the group directory exists
            os.makedirs(group_dir, exist_ok=True)

            # Read image once
            img = cv2.imread(temp_img_path)
            if img is None:
                return False, "Failed to read image"

            # Optimize image size if too large
            img, was_resized, original_shape, new_shape = (
                self.image_processor.optimize_image_size(
                    img, max_dim=self.config.max_image_dimension
                )
            )

            if was_resized:
                # Save the resized image
                cv2.imwrite(temp_img_path, img)
                print(
                    f"Resized image from {original_shape[1]}x{original_shape[0]} to {new_shape[0]}x{new_shape[1]}"
                )

            # Detect faces in the image
            face_objs, face_detection_time = self.face_detector.detect_faces(img)

            if not face_objs:
                return False, "No faces detected in the image"

            # Process each detected face
            results = []
            face_paths = []  # Keep track of face paths for batch processing

            for i, face_obj in enumerate(face_objs):
                try:
                    # Get face region and confidence
                    facial_area = face_obj.get("facial_area", {})
                    if not facial_area:
                        continue

                    # Extract the face from the image
                    detected_face = self.image_processor.extract_face(img, facial_area)

                    # Save face to a temporary file
                    face_path = self.storage_manager.save_temp_face(detected_face)
                    face_paths.append(face_path)

                except Exception as e:
                    # Log error for this face but continue processing other faces
                    print(f"Error processing face {i}: {str(e)}")
                    results.append(
                        {
                            "face_index": i,
                            "error": str(e),
                            "trace": traceback.format_exc(),
                        }
                    )

            # Check if model exists before attempting recognition
            has_trained_model = self.storage_manager.group_exists(group_id)

            # Batch process faces for recognition if trained model exists
            if has_trained_model and face_paths:
                # Process all faces
                batch_time_start = time.time()

                for i, face_path in enumerate(face_paths):
                    # Recognize faces in the image
                    success, face_result, _ = self.face_recognizer.recognize_face(
                        face_path, group_dir
                    )

                    if success and face_result and len(face_result) > 0:
                        face_info = face_result[0]  # Get the first result

                        if (
                            not face_info.get("is_new_person", True)
                            and face_info.get("confidence", 0)
                            >= self.config.confidence_threshold
                        ):
                            # Face recognized with high confidence
                            person_id = face_info.get("person_id")
                            face_img = cv2.imread(face_path)
                            new_face_path = self.storage_manager.save_face_image(
                                face_img, group_id, person_id
                            )  # Add to results
                            results.append(
                                {
                                    "face_index": i,
                                    "person_id": person_id,
                                    "is_new_person": False,
                                    "confidence": face_info.get("confidence", 0),
                                    "saved_to": new_face_path,
                                }
                            )
                        else:
                            # Face not recognized with high confidence - create new person
                            new_person_id = (
                                self.storage_manager.create_permanent_user_id()
                            )
                            face_img = cv2.imread(face_path)
                            new_face_path = self.storage_manager.save_face_image(
                                face_img, group_id, new_person_id
                            )

                            # Add to results
                            results.append(
                                {
                                    "face_index": i,
                                    "person_id": new_person_id,
                                    "is_new_person": True,
                                    "saved_to": new_face_path,
                                }
                            )

                print(
                    f"Batch face recognition completed in {time.time() - batch_time_start:.2f} seconds"
                )
            else:  # No existing model - create new person for each face
                for i, face_path in enumerate(face_paths):
                    new_person_id = self.storage_manager.create_permanent_user_id()
                    face_img = cv2.imread(face_path)
                    new_face_path = self.storage_manager.save_face_image(
                        face_img, group_id, new_person_id
                    )

                    # Add to results
                    results.append(
                        {
                            "face_index": i,
                            "person_id": new_person_id,
                            "is_new_person": True,
                            "saved_to": new_face_path,
                        }
                    )

            # Clean up temporary face files
            self.storage_manager.cleanup_temp_files(
                face_paths
            )  # Update the model after processing all faces
            model_update_time = time.time()
            if results:
                # Use first successful face path as sample for retraining
                sample_paths = []
                for result in results:
                    if "saved_to" in result:
                        sample_paths.append(result["saved_to"])
                if sample_paths:
                    # Update the model
                    self.face_recognizer.update_model(sample_paths[0], group_dir)
                    model_update_time = time.time() - model_update_time
                    print(f"Model updated in {model_update_time:.2f} seconds")

            total_time = time.time() - start_time
            print(f"Total processing time: {total_time:.2f} seconds")

            # Add timing information to results
            timing_info = {
                "total_processing_time": total_time,
                "face_detection_time": face_detection_time,
                "model_update_time": (
                    time.time() - model_update_time if results and sample_paths else 0
                ),
            }

            return True, {"results": results, "timing": timing_info}

        except Exception as e:
            error_trace = traceback.format_exc()
            return False, {"error": str(e), "trace": error_trace}

    def recognize_faces(self, temp_img_path, group_id):
        """
        Recognize faces in an image using a trained model.

        Args:
            temp_img_path (str): Path to the temporary image file
            group_id (str): Group identifier

        Returns:
            tuple: (success, face_results or error_info)
        """
        try:
            # Get the group directory
            group_dir = self.storage_manager.get_group_dir(group_id)

            # Check if the group exists
            if not self.storage_manager.group_exists(group_id):
                return False, f"No trained model found for group_id: {group_id}"

            # Recognize the face
            success, face_results, recognition_time = (
                self.face_recognizer.recognize_face(temp_img_path, group_dir)
            )

            if success:
                return True, face_results
            else:
                return False, face_results

        except Exception as e:
            error_trace = traceback.format_exc()
            return False, {"error": str(e), "trace": error_trace}

    def delete_group(self, group_id):
        """
        Delete a group and all its associated data.

        Args:
            group_id (str): The ID of the group to delete

        Returns:
            tuple: (success, message)
        """
        return self.storage_manager.delete_group(group_id)

    def set_performance_mode(self, mode="balanced"):
        """
        Configure performance settings based on predefined modes.

        Args:
            mode (str): Performance mode - one of:
                - 'speed': Prioritize speed over accuracy
                - 'accuracy': Prioritize accuracy over speed
                - 'balanced': Balance between speed and accuracy
                - 'gpu_optimized': Settings optimized for GPU processing

        Returns:
            dict: The current performance settings
        """
        # Update configuration
        settings = self.config.set_performance_mode(mode)

        # Update components with new settings
        self.face_detector = FaceDetector(detector_backend=self.config.detector_backend)
        self.face_recognizer = FaceRecognizer(
            recognition_model=self.config.recognition_model,
            detector_backend=self.config.detector_backend,
            distance_metric=self.config.distance_metric,
            confidence_threshold=self.config.confidence_threshold,
        )

        # Preload models again with the new settings if using gpu_optimized mode
        if mode == "gpu_optimized":
            self._preload_models()

        return settings

    def add_faces_to_group(self, temp_img_path, group_id):
        """
        Add faces from an image to a group. Try to recognize existing faces,
        create temp users for unrecognized faces.

        Args:
            temp_img_path (str): Path to the temporary image file
            group_id (str): Group identifier

        Returns:
            tuple: (success, result_data or error_info)
        """
        try:
            start_time = time.time()
            print(f"Starting face addition for image: {temp_img_path}")

            # Get the group directory
            group_dir = self.storage_manager.get_group_dir(group_id)

            # Ensure the group directory exists
            os.makedirs(group_dir, exist_ok=True)

            # Read image once
            img = cv2.imread(temp_img_path)
            if img is None:
                return False, "Failed to read image"

            # Optimize image size if too large
            img, was_resized, original_shape, new_shape = (
                self.image_processor.optimize_image_size(
                    img, max_dim=self.config.max_image_dimension
                )
            )

            if was_resized:
                # Save the resized image
                cv2.imwrite(temp_img_path, img)
                print(
                    f"Resized image from {original_shape[1]}x{original_shape[0]} to {new_shape[0]}x{new_shape[1]}"
                )

            # Detect faces in the image
            face_objs, face_detection_time = self.face_detector.detect_faces(img)

            if not face_objs:
                return False, "No faces detected in the image"

            # Process each detected face
            results = []
            face_paths = []  # Keep track of face paths for batch processing

            for i, face_obj in enumerate(face_objs):
                try:
                    # Get face region and confidence
                    facial_area = face_obj.get("facial_area", {})
                    if not facial_area:
                        continue

                    # Extract the face from the image
                    detected_face = self.image_processor.extract_face(img, facial_area)

                    # Save face to a temporary file
                    face_path = self.storage_manager.save_temp_face(detected_face)
                    face_paths.append((i, face_path, detected_face))

                except Exception as e:
                    # Log error for this face but continue processing other faces
                    print(f"Error processing face {i}: {str(e)}")
                    results.append(
                        {
                            "face_index": i,
                            "error": str(e),
                            "trace": traceback.format_exc(),
                        }
                    )

            # Check if model exists before attempting recognition
            has_trained_model = self.storage_manager.group_exists(group_id)

            # Process faces for recognition if trained model exists
            if has_trained_model and face_paths:
                batch_time_start = time.time()

                for i, face_path, detected_face in face_paths:
                    # Recognize faces in the image
                    success, face_result, _ = self.face_recognizer.recognize_face(
                        face_path, group_dir
                    )

                    person_id = None
                    is_new_person = True
                    confidence = 0.0
                    recognition_type = "unknown"

                    if success and face_result and len(face_result) > 0:
                        face_info = face_result[0]  # Get the first result

                        if (
                            not face_info.get("is_new_person", True)
                            and face_info.get("confidence", 0)
                            >= self.config.confidence_threshold
                        ):
                            # Face recognized with high confidence
                            person_id = face_info.get("person_id")
                            is_new_person = False
                            confidence = face_info.get("confidence", 0)

                            # Check if the matched person is already a temporary user
                            if self.storage_manager.is_temp_user(person_id):
                                recognition_type = "temp_user"
                            else:
                                recognition_type = "recognized"
                        else:
                            # Face not recognized with high confidence - create temp user
                            person_id = self.storage_manager.create_temp_user_id()
                            is_new_person = True
                            confidence = face_info.get("confidence", 0)
                            recognition_type = "temp_user"
                    else:
                        # No recognition results - create temp user
                        person_id = self.storage_manager.create_temp_user_id()
                        is_new_person = True
                        recognition_type = "temp_user"

                    # Save face image
                    new_face_path = self.storage_manager.save_face_image(
                        detected_face, group_id, person_id
                    )  # Save metadata for the user
                    from datetime import datetime

                    metadata = {
                        "created_at": datetime.now().isoformat(),
                        "recognition_type": recognition_type,
                        "confidence": confidence,
                        "source_image": os.path.basename(temp_img_path),
                    }

                    # Set is_temp_user flag based on recognition type
                    if recognition_type == "temp_user":
                        metadata["is_temp_user"] = True
                    else:
                        metadata["is_temp_user"] = False

                    self.storage_manager.save_user_metadata(
                        group_id, person_id, metadata
                    )

                    # Convert face to base64 for response
                    face_base64 = self._convert_face_to_base64(detected_face)

                    # Add to results
                    results.append(
                        {
                            "face_index": i,
                            "person_id": person_id,
                            "is_temp_user": self.storage_manager.is_temp_user(
                                person_id
                            ),
                            "is_new_person": is_new_person,
                            "confidence": confidence,
                            "recognition_type": recognition_type,
                            "saved_to": new_face_path,
                            "face_image_base64": face_base64,
                        }
                    )

                print(
                    f"Batch face processing completed in {time.time() - batch_time_start:.2f} seconds"
                )
            else:  # No existing model - create temp user for each face
                for i, face_path, detected_face in face_paths:
                    person_id = self.storage_manager.create_temp_user_id()
                    new_face_path = self.storage_manager.save_face_image(
                        detected_face, group_id, person_id
                    )

                    # Save metadata for the user
                    from datetime import datetime

                    metadata = {
                        "created_at": datetime.now().isoformat(),
                        "recognition_type": "temp_user",
                        "confidence": 0.0,
                        "source_image": os.path.basename(temp_img_path),
                        "is_temp_user": True,
                    }
                    self.storage_manager.save_user_metadata(
                        group_id, person_id, metadata
                    )

                    # Convert face to base64 for response
                    face_base64 = self._convert_face_to_base64(detected_face)

                    # Add to results
                    results.append(
                        {
                            "face_index": i,
                            "person_id": person_id,
                            "is_temp_user": True,
                            "is_new_person": True,
                            "confidence": 0.0,
                            "recognition_type": "temp_user",
                            "saved_to": new_face_path,
                            "face_image_base64": face_base64,
                        }
                    )

            # Clean up temporary face files
            temp_paths = [path for _, path, _ in face_paths]
            self.storage_manager.cleanup_temp_files(temp_paths)

            total_time = time.time() - start_time
            print(f"Total processing time: {total_time:.2f} seconds")

            # Add timing information to results
            timing_info = {
                "total_processing_time": total_time,
                "face_detection_time": face_detection_time,
            }

            return True, {"faces": results, "timing": timing_info}

        except Exception as e:
            error_trace = traceback.format_exc()
            return False, {"error": str(e), "trace": error_trace}

    def merge_users(self, group_id, source_person_ids, target_person_id):
        """
        Merge multiple users into a target user.

        Args:
            group_id (str): Group identifier
            source_person_ids (list): List of source person IDs to merge
            target_person_id (str): Target person ID to merge into

        Returns:
            tuple: (success, message, merged_info)
        """
        return self.storage_manager.merge_users(
            group_id, source_person_ids, target_person_id
        )

    def list_users_in_group(self, group_id):
        """
        List all users in a group.

        Args:
            group_id (str): Group identifier

        Returns:
            dict: Dictionary with user information including summary
        """
        users_data = self.storage_manager.list_users_in_group(group_id)

        # Calculate summary
        permanent_count = len(users_data.get("permanent", []))
        temporary_count = len(users_data.get("temporary", []))
        total_count = permanent_count + temporary_count

        summary = {
            "total_users": total_count,
            "permanent_users": permanent_count,
            "temporary_users": temporary_count,
        }

        return {"users": users_data, "summary": summary}

    def get_last_user_image(self, group_id, person_id):
        """
        Get the latest (last) image for a user.

        Args:
            group_id (str): Group identifier
            person_id (str): Person identifier

        Returns:
            tuple: (success, result_data or error_message)
                result_data contains: {"image_base64": str, "filename": str, "created_at": str}
        """
        return self.storage_manager.get_last_user_image(group_id, person_id)

    def numpy_to_base64(self, img_array):
        """
        Convert a NumPy array image to a base64 encoded string.

        Args:
            img_array (numpy.ndarray): The image as a NumPy array

        Returns:
            str: The base64 encoded string of the image
        """
        try:
            # Encode the image as a JPEG file
            _, buffer = cv2.imencode(".jpg", img_array)

            # Convert to base64
            img_base64 = base64.b64encode(buffer).decode("utf-8")

            return img_base64
        except Exception as e:
            print(f"Error converting image to base64: {str(e)}")
            return None

    def _convert_face_to_base64(self, face_array):
        """
        Convert a face numpy array to base64 encoded string.

        Args:
            face_array (numpy.ndarray): Face image as numpy array

        Returns:
            str: Base64 encoded face image
        """
        try:
            # Encode face array to JPEG format
            success, buffer = cv2.imencode(".jpg", face_array)
            if not success:
                return None

            # Convert to base64
            face_base64 = base64.b64encode(buffer).decode("utf-8")
            return face_base64
        except Exception as e:
            print(f"Error converting face to base64: {str(e)}")
            return None
