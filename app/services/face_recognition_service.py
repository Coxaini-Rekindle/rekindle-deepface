import os
import traceback

from deepface import DeepFace


class FaceRecognitionService:
    """Service for handling face recognition operations."""

    def __init__(self, data_dir, models_dir):
        """
        Initialize the face recognition service.

        Args:
            data_dir (str): Directory for storing face data
            models_dir (str): Directory for storing trained models
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.group_models = {}  # Dictionary to map group_ids to model paths

        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

    def get_group_dir(self, group_id):
        """Get the directory path for a group."""
        return os.path.join(self.data_dir, group_id)

    def get_person_dir(self, group_id, person_id):
        """Get the directory path for a person within a group."""
        return os.path.join(self.get_group_dir(group_id), person_id)

    def train_model(self, saved_image_paths, group_dir):
        """
        Train a face recognition model using the saved images.

        Args:
            saved_image_paths (list): List of paths to saved face images
            group_dir (str): Directory containing the group data

        Returns:
            tuple: (success, result_data or error_info)
        """
        if not saved_image_paths:
            return False, "No valid images were processed"

        # We need at least one valid image path for DeepFace.find to work properly
        sample_img_path = saved_image_paths[0]

        # Verify the path exists
        if not os.path.exists(sample_img_path):
            path_check = {
                "saved_paths": saved_image_paths,
                "dir_exists": os.path.exists(os.path.dirname(sample_img_path)),
                "absolute_path": os.path.abspath(sample_img_path),
            }
            return False, {
                "error": f"Sample image path does not exist: {sample_img_path}",
                "path_check": path_check,
            }

        try:
            # Create representations using DeepFace
            _ = DeepFace.find(
                img_path=sample_img_path,  # Use one of the saved images as a reference
                db_path=group_dir,
                model_name="VGG-Face",
                detector_backend="retinaface",
                enforce_detection=False,
                align=True,
                normalization="base",
                silent=True,
            )

            # Get all person_ids from the group directory
            person_ids = []
            for directory in os.listdir(group_dir):
                dir_path = os.path.join(group_dir, directory)
                if os.path.isdir(dir_path):
                    person_ids.append(directory)

            return True, person_ids

        except Exception as e:
            error_trace = traceback.format_exc()
            return False, {
                "error": str(e),
                "trace": error_trace,
                "details": "Error during DeepFace model training",
                "group_dir": group_dir,
                "sample_img": sample_img_path,
            }

    def recognize_faces(self, temp_img_path, group_dir):
        """
        Recognize faces in an image using a trained model.

        Args:
            temp_img_path (str): Path to the temporary image file
            group_dir (str): Directory containing the group data

        Returns:
            tuple: (success, face_results or error_info)
        """
        try:
            # Find faces in the image
            matches = DeepFace.find(
                img_path=temp_img_path,
                db_path=group_dir,
                model_name="VGG-Face",
                detector_backend="retinaface",
                distance_metric="cosine",
                enforce_detection=False,
                silent=True,
            )

            # Process matches
            face_results = []
            if matches and len(matches) > 0:
                for i, match_df in enumerate(matches):
                    if not match_df.empty and len(match_df) > 0:
                        # Get the best match (first row)
                        best_match = match_df.iloc[0]

                        # Extract person_id from identity path
                        identity_path = best_match["identity"]
                        person_id = os.path.basename(os.path.dirname(identity_path))

                        # Check available distance columns - DeepFace versions may vary
                        distance_value = None
                        distance_column = None

                        # Try both column name formats
                        if "VGG-Face_cosine" in best_match:
                            distance_column = "VGG-Face_cosine"
                            distance_value = best_match["VGG-Face_cosine"]
                        elif "distance" in best_match:
                            distance_column = "distance"
                            distance_value = best_match["distance"]

                        if distance_value is not None:
                            # Determine confidence based on distance (lower distance = higher confidence)
                            confidence = (
                                1 - distance_value
                            )  # Convert distance to confidence score

                            if confidence >= 0.6:  # Threshold for recognition
                                face_results.append(
                                    {
                                        "person_id": person_id,
                                        "confidence": float(confidence),
                                        "is_new_person": False,
                                        "distance_metric": distance_column,
                                        "distance_value": float(distance_value),
                                    }
                                )
                            else:
                                # Low confidence, treat as new person
                                face_results.append(
                                    {
                                        "person_id": f"unknown_{os.path.basename(temp_img_path)}",
                                        "confidence": float(confidence),
                                        "is_new_person": True,
                                        "distance_metric": distance_column,
                                        "distance_value": float(distance_value),
                                        "closest_match": person_id,
                                    }
                                )
                        else:
                            # Missing distance column
                            columns = list(best_match.keys())
                            face_results.append(
                                {
                                    "person_id": person_id,  # Still use the matched person but mark as uncertain
                                    "confidence": 0.0,
                                    "is_new_person": False,
                                    "uncertain": True,
                                    "error": f"Distance metric column not found. Available columns: {columns}",
                                }
                            )
                    else:
                        # No matches found
                        face_results.append(
                            {
                                "person_id": f"unknown_{os.path.basename(temp_img_path)}",
                                "confidence": 0.0,
                                "is_new_person": True,
                                "error": "No matches found in database",
                            }
                        )

            if not face_results:  # No faces detected or processed
                face_results.append(
                    {
                        "person_id": f"unknown_{os.path.basename(temp_img_path)}",
                        "confidence": 0.0,
                        "is_new_person": True,
                        "error": "No face detected or no matches found in database",
                    }
                )

            return True, face_results

        except Exception as e:
            error_trace = traceback.format_exc()
            return False, {"error": str(e), "trace": error_trace}

    def add_person(self, saved_image_paths, group_dir):
        """
        Add a new person to an existing group model.

        Args:
            saved_image_paths (list): List of paths to saved face images
            group_dir (str): Directory containing the group data

        Returns:
            tuple: (success, result_data or error_info)
        """
        if not saved_image_paths:
            return False, "No valid images were processed"

        # Use one of the saved images as reference for DeepFace.find
        sample_img_path = saved_image_paths[0]

        try:
            # Retrain model to include the new person
            DeepFace.find(
                img_path=sample_img_path,
                db_path=group_dir,
                model_name="VGG-Face",
                detector_backend="retinaface",
                enforce_detection=False,
                align=True,
                normalization="base",
                silent=True,
            )

            return True, [os.path.basename(path) for path in saved_image_paths]

        except Exception as e:
            error_trace = traceback.format_exc()
            return False, {
                "error": str(e),
                "trace": error_trace,
                "details": "Error during DeepFace model update",
                "group_dir": group_dir,
                "sample_img": sample_img_path,
            }

    def delete_group(self, group_id):
        """
        Delete a group and all its associated data.

        Args:
            group_id (str): The ID of the group to delete

        Returns:
            tuple: (success, message)
        """
        group_dir = self.get_group_dir(group_id)

        # Check if group exists
        if not os.path.exists(group_dir):
            return False, f"Group {group_id} not found"

        try:
            # Delete the group directory
            import shutil

            shutil.rmtree(group_dir)

            # Remove from models dictionary
            if group_id in self.group_models:
                del self.group_models[group_id]

            return True, f"Group {group_id} deleted successfully"

        except Exception as e:
            return False, str(e)
