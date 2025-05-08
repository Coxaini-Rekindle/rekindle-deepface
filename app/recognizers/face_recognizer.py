"""Face recognition functionality."""

import os
import time

from deepface import DeepFace


class FaceRecognizer:
    """Handles face recognition operations."""

    def __init__(
        self,
        recognition_model="VGG-Face",
        detector_backend="retinaface",
        distance_metric="cosine",
        confidence_threshold=0.5,
    ):
        """
        Initialize the face recognizer.

        Args:
            recognition_model (str): Model to use for face recognition
            detector_backend (str): Backend to use for face detection
            distance_metric (str): Distance metric for face comparison
            confidence_threshold (float): Confidence threshold for recognition
        """
        self.recognition_model = recognition_model
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        self.confidence_threshold = confidence_threshold

    def preload_models(self, dummy_path):
        """
        Preload face recognition models to improve inference speed.

        Args:
            dummy_path (str): Path to a dummy image for model warming

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Warm up the face detection model
            _ = DeepFace.extract_faces(
                img_path=dummy_path,
                detector_backend=self.detector_backend,
                enforce_detection=False,
            )

            # Warm up the face recognition model
            _ = DeepFace.represent(
                img_path=dummy_path,
                model_name=self.recognition_model,
                detector_backend=self.detector_backend,
                enforce_detection=False,
            )

            print("Face recognition models preloaded successfully")
            return True
        except Exception as e:
            print(f"Error preloading models: {str(e)}")
            return False

    def recognize_face(self, face_img_path, db_path):
        """
        Recognize a face against a database of known faces.

        Args:
            face_img_path (str): Path to the face image
            db_path (str): Path to the database of known faces

        Returns:
            tuple: (success, face_results, recognition_time)
        """
        try:
            start_time = time.time()

            # Find faces in the image with optimized parameters
            matches = DeepFace.find(
                img_path=face_img_path,
                db_path=db_path,
                model_name=self.recognition_model,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=False,
                silent=True,
            )

            recognition_time = time.time() - start_time
            print(f"Face recognition completed in {recognition_time:.2f} seconds")

            # Process matches
            face_results = self._process_matches(
                matches, face_img_path, recognition_time
            )

            return True, face_results, recognition_time

        except Exception as e:
            return False, {"error": str(e)}, 0

    def _process_matches(self, matches, img_path, recognition_time):
        """
        Process and format recognition matches.

        Args:
            matches: DeepFace match result
            img_path (str): Path to the face image
            recognition_time (float): Time taken for recognition

        Returns:
            list: Processed face results
        """
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
                    if f"{self.recognition_model}_{self.distance_metric}" in best_match:
                        distance_column = (
                            f"{self.recognition_model}_{self.distance_metric}"
                        )
                        distance_value = best_match[distance_column]
                    elif "distance" in best_match:
                        distance_column = "distance"
                        distance_value = best_match["distance"]

                    if distance_value is not None:
                        # Determine confidence based on distance (lower distance = higher confidence)
                        confidence = (
                            1 - distance_value
                        )  # Convert distance to confidence score

                        if (
                            confidence >= self.confidence_threshold
                        ):  # Threshold for recognition
                            face_results.append(
                                {
                                    "person_id": person_id,
                                    "confidence": float(confidence),
                                    "is_new_person": False,
                                    "distance_metric": distance_column,
                                    "distance_value": float(distance_value),
                                    "recognition_time_seconds": recognition_time,
                                }
                            )
                        else:
                            # Low confidence, treat as new person
                            face_results.append(
                                {
                                    "person_id": f"unknown_{os.path.basename(img_path)}",
                                    "confidence": float(confidence),
                                    "is_new_person": True,
                                    "distance_metric": distance_column,
                                    "distance_value": float(distance_value),
                                    "closest_match": person_id,
                                    "recognition_time_seconds": recognition_time,
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
                                "recognition_time_seconds": recognition_time,
                            }
                        )
                else:
                    # No matches found
                    face_results.append(
                        {
                            "person_id": f"unknown_{os.path.basename(img_path)}",
                            "confidence": 0.0,
                            "is_new_person": True,
                            "error": "No matches found in database",
                            "recognition_time_seconds": recognition_time,
                        }
                    )

        if not face_results:  # No faces detected or processed
            face_results.append(
                {
                    "person_id": f"unknown_{os.path.basename(img_path)}",
                    "confidence": 0.0,
                    "is_new_person": True,
                    "error": "No face detected or no matches found in database",
                    "recognition_time_seconds": recognition_time,
                }
            )

        return face_results

    def update_model(self, sample_path, db_path):
        """
        Update the face recognition model with a new sample.

        Args:
            sample_path (str): Path to a sample face image
            db_path (str): Path to the database to update

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            start_time = time.time()

            # Update the database with the new face
            _ = DeepFace.find(
                img_path=sample_path,
                db_path=db_path,
                model_name=self.recognition_model,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True,
                normalization="base",
                silent=True,
            )

            update_time = time.time() - start_time
            print(f"Model updated in {update_time:.2f} seconds")
            return True
        except Exception as e:
            print(f"Error updating model: {str(e)}")
            return False
