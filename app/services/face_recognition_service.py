import os
import shutil
import traceback
import uuid

import cv2
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

        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

    def get_group_dir(self, group_id):
        """Get the directory path for a group."""
        return os.path.join(self.data_dir, group_id)

    def get_person_dir(self, group_id, person_id):
        """Get the directory path for a person within a group."""
        return os.path.join(self.get_group_dir(group_id), person_id)

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
            # Get the group directory
            group_dir = self.get_group_dir(group_id)

            # Ensure the group directory exists
            os.makedirs(group_dir, exist_ok=True)

            # Detect faces in the image using DeepFace
            face_objs = DeepFace.extract_faces(
                img_path=temp_img_path,
                detector_backend="retinaface",
                enforce_detection=False,
                align=True,
            )

            if not face_objs:
                return False, "No faces detected in the image"

            # Process each detected face
            results = []
            for i, face_obj in enumerate(face_objs):
                try:
                    # Get face region and confidence
                    facial_area = face_obj.get("facial_area", {})
                    if not facial_area:
                        continue

                    # Extract face coordinates
                    x = facial_area.get("x", 0)
                    y = facial_area.get("y", 0)
                    w = facial_area.get("w", 0)
                    h = facial_area.get("h", 0)

                    # Extract the face from the image
                    img = cv2.imread(temp_img_path)
                    detected_face = img[y : y + h, x : x + w]

                    # Save face to a temporary file
                    face_filename = f"face_{i}_{uuid.uuid4()}.jpg"
                    face_path = os.path.join(self.data_dir, face_filename)
                    cv2.imwrite(face_path, detected_face)

                    # Try to recognize this face if the group has trained data
                    if os.path.exists(group_dir) and os.listdir(group_dir):
                        success, face_result = self.recognize_faces(
                            face_path, group_dir
                        )

                        if success and face_result and len(face_result) > 0:
                            face_info = face_result[0]  # Get the first result

                            if (
                                not face_info.get("is_new_person", True)
                                and face_info.get("confidence", 0) >= 0.5
                            ):
                                # Face recognized with high confidence
                                person_id = face_info.get("person_id")
                                person_dir = self.get_person_dir(group_id, person_id)
                                os.makedirs(person_dir, exist_ok=True)

                                # Save face to the person's directory
                                new_face_path = os.path.join(
                                    person_dir, f"{uuid.uuid4()}.jpg"
                                )
                                cv2.imwrite(new_face_path, detected_face)

                                # Add to results
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
                                new_person_id = f"person_{uuid.uuid4()}"
                                new_person_dir = self.get_person_dir(
                                    group_id, new_person_id
                                )
                                os.makedirs(new_person_dir, exist_ok=True)

                                # Save face to the new person's directory
                                new_face_path = os.path.join(
                                    new_person_dir, f"{uuid.uuid4()}.jpg"
                                )
                                cv2.imwrite(new_face_path, detected_face)

                                # Add to results
                                results.append(
                                    {
                                        "face_index": i,
                                        "person_id": new_person_id,
                                        "is_new_person": True,
                                        "saved_to": new_face_path,
                                    }
                                )
                    else:
                        # No existing model - create new person
                        new_person_id = f"person_{uuid.uuid4()}"
                        new_person_dir = self.get_person_dir(group_id, new_person_id)
                        os.makedirs(new_person_dir, exist_ok=True)

                        # Save face to the new person's directory
                        new_face_path = os.path.join(
                            new_person_dir, f"{uuid.uuid4()}.jpg"
                        )
                        cv2.imwrite(new_face_path, detected_face)

                        # Add to results
                        results.append(
                            {
                                "face_index": i,
                                "person_id": new_person_id,
                                "is_new_person": True,
                                "saved_to": new_face_path,
                            }
                        )

                    # Clean up temporary face file
                    if os.path.exists(face_path):
                        os.remove(face_path)

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

            # Update the model after processing all faces
            if results:
                # Use first successful face path as sample for retraining
                sample_paths = []
                for result in results:
                    if "saved_to" in result:
                        sample_paths.append(result["saved_to"])

                if sample_paths:
                    # Create representations using DeepFace
                    _ = DeepFace.find(
                        img_path=sample_paths[0],
                        db_path=group_dir,
                        model_name="VGG-Face",
                        detector_backend="retinaface",
                        enforce_detection=False,
                        align=True,
                        normalization="base",
                        silent=True,
                    )

            return True, results

        except Exception as e:
            error_trace = traceback.format_exc()
            return False, {"error": str(e), "trace": error_trace}

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

                            if confidence >= 0.5:  # Threshold for recognition
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
            shutil.rmtree(group_dir)
            return True, f"Group {group_id} deleted successfully"
        except Exception as e:
            return False, str(e)
