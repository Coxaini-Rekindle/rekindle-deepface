"""Storage management for face recognition data."""

import json
import os
import shutil
import uuid
from datetime import datetime

import cv2


class StorageManager:
    """Manages the storage of face data and models."""

    def __init__(self, data_dir, models_dir):
        """
        Initialize the storage manager.

        Args:
            data_dir (str): Base directory for storing face data
            models_dir (str): Directory for storing models
        """
        self.data_dir = data_dir
        self.models_dir = models_dir

        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

    def get_group_dir(self, group_id):
        """
        Get the directory path for a group.

        Args:
            group_id (str): Group identifier

        Returns:
            str: Path to the group directory
        """
        return os.path.join(self.data_dir, group_id)

    def get_person_dir(self, group_id, person_id):
        """
        Get the directory path for a person within a group.

        Args:
            group_id (str): Group identifier
            person_id (str): Person identifier

        Returns:
            str: Path to the person directory
        """
        return os.path.join(self.get_group_dir(group_id), person_id)

    def create_person_directory(self, group_id, person_id):
        """
        Create a directory for a person.

        Args:
            group_id (str): Group identifier
            person_id (str): Person identifier

        Returns:
            str: Path to the created person directory
        """
        person_dir = self.get_person_dir(group_id, person_id)
        os.makedirs(person_dir, exist_ok=True)
        return person_dir

    def save_face_image(self, face_img, group_id, person_id):
        """
        Save a face image to the appropriate person directory.

        Args:
            face_img: Face image as numpy array
            group_id (str): Group identifier
            person_id (str): Person identifier

        Returns:
            str: Path to the saved image
        """
        person_dir = self.create_person_directory(group_id, person_id)
        image_filename = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join(person_dir, image_filename)
        cv2.imwrite(image_path, face_img)
        return image_path

    def save_temp_face(self, face_img):
        """
        Save a face image to a temporary location.

        Args:
            face_img: Face image as numpy array

        Returns:
            str: Path to the saved temporary image
        """
        temp_filename = f"face_{uuid.uuid4()}.jpg"
        temp_path = os.path.join(self.data_dir, temp_filename)
        cv2.imwrite(temp_path, face_img)
        return temp_path

    def create_temp_user_id(self):
        """
        Create a temporary user ID with a specific prefix.

        Returns:
            str: Temporary user ID
        """
        return f"temp_{uuid.uuid4().hex[:8]}"

    def is_temp_user(self, person_id):
        """
        Check if a person ID is a temporary user.

        Args:
            person_id (str): Person identifier

        Returns:
            bool: True if temporary user, False otherwise
        """
        return person_id.startswith("temp_")

    def get_user_metadata_path(self, group_id, person_id):
        """
        Get the path to user metadata file.

        Args:
            group_id (str): Group identifier
            person_id (str): Person identifier

        Returns:
            str: Path to metadata file
        """
        person_dir = self.get_person_dir(group_id, person_id)
        return os.path.join(person_dir, "metadata.json")

    def save_user_metadata(self, group_id, person_id, metadata):
        """
        Save metadata for a user.

        Args:
            group_id (str): Group identifier
            person_id (str): Person identifier
            metadata (dict): Metadata to save
        """
        self.create_person_directory(group_id, person_id)
        metadata_path = self.get_user_metadata_path(group_id, person_id)

        # Add timestamp
        metadata["last_updated"] = datetime.now().isoformat()
        metadata["is_temp_user"] = self.is_temp_user(person_id)

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def get_user_metadata(self, group_id, person_id):
        """
        Get metadata for a user.

        Args:
            group_id (str): Group identifier
            person_id (str): Person identifier

        Returns:
            dict: User metadata or empty dict if not found
        """
        metadata_path = self.get_user_metadata_path(group_id, person_id)
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def list_users_in_group(self, group_id):
        """
        List all users (permanent and temporary) in a group.

        Args:
            group_id (str): Group identifier

        Returns:
            dict: Dictionary with user information
        """
        group_dir = self.get_group_dir(group_id)
        users = {"permanent": [], "temporary": []}

        if not os.path.exists(group_dir):
            return users

        for person_id in os.listdir(group_dir):
            person_dir = os.path.join(group_dir, person_id)
            if os.path.isdir(person_dir):
                # Count face images
                face_count = len(
                    [
                        f
                        for f in os.listdir(person_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                )

                # Get metadata
                metadata = self.get_user_metadata(group_id, person_id)

                user_info = {
                    "person_id": person_id,
                    "face_count": face_count,
                    "created_at": metadata.get("created_at"),
                    "last_updated": metadata.get("last_updated"),
                    "metadata": metadata,
                }

                if self.is_temp_user(person_id):
                    users["temporary"].append(user_info)
                else:
                    users["permanent"].append(user_info)

        return users

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
        try:
            target_dir = self.get_person_dir(group_id, target_person_id)

            # Create target directory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)

            merged_info = {
                "target_person_id": target_person_id,
                "merged_sources": [],
                "total_faces_moved": 0,
                "errors": [],
            }

            for source_person_id in source_person_ids:
                source_dir = self.get_person_dir(group_id, source_person_id)

                if not os.path.exists(source_dir):
                    merged_info["errors"].append(
                        f"Source user {source_person_id} not found"
                    )
                    continue

                if source_person_id == target_person_id:
                    merged_info["errors"].append(
                        f"Cannot merge user into itself: {source_person_id}"
                    )
                    continue

                # Move all face images
                faces_moved = 0
                for filename in os.listdir(source_dir):
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        source_path = os.path.join(source_dir, filename)
                        # Generate new unique filename to avoid conflicts
                        new_filename = f"{uuid.uuid4()}.jpg"
                        target_path = os.path.join(target_dir, new_filename)

                        try:
                            shutil.move(source_path, target_path)
                            faces_moved += 1
                        except Exception as e:
                            merged_info["errors"].append(
                                f"Failed to move {filename}: {str(e)}"
                            )

                # Get source metadata
                source_metadata = self.get_user_metadata(group_id, source_person_id)

                merged_info["merged_sources"].append(
                    {
                        "person_id": source_person_id,
                        "faces_moved": faces_moved,
                        "was_temp_user": self.is_temp_user(source_person_id),
                        "source_metadata": source_metadata,
                    }
                )
                merged_info["total_faces_moved"] += faces_moved

                # Remove source directory
                try:
                    shutil.rmtree(source_dir)
                except Exception as e:
                    merged_info["errors"].append(
                        f"Failed to remove source directory {source_person_id}: {str(e)}"
                    )

            # Update target metadata
            target_metadata = self.get_user_metadata(group_id, target_person_id)
            target_metadata["merge_history"] = target_metadata.get("merge_history", [])
            target_metadata["merge_history"].append(
                {
                    "merged_at": datetime.now().isoformat(),
                    "merged_sources": [
                        info["person_id"] for info in merged_info["merged_sources"]
                    ],
                    "total_faces_added": merged_info["total_faces_moved"],
                }
            )

            self.save_user_metadata(group_id, target_person_id, target_metadata)

            return (
                True,
                f"Successfully merged {len(merged_info['merged_sources'])} users into {target_person_id}",
                merged_info,
            )

        except Exception as e:
            return False, f"Error during merge: {str(e)}", {}

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

    def cleanup_temp_files(self, file_paths):
        """
        Clean up temporary files.

        Args:
            file_paths (list): List of file paths to delete
        """
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)

    def group_exists(self, group_id):
        """
        Check if a group exists and has face data.

        Args:
            group_id (str): Group identifier

        Returns:
            bool: True if the group exists and has data, False otherwise
        """
        group_dir = self.get_group_dir(group_id)
        return os.path.exists(group_dir) and os.listdir(group_dir)
