"""Storage management for face recognition data."""

import base64
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
        Create a temporary user ID with UUID format (no prefix).

        Returns:
            str: Temporary user ID
        """
        return str(uuid.uuid4())

    def is_temp_user(self, person_id):
        """
        Check if a person ID is a temporary user by checking metadata.

        Args:
            person_id (str): Person identifier

        Returns:
            bool: True if temporary user, False otherwise
        """
        # Check metadata to determine if user is temporary
        for group_dir in os.listdir(self.data_dir):
            group_path = os.path.join(self.data_dir, group_dir)
            if os.path.isdir(group_path):
                person_dir = os.path.join(group_path, person_id)
                if os.path.exists(person_dir):
                    metadata = self.get_user_metadata(group_dir, person_id)
                    return metadata.get("is_temp_user", False)
        return False

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

        # Don't override is_temp_user if it's already set in the metadata
        # This prevents circular dependency issues
        if "is_temp_user" not in metadata:
            metadata["is_temp_user"] = False  # Default to permanent user

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
                )  # Get metadata
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
            target_exists = os.path.exists(target_dir)

            # Create target directory if it doesn't exist
            if not target_exists:
                self.create_person_directory(group_id, target_person_id)

            merged_info = {
                "target_person_id": target_person_id,
                "merged_sources": [],
                "total_faces_moved": 0,
                "errors": [],
                "target_existed": target_exists,
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
                        base_name, ext = os.path.splitext(filename)
                        target_filename = filename
                        counter = 1

                        # Check for filename conflicts and generate unique name if needed
                        while os.path.exists(os.path.join(target_dir, target_filename)):
                            target_filename = f"{base_name}_{counter}{ext}"
                            counter += 1

                        target_path = os.path.join(target_dir, target_filename)

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
                merged_info[
                    "total_faces_moved"
                ] += faces_moved  # Remove source directory
                try:
                    shutil.rmtree(source_dir)
                except Exception as e:
                    merged_info["errors"].append(
                        f"Failed to remove source directory {source_person_id}: {str(e)}"
                    )

            # Update target metadata
            target_metadata = self.get_user_metadata(group_id, target_person_id)

            # If target user didn't exist, initialize basic metadata
            if not merged_info["target_existed"]:
                target_metadata.update(
                    {
                        "created_at": datetime.now().isoformat(),
                        "person_id": target_person_id,
                        "is_temp_user": False,  # Merged users are considered permanent
                        "created_from_merge": True,
                    }
                )

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
        try:
            person_dir = self.get_person_dir(group_id, person_id)

            if not os.path.exists(person_dir):
                return False, f"User '{person_id}' not found in group '{group_id}'"

            # Get all image files in the person directory
            image_files = []
            for filename in os.listdir(person_dir):
                if (
                    filename.lower().endswith((".jpg", ".jpeg", ".png"))
                    and filename != "metadata.json"
                ):
                    file_path = os.path.join(person_dir, filename)
                    file_stat = os.stat(file_path)
                    image_files.append(
                        {
                            "filename": filename,
                            "path": file_path,
                            "created_at": file_stat.st_ctime,
                        }
                    )

            if not image_files:
                return (
                    False,
                    f"No images found for user '{person_id}' in group '{group_id}'",
                )

            # Sort by creation time (most recent first)
            image_files.sort(key=lambda x: x["created_at"], reverse=True)
            latest_image = image_files[0]

            # Read and encode the image to base64
            with open(latest_image["path"], "rb") as image_file:
                image_binary = image_file.read()
                image_base64 = base64.b64encode(image_binary).decode("utf-8")

            result_data = {
                "image_base64": image_base64,
                "filename": latest_image["filename"],
                "created_at": datetime.fromtimestamp(
                    latest_image["created_at"]
                ).isoformat(),
                "file_size": len(image_binary),
            }

            return True, result_data

        except Exception as e:
            return False, f"Error retrieving last image: {str(e)}"

    def create_permanent_user_id(self):
        """
        Create a permanent user ID with UUID format (no prefix).

        Returns:
            str: Permanent user ID
        """
        return str(uuid.uuid4())
