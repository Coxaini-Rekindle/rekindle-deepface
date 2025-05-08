"""Storage management for face recognition data."""
import os
import shutil
import uuid
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