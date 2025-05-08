"""Image processing utilities for face recognition."""
import cv2
import numpy as np


class ImageProcessor:
    """Handles image processing operations for face recognition."""
    
    @staticmethod
    def optimize_image_size(img, max_dim=1280):
        """
        Resize an image if its dimensions exceed the maximum allowed.
        
        Args:
            img: OpenCV image (numpy array)
            max_dim: Maximum allowed dimension
            
        Returns:
            tuple: (resized_image, was_resized, original_shape, new_shape)
        """
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            resized_img = cv2.resize(img, new_size)
            return resized_img, True, (h, w), new_size
        return img, False, (h, w), (h, w)
    
    @staticmethod
    def extract_face(img, facial_area):
        """
        Extract a face from an image using facial area coordinates.
        
        Args:
            img: OpenCV image (numpy array)
            facial_area: Dict with keys 'x', 'y', 'w', 'h'
            
        Returns:
            numpy.ndarray: Extracted face image
        """
        x = facial_area.get('x', 0)
        y = facial_area.get('y', 0)
        w = facial_area.get('w', 0)
        h = facial_area.get('h', 0)
        
        # Extract the face from the image
        return img[y:y+h, x:x+w]
    
    @staticmethod
    def create_dummy_image(size=(100, 100)):
        """
        Create a small dummy image for model preloading.
        
        Args:
            size: Tuple of (width, height)
            
        Returns:
            numpy.ndarray: Dummy image
        """
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)