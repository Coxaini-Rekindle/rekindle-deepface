"""Configuration settings for face recognition."""

class FaceRecognitionConfig:
    """Configuration class for face recognition settings."""
    
    def __init__(self):
        # Default performance settings
        self.detector_backend = "retinaface"  # Options: retinaface, mtcnn, opencv, ssd, dlib, mediapipe
        self.recognition_model = "VGG-Face"  # Options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, ArcFace
        self.distance_metric = "cosine"  # Options: cosine, euclidean, euclidean_l2
        self.confidence_threshold = 0.5  # Threshold for face recognition confidence
        self.max_image_dimension = 1280  # Maximum dimension for processing images
        
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
        if mode == "speed":
            self.detector_backend = "ssd"  # Faster but less accurate
            self.recognition_model = "VGG-Face"  # Good balance
            self.confidence_threshold = 0.4  # Lower threshold for faster matching
        elif mode == "accuracy":
            self.detector_backend = "retinaface"  # More accurate
            self.recognition_model = "ArcFace"  # More accurate
            self.confidence_threshold = 0.6  # Higher threshold for higher accuracy
        elif mode == "balanced":
            self.detector_backend = "retinaface"
            self.recognition_model = "VGG-Face"
            self.confidence_threshold = 0.5
        elif mode == "gpu_optimized":
            self.detector_backend = "retinaface"
            self.recognition_model = "VGG-Face"
            self.confidence_threshold = 0.5

        return {
            "mode": mode,
            "detector_backend": self.detector_backend,
            "recognition_model": self.recognition_model,
            "confidence_threshold": self.confidence_threshold,
        }