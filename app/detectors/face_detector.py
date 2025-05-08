"""Face detection functionality."""

import time

from deepface import DeepFace


class FaceDetector:
    """Handles face detection operations."""

    def __init__(self, detector_backend="retinaface"):
        """
        Initialize the face detector.

        Args:
            detector_backend (str): Backend to use for face detection
        """
        self.detector_backend = detector_backend

    def detect_faces(self, img, align=True):
        """
        Detect faces in an image.

        Args:
            img: Image as numpy array or path to image file
            align (bool): Whether to align detected faces

        Returns:
            tuple: (face_objects, detection_time)
        """
        start_time = time.time()

        # Detect faces using DeepFace
        face_objs = DeepFace.extract_faces(
            img_path=img,
            detector_backend=self.detector_backend,
            enforce_detection=False,
            align=align,
        )

        detection_time = time.time() - start_time
        print(
            f"Face detection completed in {detection_time:.2f} seconds. Found {len(face_objs)} faces."
        )

        return face_objs, detection_time
