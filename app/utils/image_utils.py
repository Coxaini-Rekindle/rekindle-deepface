import base64
import io
import os
import uuid

from PIL import Image


def decode_base64_image(image_data):
    """
    Decode a base64 encoded image.

    Args:
        image_data (str): Base64 encoded image data

    Returns:
        bytes: Decoded image binary data
    """
    return base64.b64decode(
        image_data.split(",")[1] if "," in image_data else image_data
    )


def save_image_from_base64(image_data, directory, filename=None):
    """
    Save a base64 encoded image to the specified directory.

    Args:
        image_data (str): Base64 encoded image data
        directory (str): Directory to save the image to
        filename (str, optional): Filename to use. If None, a UUID will be generated.

    Returns:
        tuple: (success, image_path or error_message)
    """
    try:
        # Make sure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Decode base64 image
        image_binary = decode_base64_image(image_data)
        image = Image.open(io.BytesIO(image_binary))

        # Generate a unique filename if not provided
        if not filename:
            filename = f"{uuid.uuid4()}.jpg"

        image_path = os.path.join(directory, filename)
        image.save(image_path)

        return True, image_path
    except Exception as e:
        return False, str(e)


def save_temp_image(image_data, temp_dir):
    """
    Save a base64 encoded image to a temporary file.

    Args:
        image_data (str): Base64 encoded image data
        temp_dir (str): Directory to save the temporary image

    Returns:
        str: Path to the temporary image file
    """
    # Make sure the directory exists
    os.makedirs(temp_dir, exist_ok=True)

    # Generate a temporary filename
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
    temp_path = os.path.join(temp_dir, temp_filename)

    # Decode and save the image
    image_binary = decode_base64_image(image_data)

    with open(temp_path, "wb") as f:
        f.write(image_binary)

    return temp_path
