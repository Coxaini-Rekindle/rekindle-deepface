# DeepFace Recognition Service

A Python Flask server that provides face recognition capabilities using the DeepFace library.

## Features

- Train face recognition models for different groups
- Recognize faces in images
- Add new people to existing models
- Delete groups and their data
- GPU-accelerated face recognition with RTX support

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. For GPU acceleration (recommended):
   - Make sure you have the latest NVIDIA drivers installed
   - Install CUDA Toolkit and cuDNN compatible with your TensorFlow version

## Usage

### Starting the Server

Run the Flask application:

```
python app.py
```

The server will be available at `http://localhost:5000`.

### API Endpoints

#### Health Check

```
GET /healthcheck
```

#### Train a Model

```
POST /api/train
Content-Type: application/json

{
    "group_id": "your_group_id",
    "faces": [
        {
            "person_id": "person1",
            "image": "base64_encoded_image"
        },
        {
            "person_id": "person2",
            "image": "base64_encoded_image"
        }
    ]
}
```

#### Recognize Faces

```
POST /api/recognize
Content-Type: application/json

{
    "group_id": "your_group_id",
    "images": [
        "base64_encoded_image1",
        "base64_encoded_image2"
    ]
}
```

#### Add a Person

```
POST /api/add_person
Content-Type: application/json

{
    "group_id": "your_group_id",
    "person_id": "new_person",
    "images": [
        "base64_encoded_image1",
        "base64_encoded_image2"
    ]
}
```

#### Delete a Group

```
DELETE /api/delete_group
Content-Type: application/json

{
    "group_id": "your_group_id"
}
```

#### Set Performance Mode

```
POST /api/set_performance
Content-Type: application/json

{
    "mode": "gpu_optimized"
}
```

Available modes:
- `speed`: Prioritize speed over accuracy
- `accuracy`: Prioritize accuracy over speed
- `balanced`: Balance between speed and accuracy
- `gpu_optimized`: Settings optimized for GPU processing (default)

## Performance Optimization

The service is optimized for NVIDIA GPUs and includes:
- Automatic GPU detection and configuration
- Mixed precision training for faster processing
- Model preloading to reduce inference time
- Image resizing to optimize memory usage
- Configurable detector backends and recognition models

## Testing

A test script is provided to verify API functionality:

```
python test_api.py --action healthcheck
python test_api.py --action train --group_id test_group --image_dir ./test_images
python test_api.py --action recognize --group_id test_group --image test_image.jpg
python test_api.py --action add --group_id test_group --person_id new_person --image face.jpg
python test_api.py --action delete --group_id test_group
```

## Directory Structure

- `data/`: Stores face data organized by groups and persons
- `models/`: Stores trained models
- `app.py`: Main Flask application
- `test_api.py`: Script to test API endpoints

## Notes

- The server uses RTX GPU acceleration when available
- Default face recognition model is VGG-Face with RetinaFace detection
- Face recognition threshold is configurable via the API
- All images are processed as base64 encoded strings