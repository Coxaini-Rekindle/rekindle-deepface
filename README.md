# DeepFace Recognition Service

A Python Flask server that provides face recognition capabilities using the DeepFace library with support for temporary user management and merging workflows.

## Features

- **Temporary User Management**: Unrecognized faces get temporary IDs for later association
- **User Merging**: Merge temporary users with existing permanent users
- **Face Recognition**: Recognize faces with confidence scoring
- **Group Management**: Organize users into separate groups
- **GPU Acceleration**: RTX/CUDA support for fast processing
- **Performance Modes**: Configurable speed vs accuracy settings

## New Workflow

1. **Add Faces**: Upload images to detect and recognize faces
   - Recognized faces → Associated with existing users
   - Unrecognized faces → Assigned temporary user IDs
2. **Review Users**: List all permanent and temporary users in a group
3. **Merge Users**: Associate temporary users with the correct permanent users
4. **Recognize**: Use trained models to identify faces in new images

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

#### Add Faces to Group (NEW!)

```
POST /api/add_faces
```

Add faces from an image to a group. Recognizes existing faces and creates temporary users for unrecognized ones.

**Request:**
```json
{
    "group_id": "office_team",
    "image": "base64_encoded_image"
}
```

#### Merge Users (NEW!)

```
POST /api/merge_users
```

Merge temporary users with existing permanent users.

**Request:**
```json
{
    "group_id": "office_team",
    "source_person_ids": ["temp_abc123", "temp_def456"],
    "target_person_id": "person_john_doe"
}
```

#### List Users in Group (NEW!)

```
GET /api/groups/{group_id}/users
```

List all permanent and temporary users in a group.

#### Recognize Faces

```
POST /api/recognize
```

Recognize faces in images against a trained model.

**Request:**
```json
{
    "group_id": "office_team",
    "images": ["base64_encoded_image1", "base64_encoded_image2"]
}
```

#### Delete Group

```
DELETE /api/delete_group
```

Delete a group and all its associated data.

**Request:**
```json
{
    "group_id": "office_team"
}
```

#### Set Performance Mode

```
POST /api/set_performance
```

Configure performance settings (speed, accuracy, balanced, gpu_optimized).

**Request:**
```json
{
    "mode": "balanced"
}
```

### Migration from Old API

The old `/api/process_faces` endpoint has been replaced with `/api/add_faces`. The new endpoint:
- Supports temporary user creation for unrecognized faces
- Provides detailed recognition information
- Includes timing data for performance monitoring

For detailed API documentation with request/response examples, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

## Example Workflow

1. **Add faces to a group:**
   ```bash
   curl -X POST http://localhost:5000/api/add_faces \
     -H "Content-Type: application/json" \
     -d '{"group_id": "team1", "image": "base64_data..."}'
   ```

2. **List users to see temporary users:**
   ```bash
   curl http://localhost:5000/api/groups/team1/users
   ```

3. **Merge temporary user with existing person:**
   ```bash
   curl -X POST http://localhost:5000/api/merge_users \
     -H "Content-Type: application/json" \
     -d '{"group_id": "team1", "source_person_ids": ["temp_123"], "target_person_id": "person_john"}'
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