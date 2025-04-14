# DeepFace Recognition Service

A Python Flask server that provides face recognition capabilities using the DeepFace library.

## Features

- Train face recognition models for different groups
- Recognize faces in images
- Add new people to existing models
- Delete groups and their data

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

- The server uses the VGG-Face model and RetinaFace detection backend
- Face recognition threshold is set to 0.6 (configurable in code)
- All images are processed as base64 encoded strings