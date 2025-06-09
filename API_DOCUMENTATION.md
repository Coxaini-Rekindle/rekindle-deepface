# Face Recognition API Documentation

## Overview

This Face Recognition API provides a workflow for managing face recognition with temporary user support and merging capabilities. The API is designed to handle scenarios where faces may not be immediately recognized and need to be manually associated with existing users later.

## Workflow

### 1. Add Faces to Group
Use this endpoint to add faces from images to a group. The system will try to recognize existing faces and create temporary users for unrecognized faces.

### 2. List Users in Group
Review all users (permanent and temporary) in a group to understand what faces were detected.

### 3. Merge Users
Merge temporary users with existing permanent users when you identify who they should be associated with.

### 4. Recognize Faces
Use trained models to recognize faces in new images.

## API Endpoints

### POST /api/add_faces

Add faces from an image to a group.

**Request:**
```json
{
    "group_id": "unique_group_identifier",
    "image": "base64_encoded_image"
}
```

**Success Response:**
```json
{
    "status": "success",
    "group_id": "example_group",
    "faces": [
        {
            "face_index": 0,
            "person_id": "person_12345678-1234-1234-1234-123456789abc",
            "is_temp_user": false,
            "is_new_person": false,
            "confidence": 0.85,
            "recognition_type": "recognized",
            "saved_to": "/path/to/saved/image.jpg"
        },
        {
            "face_index": 1,
            "person_id": "temp_98765432-4321-4321-4321-cba987654321",
            "is_temp_user": true,
            "is_new_person": true,
            "confidence": 0.0,
            "recognition_type": "temp_user",
            "saved_to": "/path/to/saved/temp_image.jpg"
        }
    ],
    "timing": {
        "total_processing_time": 2.45,
        "face_detection_time": 0.32
    }
}
```

**Recognition Types:**
- `recognized`: Face matched an existing person with high confidence
- `temp_user`: Face not recognized, assigned a temporary user ID
- `unknown`: Face detection/recognition failed

### POST /api/merge_users

Merge multiple users into a target user.

**Request:**
```json
{
    "group_id": "unique_group_identifier",
    "source_person_ids": ["12345678-1234-1234-1234-123456789abc", "87654321-4321-4321-4321-cba987654321"],
    "target_person_id": "abcdef12-3456-7890-abcd-ef1234567890"
}
```

**Success Response:**
```json
{
    "status": "success",
    "group_id": "example_group",
    "merged_count": 2,
    "target_person_id": "abcdef12-3456-7890-abcd-ef1234567890",
    "source_person_ids": ["12345678-1234-1234-1234-123456789abc", "87654321-4321-4321-4321-cba987654321"],
    "message": "Successfully merged 2 users into abcdef12-3456-7890-abcd-ef1234567890"
}
```

### GET /api/groups/{group_id}/users

List all users in a group.

**Success Response:**
```json
{
    "status": "success",
    "group_id": "example_group",
    "users": {
        "permanent": [
            {
                "person_id": "person_12345678-1234-1234-1234-123456789abc",
                "face_count": 5,
                "metadata": {
                    "created_at": "2025-06-08T10:30:00",
                    "recognition_type": "recognized",
                    "confidence": 0.85
                }
            }
        ],
        "temporary": [
            {
                "person_id": "temp_98765432-4321-4321-4321-cba987654321",
                "face_count": 2,
                "metadata": {
                    "created_at": "2025-06-08T11:15:00",
                    "recognition_type": "temp_user",
                    "confidence": 0.0
                }
            }
        ]
    },
    "summary": {
        "total_users": 2,
        "permanent_users": 1,
        "temporary_users": 1
    }
}
```

### GET /api/groups/{group_id}/users/{person_id}/last_image

Get the latest (last) image for a specific user.

**Success Response:**
```json
{
    "status": "success",
    "group_id": "example_group",
    "person_id": "person_12345678-1234-1234-1234-123456789abc",
    "image": {
        "image_base64": "base64_encoded_image_data",
        "filename": "abc123def456.jpg",
        "created_at": "2025-06-08T15:30:45.123456",
        "file_size": 25648
    }
}
```

**Error Response:**
```json
{
    "error": "User 'person_id' not found in group 'group_id'"
}
```

### POST /api/recognize

Recognize faces in images against a trained model.

**Request:**
```json
{
    "group_id": "unique_group_identifier",
    "images": [
        "base64_encoded_image1",
        "base64_encoded_image2"
    ]
}
```

**Success Response:**
```json
{
    "status": "success",
    "group_id": "example_group",
    "results": [
        {
            "image_index": 0,
            "faces": [
                {
                    "person_id": "person_12345678-1234-1234-1234-123456789abc",
                    "confidence": 0.85,
                    "facial_area": {
                        "x": 100,
                        "y": 50,
                        "w": 200,
                        "h": 250
                    }
                }
            ]
        },
        {
            "image_index": 1,
            "error": "No faces detected in the image"
        }
    ]
}
```

### DELETE /api/delete_group

Delete a group and all its associated data.

**Request:**
```json
{
    "group_id": "unique_group_identifier"
}
```

**Success Response:**
```json
{
    "status": "success",
    "message": "Group 'example_group' and all associated data deleted successfully"
}
```

### POST /api/set_performance

Set the performance mode for face recognition.

**Request:**
```json
{
    "mode": "balanced"
}
```

**Valid Modes:**
- `speed`: Prioritize speed over accuracy
- `accuracy`: Prioritize accuracy over speed
- `balanced`: Balance between speed and accuracy
- `gpu_optimized`: Settings optimized for GPU processing

**Success Response:**
```json
{
    "status": "success",
    "message": "Performance mode set to: balanced",
    "settings": {
        "detector_backend": "retinaface",
        "recognition_model": "VGG-Face",
        "distance_metric": "cosine",
        "confidence_threshold": 0.7
    }
}
```

## Error Responses

All endpoints return error responses in this format:

```json
{
    "error": "Error message describing what went wrong",
    "trace": "Optional stack trace for debugging (in development mode)"
}
```

Common HTTP status codes:
- `400`: Bad Request (missing parameters, invalid input)
- `404`: Not Found (group doesn't exist)
- `500`: Internal Server Error (processing failure)

## User ID Formats

All user IDs are now standardized as plain UUIDs without prefixes:

- **All Users**: UUID format (e.g., `12345678-1234-1234-1234-123456789abc`)
- **User Type Detection**: Temporary vs permanent users are distinguished by metadata, not ID format
- **Backward Compatibility**: The system can still identify user types through stored metadata

## Example Workflows

### Workflow 1: Add New Faces and Merge

1. **Add faces to a group:**
   ```bash
   POST /api/add_faces
   {
       "group_id": "office_team",
       "image": "base64_image_data"
   }
   ```

2. **Check users in the group:**
   ```bash
   GET /api/groups/office_team/users
   ```

3. **Merge temporary user with existing person:**
   ```bash
   POST /api/merge_users
   {
       "group_id": "office_team",
       "source_person_ids": ["temp_abc123"],
       "target_person_id": "person_john_doe"
   }
   ```

### Workflow 2: Recognize Faces

1. **Recognize faces in new images:**
   ```bash
   POST /api/recognize
   {
       "group_id": "office_team",
       "images": ["base64_image1", "base64_image2"]
   }
   ```

## Technical Notes

- All images should be base64 encoded
- The API supports common image formats (JPEG, PNG, etc.)
- Face detection and recognition are performed using deep learning models
- Temporary user metadata includes creation timestamps and recognition confidence
- Merging operations move all face images from source users to the target user
- Group data is persisted to disk and survives server restarts
