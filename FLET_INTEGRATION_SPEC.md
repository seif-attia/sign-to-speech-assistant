# Flet App Integration: Landmark Extraction & Normalization Specification

This document provides the exact feature extraction and coordinate normalization instructions for integrating MediaPipe with the Sign Language Recognition model (`sign_language_holistic_model.tflite`).

---

## 1. Input Tensor Specifications

- **Tensor Shape**: `(1, 30, 525)` (Batch Size: 1, Sequence Length: 30 frames, Features: 525)
- **Data Type**: `float32`
- **Output**: Softmax probabilities of length `2000` (mapped to `labels.json`).

---

## 2. Feature Vector Composition (525 Features per Frame)

Each video frame produces a 1D vector of **525 floats** composed of **175 landmarks $\times$ 3 coordinates $(x, y, z)$** in this exact order:

| Section | Landmark Count | Values $(x, y, z)$ | Vector Indices |
| :--- | :--- | :--- | :--- |
| **1. Pose** | 33 landmarks | 99 floats | `[0 : 99]` |
| **2. Left Hand** | 21 landmarks | 63 floats | `[99 : 162]` |
| **3. Right Hand** | 21 landmarks | 63 floats | `[162 : 225]` |
| **4. Face (Keypoints)** | 100 landmarks | 300 floats | `[225 : 525]` |
| **Total** | **175 landmarks** | **525 floats** | `[0 : 525]` |

### Exact Landmark Indices:

1. **Pose (33 landmarks)**:
   - MediaPipe Pose landmarks `0` to `32`.
   - Shoulder landmarks: Left Shoulder = `11`, Right Shoulder = `12`.
2. **Left Hand (21 landmarks)**:
   - MediaPipe Hand landmarks `0` to `20`.
3. **Right Hand (21 landmarks)**:
   - MediaPipe Hand landmarks `0` to `20`.
4. **Face (100 landmarks in sorted index order)**:
   - **Chin (10 points)**: `[18, 148, 150, 152, 175, 176, 199, 200, 377, 400]` (`152` is chin tip).
   - **Lips / Mouth (40 points)**: `[0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]`
   - **Nose (8 points)**: `[1, 2, 4, 5, 6, 168, 195, 197]`
   - **Left Eye (16 points)**: `[7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]`
   - **Right Eye (16 points)**: `[249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]`
   - **Left Eyebrow (5 points)**: `[63, 66, 70, 105, 107]`
   - **Right Eyebrow (5 points)**: `[293, 296, 300, 334, 336]`

---

## 3. Mathematical Normalization (Mid-Shoulder Centering)

All coordinates in the frame (Pose, Left Hand, Right Hand, Face) are normalized relative to the **Mid-Shoulder Centroid**:

### Step 1: Calculate Origin and Scale
From Pose landmark 11 (Left Shoulder) and landmark 12 (Right Shoulder):
$$\text{mid}_x = \frac{\text{ls}_x + \text{rs}_x}{2}, \quad \text{mid}_y = \frac{\text{ls}_y + \text{rs}_y}{2}, \quad \text{mid}_z = \frac{\text{ls}_z + \text{rs}_z}{2}$$
$$\text{scale} = \sqrt{(\text{ls}_x - \text{rs}_x)^2 + (\text{ls}_y - \text{rs}_y)^2 + (\text{ls}_z - \text{rs}_z)^2}$$
*(If shoulders are missing or scale < `1e-4`, set scale to `1.0`).*

### Step 2: Normalize Coordinates
For every detected landmark $(x, y, z)$:
$$x_{\text{norm}} = \frac{x - \text{mid}_x}{\text{scale}}, \quad y_{\text{norm}} = \frac{y - \text{mid}_y}{\text{scale}}, \quad z_{\text{norm}} = \frac{z - \text{mid}_z}{\text{scale}}$$

### Step 3: Handling Undetected Hands / Missing Face
- If a hand is **not in frame**, its 63 values must remain **strictly `0.0`** (do not subtract `mid` from zeros!).
- If face is **not detected**, its 300 values must remain **strictly `0.0`**.

---

## 4. Python Implementation Snippet (Copy-Paste for Flet App)

```python
import numpy as np

# 100 Face Landmarks in sorted order
FACE_INDICES = sorted([
    # Chin (10)
    18, 148, 150, 152, 175, 176, 199, 200, 377, 400,
    # Lips (40)
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146,
    178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317,
    318, 321, 324, 375, 402, 405, 409, 415,
    # Nose (8)
    1, 2, 4, 5, 6, 168, 195, 197,
    # Eyes (32)
    7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
    249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466,
    # Eyebrows (10)
    63, 66, 70, 105, 107, 293, 296, 300, 334, 336
])

def extract_and_normalize_frame(pose_result, hand_result, face_result) -> np.ndarray:
    """
    Extracts and normalizes a single video frame into a 525-element float32 vector.
    """
    # 1. Calculate Mid-Shoulder Origin and Scale
    mid_x, mid_y, mid_z = 0.0, 0.0, 0.0
    scale = 1.0

    if pose_result and pose_result.pose_landmarks:
        pose_lms = pose_result.pose_landmarks[0]
        ls = pose_lms[11]  # Left shoulder
        rs = pose_lms[12]  # Right shoulder
        mid_x = (ls.x + rs.x) / 2.0
        mid_y = (ls.y + rs.y) / 2.0
        mid_z = (ls.z + rs.z) / 2.0
        dist = np.sqrt((ls.x - rs.x)**2 + (ls.y - rs.y)**2 + (ls.z - rs.z)**2)
        if dist > 1e-4:
            scale = dist

    # 2. Pose (33 landmarks = 99 features)
    pose_vector = [0.0] * 99
    if pose_result and pose_result.pose_landmarks:
        pose_lms = pose_result.pose_landmarks[0]
        for i, lm in enumerate(pose_lms):
            pose_vector[i * 3]     = (lm.x - mid_x) / scale
            pose_vector[i * 3 + 1] = (lm.y - mid_y) / scale
            pose_vector[i * 3 + 2] = (lm.z - mid_z) / scale

    # 3. Left Hand & Right Hand (21 landmarks each = 63 + 63 features)
    left_hand_vector = [0.0] * 63
    right_hand_vector = [0.0] * 63

    if hand_result and hand_result.hand_landmarks and hand_result.handedness:
        for idx, hlms in enumerate(hand_result.hand_landmarks):
            label = hand_result.handedness[idx][0].category_name
            coords = []
            for lm in hlms:
                coords.extend([
                    (lm.x - mid_x) / scale,
                    (lm.y - mid_y) / scale,
                    (lm.z - mid_z) / scale
                ])
            # If camera view is mirrored: Left category is user's Right hand
            if label == "Left":
                right_hand_vector = coords
            elif label == "Right":
                left_hand_vector = coords

    # 4. Face (100 landmarks = 300 features)
    face_vector = [0.0] * 300
    if face_result and face_result.face_landmarks:
        face_lms = face_result.face_landmarks[0]
        coords = []
        for idx in FACE_INDICES:
            lm = face_lms[idx]
            coords.extend([
                (lm.x - mid_x) / scale,
                (lm.y - mid_y) / scale,
                (lm.z - mid_z) / scale
            ])
        face_vector = coords

    # 5. Concatenate to 525-dimensional float32 vector
    return np.array(pose_vector + left_hand_vector + right_hand_vector + face_vector, dtype=np.float32)
```

---

## 5. Sequence Buffering & TFLite Inference

```python
# Maintain a rolling circular buffer of 30 frames
sequence_buffer = []

def process_camera_frame(mp_image):
    pose_res = pose_detector.detect(mp_image)
    hand_res = hand_detector.detect(mp_image)
    face_res = face_detector.detect(mp_image)

    # 1. Extract normalized 525-vector
    frame_features = extract_and_normalize_frame(pose_res, hand_res, face_res)
    
    # 2. Append to circular 30-frame buffer
    sequence_buffer.append(frame_features)
    if len(sequence_buffer) > 30:
        sequence_buffer.pop(0)

    # 3. Predict when buffer reaches 30 frames
    if len(sequence_buffer) == 30:
        input_data = np.expand_dims(sequence_buffer, axis=0)  # Shape: (1, 30, 525), float32
        
        # Run TFLite / LiteRT
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        probabilities = interpreter.get_tensor(output_details[0]["index"])[0]  # Shape: (2000,)
        
        pred_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_idx])
        return pred_idx, confidence

    return None, 0.0
```
