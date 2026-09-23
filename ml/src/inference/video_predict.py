import base64

import cv2
import numpy as np
import torch
from PIL import Image

from ml.src.inference.predict import (
    load_model,
    _transform,
    DEVICE,
)


def get_face_detector():
    cascade = cv2.data.haarcascades + (
        "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(cascade)

    if detector.empty():
        raise RuntimeError(
            "Unable to load OpenCV Haar cascade."
        )

    return detector


def create_thumbnail(frame):
    """
    Create a small JPEG thumbnail and return it
    as a base64 data URI for the frontend.
    """

    height, width = frame.shape[:2]

    max_width = 480

    if width > max_width:
        scale = max_width / width
        new_width = max_width
        new_height = int(height * scale)

        frame = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            75,
        ],
    )

    if not success:
        return None

    base64_image = base64.b64encode(
        encoded.tobytes()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{base64_image}"


def predict_video(
    video_path,
    sample_every=15,
    max_frames=40,
):
    model = load_model()
    detector = get_face_detector()

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            "Unable to open video."
        )

    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    frame_predictions = []
    frame_index = 0

    try:
        while len(frame_predictions) < max_frames:
            success, frame = capture.read()

            if not success:
                break

            current_frame = frame_index

            frame_index += 1

            if current_frame % sample_every != 0:
                continue

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )

            if len(faces) == 0:
                continue

            # Select the largest detected face.
            x, y, w, h = max(
                faces,
                key=lambda box: box[2] * box[3],
            )

            margin_x = int(w * 0.20)
            margin_y = int(h * 0.20)

            x1 = max(
                0,
                x - margin_x,
            )

            y1 = max(
                0,
                y - margin_y,
            )

            x2 = min(
                frame.shape[1],
                x + w + margin_x,
            )

            y2 = min(
                frame.shape[0],
                y + h + margin_y,
            )

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB,
            )

            image = Image.fromarray(face_rgb)

            tensor = _transform(image)
            tensor = tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(tensor)

                probabilities = torch.softmax(
                    output,
                    dim=1,
                )[0]

            fake_probability = float(
                probabilities[1].item()
            )

            thumbnail = create_thumbnail(frame)

            timestamp = current_frame / fps

            frame_predictions.append({
                "frame": current_frame,
                "timestamp_seconds": round(
                    timestamp,
                    2,
                ),
                "fake_probability": fake_probability,
                "thumbnail": thumbnail,
            })

    finally:
        capture.release()

    if not frame_predictions:
        raise RuntimeError(
            "No usable faces were detected in the video."
        )

    probabilities = [
        item["fake_probability"]
        for item in frame_predictions
    ]

    # Mean probability across sampled frames.
    video_probability = float(
        np.mean(probabilities)
    )

    prediction = (
        "FAKE"
        if video_probability >= 0.5
        else "REAL"
    )

    suspicious_frames = sorted(
        frame_predictions,
        key=lambda item: item["fake_probability"],
        reverse=True,
    )[:5]

    return {
        "assessment": prediction,
        "confidence": max(
            video_probability,
            1.0 - video_probability,
        ),
        "real_probability": 1.0 - video_probability,
        "fake_probability": video_probability,
        "frames_analyzed": len(frame_predictions),
        "suspicious_frames": suspicious_frames,
    }