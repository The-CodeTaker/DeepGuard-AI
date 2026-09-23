import base64
import os
import sys
import tempfile

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@api_view(["GET"])
def health(request):
    return Response({
        "status": "ok",
        "service": "DeepGuard AI",
    })


@api_view(["POST"])
def predict(request):
    if "image" not in request.FILES:
        return Response(
            {
                "error": "No image uploaded.",
                "expected_field": "image",
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    uploaded_file = request.FILES["image"]

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if uploaded_file.content_type not in allowed_types:
        return Response(
            {
                "error": "Unsupported image type.",
                "allowed": list(allowed_types),
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    if uploaded_file.size > 10 * 1024 * 1024:
        return Response(
            {"error": "Image exceeds 10 MB limit."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    temp_path = None

    try:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name

        from ml.src.inference.predict import (
            extract_features,
            predict_image,
        )
        result = predict_image(temp_path)

        from ml.src.rag.vector_store import query_similar_artifacts
        retrieved_artifacts = query_similar_artifacts(
            extract_features(temp_path)
        )

        from ml.src.explainability.llm_explainer import generate_explanation
        forensic_report = generate_explanation(
            result["label"],
            result["confidence"],
            result["real_probability"],
            result["fake_probability"],
            retrieved_artifacts,
        )

        from ml.src.explainability.gradcam import generate_gradcam
        explanation = generate_gradcam(temp_path)

        return Response({
            "success": True,
            "filename": uploaded_file.name,
            "assessment": result["label"],
            "confidence": result["confidence"],
            "real_probability": result["real_probability"],
            "fake_probability": result["fake_probability"],
            "model": result["model"],
            "heatmap": explanation["heatmap"],
            "forensic_report": forensic_report,
            "retrieved_artifacts": retrieved_artifacts,
        })

    except Exception as exc:
        return Response(
            {
                "success": False,
                "error": str(exc),
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@api_view(["POST"])
def predict_video(request):
    if "video" not in request.FILES:
        return Response(
            {
                "error": "No video uploaded.",
                "expected_field": "video",
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    uploaded_file = request.FILES["video"]

    allowed_types = {
        ".mp4",
        ".webm",
        ".mov",
        ".avi",
    }
    
    extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    if extension not in allowed_types:
        return Response(
            {
                "error": "Unsupported video type.",
                "allowed": sorted(allowed_types),
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    if uploaded_file.size > 100 * 1024 * 1024:
        return Response(
            {"error": "Video exceeds 100 MB limit."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    temp_path = None

    try:
        suffix = os.path.splitext(
            uploaded_file.name
        )[1].lower()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name

        from ml.src.inference.video_predict import predict_video

        result = predict_video(temp_path)

        from ml.src.inference.predict import extract_features
        from ml.src.rag.vector_store import query_similar_artifacts

        retrieved_artifacts = []
        for frame in result.get("suspicious_frames", []):
            thumbnail = frame.get("thumbnail")
            if not thumbnail or "," not in thumbnail:
                continue

            frame_path = None
            try:
                image_bytes = base64.b64decode(
                    thumbnail.split(",", 1)[1]
                )
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg",
                ) as frame_file:
                    frame_file.write(image_bytes)
                    frame_path = frame_file.name

                matches = query_similar_artifacts(
                    extract_features(frame_path)
                )
                for match in matches:
                    if match not in retrieved_artifacts:
                        retrieved_artifacts.append(match)
            finally:
                if frame_path and os.path.exists(frame_path):
                    os.remove(frame_path)

        from ml.src.explainability.llm_explainer import generate_explanation
        forensic_report = generate_explanation(
            result["assessment"],
            result["confidence"],
            result["real_probability"],
            result["fake_probability"],
            retrieved_artifacts,
        )

        return Response({
            "success": True,
            "filename": uploaded_file.name,
            **result,
            "model": "efficientnet-b0",
            "forensic_report": forensic_report,
            "retrieved_artifacts": retrieved_artifacts,
        })

    except Exception as exc:
        return Response(
            {
                "success": False,
                "error": str(exc),
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)