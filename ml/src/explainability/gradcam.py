from pathlib import Path
import base64
import io

import cv2
import numpy as np
import torch
from PIL import Image

from ml.src.inference.predict import load_model, _transform, DEVICE


def generate_gradcam(image_path, target_class=None):
    model = load_model()

    target_layer = model.features[-1][0]

    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:
        image = Image.open(image_path).convert("RGB")
        original = np.array(image)

        tensor = _transform(image)
        tensor = tensor.unsqueeze(0).to(DEVICE)

        model.zero_grad(set_to_none=True)

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1,
        )[0]

        if target_class is None:
            target_class = int(
                torch.argmax(probabilities).item()
            )

        score = output[0, target_class]
        score.backward()

        activation = activations[0]
        gradient = gradients[0]

        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights * activation
        ).sum(dim=1).squeeze(0)

        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(
            cam,
            (original.shape[1], original.shape[0]),
        )

        heatmap = np.uint8(
            255 * cam
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET,
        )

        original_bgr = cv2.cvtColor(
            original,
            cv2.COLOR_RGB2BGR,
        )

        overlay = cv2.addWeighted(
            original_bgr,
            0.55,
            heatmap,
            0.45,
            0,
        )

        overlay_rgb = cv2.cvtColor(
            overlay,
            cv2.COLOR_BGR2RGB,
        )

        result_image = Image.fromarray(
            overlay_rgb
        )

        buffer = io.BytesIO()

        result_image.save(
            buffer,
            format="JPEG",
            quality=90,
        )

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return {
            "target_class": target_class,
            "heatmap": f"data:image/jpeg;base64,{encoded}",
        }

    finally:
        forward_handle.remove()
        backward_handle.remove()