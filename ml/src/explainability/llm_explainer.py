import os


def generate_explanation(
    prediction_label,
    confidence,
    real_probability,
    fake_probability,
    retrieved_artifacts,
):
    """Generate a concise local forensic report with a safe fallback."""
    model = os.getenv("OLLAMA_MODEL", "llava")
    artifacts = retrieved_artifacts or []
    artifact_text = "\n".join(
        f"- {item.get('artifact_label', 'Unknown')}: "
        f"{item.get('description', '')}"
        for item in artifacts
    ) or "No matching reference artifacts were retrieved."

    prompt = (
        "You are a forensic media analyst. Write exactly three concise "
        "sentences as a Forensic Analysis Report. State the prediction "
        f"({prediction_label}) and confidence ({confidence:.2%}), compare "
        f"real probability ({real_probability:.2%}) with fake probability "
        f"({fake_probability:.2%}), and interpret these retrieved artifact "
        f"signatures:\n{artifact_text}\n"
        "Avoid claiming certainty and do not use bullet points."
    )

    try:
        from ollama import Client

        client = Client(
            host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        )
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        if hasattr(response, "get"):
            message = response.get("message", {})
            report = message.get("content", "")
        else:
            message = getattr(response, "message", None)
            report = getattr(message, "content", "")

        report = report.strip()
        if report:
            return report
    except Exception:
        pass

    return (
        "Local language-model analysis is unavailable. "
        f"The classifier assessed this media as {prediction_label} with "
        f"{confidence:.2%} confidence. Review the probability scores and "
        "retrieved artifact signatures as supporting evidence only."
    )