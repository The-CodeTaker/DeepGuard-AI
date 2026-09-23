from pathlib import Path

from ml.src.inference.predict import extract_features


COLLECTION_NAME = "deepfake_artifacts"
STORE_PATH = Path(__file__).resolve().parents[2] / "chroma_db"

_collection = None


def _get_collection():
    global _collection

    if _collection is not None:
        return _collection

    try:
        import chromadb
    except ImportError:
        return None

    client = chromadb.PersistentClient(path=str(STORE_PATH))
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def index_reference_image(
    image_path,
    artifact_label,
    description,
    artifact_id,
):
    """Index a reference image and its forensic signature."""
    collection = _get_collection()
    if collection is None:
        return False

    embedding = extract_features(image_path).tolist()
    collection.upsert(
        ids=[str(artifact_id)],
        embeddings=[embedding],
        metadatas=[{
            "artifact_label": str(artifact_label),
            "description": str(description),
        }],
    )
    return True


def query_similar_artifacts(embedding, top_k=2):
    """Return nearby artifact descriptions, or [] when none are indexed."""
    collection = _get_collection()
    if collection is None or collection.count() == 0:
        return []

    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    result = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, collection.count()),
        include=["metadatas", "distances"],
    )

    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    return [
        {
            "artifact_label": metadata.get("artifact_label", "Unknown"),
            "description": metadata.get("description", ""),
            "distance": distance,
        }
        for metadata, distance in zip(metadatas, distances)
    ]