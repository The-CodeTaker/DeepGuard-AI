import { useEffect, useState } from "react";

import "./App.css";

import Header from "./components/Header";
import Hero from "./components/Hero";
import UploadPanel from "./components/UploadPanel";
import AnalysisResult from "./components/AnalysisResult";

import { analyzeMedia } from "./services/detectorApi";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [mediaType, setMediaType] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFile = (selectedFile) => {
    if (!selectedFile) return;

    const isVideo = selectedFile.type.startsWith("video/");
    const isImage = selectedFile.type.startsWith("image/");

    if (!isImage && !isVideo) {
      setError("Please select an image or video file.");
      return;
    }

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    const objectUrl = URL.createObjectURL(selectedFile);

    setFile(selectedFile);
    setMediaType(isVideo ? "video" : "image");
    setPreview(objectUrl);
    setResult(null);
    setError("");
  };

  const analyze = async () => {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await analyzeMedia(file, mediaType);
      setResult(data);
    } catch (err) {
      setError(
        err.response?.data?.error ||
          "Unable to analyze the media."
      );
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setFile(null);
    setPreview("");
    setMediaType("");
    setResult(null);
    setError("");
  };

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  return (
    <div className="app">
      <Header />

      <main className="container">
        <Hero />

        <section className="workspace">
          <UploadPanel
            file={file}
            preview={preview}
            mediaType={mediaType}
            loading={loading}
            onFile={handleFile}
            onAnalyze={analyze}
            onReset={reset}
          />

          {error && (
            <div className="error">
              {error}
            </div>
          )}

          <AnalysisResult
            result={result}
            mediaType={mediaType}
          />
        </section>
      </main>
    </div>
  );
}

export default App;