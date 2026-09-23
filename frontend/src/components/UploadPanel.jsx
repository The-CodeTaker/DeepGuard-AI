export default function UploadPanel({
  file,
  preview,
  mediaType,
  loading,
  onFile,
  onAnalyze,
  onReset,
}) {
  const handleChange = (event) => {
    onFile(event.target.files[0]);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    onFile(event.dataTransfer.files[0]);
  };

  return (
    <div
      className={`upload-card ${file ? "has-file" : ""}`}
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
    >
      {!preview ? (
        <>
          <div className="upload-icon">↑</div>

          <h3>Upload media</h3>

          <p>
            Drag and drop an image or video here,
            or select a file from your computer.
          </p>

          <label className="select-button">
            Select Media

            <input
              type="file"
              accept="
                image/jpeg,
                image/png,
                image/webp,
                video/mp4,
                video/webm,
                video/quicktime,
                video/x-msvideo
              "
              onChange={handleChange}
              hidden
            />
          </label>

          <span className="formats">
            JPG · PNG · WEBP · MP4 · WEBM · MOV · AVI
          </span>
        </>
      ) : (
        <>
          <div className="preview-wrapper">
            {mediaType === "video" ? (
              <video
                src={preview}
                controls
                className="preview"
              />
            ) : (
              <img
                src={preview}
                alt="Selected media"
                className="preview"
              />
            )}
          </div>

          <p className="filename">{file.name}</p>

          <div className="actions">
            <button
              className="analyze-button"
              onClick={onAnalyze}
              disabled={loading}
            >
              {loading
                ? "Analyzing..."
                : `Analyze ${
                    mediaType === "video" ? "Video" : "Image"
                  }`}
            </button>

            <button
              className="reset-button"
              onClick={onReset}
              disabled={loading}
            >
              Remove
            </button>
          </div>
        </>
      )}
    </div>
  );
}