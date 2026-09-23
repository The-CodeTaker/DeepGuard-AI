export default function VideoAnalysis({ result }) {
  if (!result?.suspicious_frames?.length) {
    return null;
  }

  return (
    <div className="video-analysis">
      <p className="result-label">VIDEO ANALYSIS</p>

      <h4 className="section-title">
        Most suspicious frames
      </h4>

      <div className="suspicious-grid">
        {result.suspicious_frames.map((item) => (
          <div
            className="suspicious-card"
            key={item.frame}
          >
            {item.thumbnail && (
              <img
                src={item.thumbnail}
                alt={`Frame ${item.frame}`}
                className="suspicious-thumbnail"
              />
            )}

            <div className="suspicious-info">
              <div>
                <span>FRAME</span>
                <strong>{item.frame}</strong>
              </div>

              <div>
                <span>TIME</span>
                <strong>
                  {item.timestamp_seconds}s
                </strong>
              </div>

              <div>
                <span>FAKE SCORE</span>
                <strong className="fake-score">
                  {(item.fake_probability * 100).toFixed(1)}%
                </strong>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}