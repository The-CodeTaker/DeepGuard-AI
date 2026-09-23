export default function ImageExplanation({ heatmap }) {
  if (!heatmap) return null;

  return (
    <div className="explanation">
      <p className="result-label">AI EXPLANATION</p>

      <h4>Areas influencing the assessment</h4>

      <img
        src={heatmap}
        alt="Grad-CAM explanation heatmap"
        className="heatmap"
      />

      <p className="explanation-note">
        Highlighted regions represent areas that contributed
        to the model&apos;s prediction. They are model evidence,
        not a definitive manipulation mask.
      </p>
    </div>
  );
}