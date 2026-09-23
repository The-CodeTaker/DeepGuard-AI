function ProbabilityBar({ label, value }) {
  const percentage = value * 100;

  return (
    <div className="probability">
      <div>
        <span>{label}</span>

        <strong>{percentage.toFixed(1)}%</strong>
      </div>

      <div className="bar">
        <div style={{ width: `${percentage}%` }} />
      </div>
    </div>
  );
}

export default function ProbabilityBars({
  realProbability,
  fakeProbability,
}) {
  return (
    <div className="probabilities">
      <ProbabilityBar
        label="REAL"
        value={realProbability}
      />

      <ProbabilityBar
        label="FAKE"
        value={fakeProbability}
      />
    </div>
  );
}