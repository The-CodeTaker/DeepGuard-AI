export default function Header() {
  return (
    <header className="header">
      <div className="brand">
        <div className="logo">DG</div>

        <div>
          <h1>DeepGuard AI</h1>
          <p>AI-Powered Deepfake Detection</p>
        </div>
      </div>

      <div className="status">
        <span />
        AI Engine Online
      </div>
    </header>
  );
}