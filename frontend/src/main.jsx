import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_URL = "http://127.0.0.1:8000";

const examples = [
  {
    name: "Positive",
    text: "I absolutely loved this flight. The service was amazing and everything was perfect.",
  },
  {
    name: "Negative",
    text: "The flight was terrible. I hated the service and the whole experience was awful.",
  },
  {
    name: "Neutral",
    text: "The flight departed at 8 PM and arrived at the airport two hours later.",
  },
  {
    name: "Negation",
    text: "The service was not good and I would not recommend this airline.",
  },
];

function App() {
  const [page, setPage] = useState("overview");

  const [text, setText] = useState(
    "I absolutely love how smooth this experience feels!"
  );

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [history, setHistory] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("sentiment_history")) || [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    localStorage.setItem(
      "sentiment_history",
      JSON.stringify(history)
    );
  }, [history]);

  async function analyze(customText) {
    const value = (customText ?? text).trim();

    if (!value) {
      setError("Please enter some text first.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: value,
        }),
      });

      if (!response.ok) {
        throw new Error("Prediction request failed.");
      }

      const data = await response.json();

      setResult(data);

      const item = {
        id: Date.now(),
        text: value,
        sentiment: data.sentiment,
        confidence: data.confidence,
        scores: data.scores,
        time: new Date().toLocaleTimeString(),
      };

      setHistory((prev) => [item, ...prev].slice(0, 50));
    } catch (err) {
      setError(
        "Could not reach the backend. Make sure FastAPI is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  }

  const stats = useMemo(() => {
    const base = {
      positive: 0,
      neutral: 0,
      negative: 0,
    };

    history.forEach((item) => {
      const key = item.sentiment?.toLowerCase();
      if (base[key] !== undefined) {
        base[key] += 1;
      }
    });

    return base;
  }, [history]);

  const total = history.length;

  function scorePercent(label) {
    if (!result?.scores) return 0;

    const value =
      result.scores[label] ??
      result.scores[label.toLowerCase()] ??
      0;

    return Math.round(value * 100);
  }

  return (
    <div className="app-shell">
      <Sidebar page={page} setPage={setPage} />

      <main className="main-content">

        {page === "overview" && (
          <Overview
            text={text}
            setText={setText}
            result={result}
            analyze={analyze}
            loading={loading}
            error={error}
            scorePercent={scorePercent}
            setPage={setPage}
          />
        )}

        {page === "analyze" && (
          <AnalyzePage
            text={text}
            setText={setText}
            result={result}
            analyze={analyze}
            loading={loading}
            error={error}
            scorePercent={scorePercent}
          />
        )}

        {page === "insights" && (
          <InsightsPage
            history={history}
            stats={stats}
            total={total}
            setHistory={setHistory}
          />
        )}

        {page === "model" && (
          <ModelPage />
        )}

      </main>
    </div>
  );
}

function Sidebar({ page, setPage }) {
  const items = [
    ["overview", "▦", "Overview"],
    ["analyze", "▤", "Analyze"],
    ["insights", "⌁", "Insights"],
    ["model", "◉", "Model"],
  ];

  return (
    <aside className="sidebar">

      <div className="brand">
        <div className="brand-icon">🧠</div>

        <div>
          <div className="brand-title">
            Sentiment<span>AI</span>
          </div>

          <div className="brand-subtitle">
            RNN Intelligence
          </div>
        </div>
      </div>

      <nav className="nav-list">

        {items.map(([key, icon, title]) => (
          <button
            key={key}
            className={`nav-item ${
              page === key ? "active" : ""
            }`}
            onClick={() => setPage(key)}
          >
            <span className="nav-icon">{icon}</span>
            {title}
          </button>
        ))}

      </nav>

      <div className="sidebar-status">
        <div className="status-line">
          <span className="online-dot"></span>
          Model online
        </div>

        <div className="status-small">
          Simple RNN · GloVe 50D
        </div>
      </div>

    </aside>
  );
}

function Overview({
  text,
  setText,
  result,
  analyze,
  loading,
  error,
  scorePercent,
  setPage,
}) {
  return (
    <>
      <header className="hero">

        <div>
          <div className="eyebrow">
            NLP CONTROL CENTER
          </div>

          <h1>
            Understand the emotion
            <br />
            behind every sentence.
          </h1>

          <p>
            Real-time sentiment classification powered by your
            trained Simple RNN model.
          </p>
        </div>

        <button
          className="secondary-button"
          onClick={() => setPage("model")}
        >
          Model details ↗
        </button>

      </header>

      <section className="analysis-grid">

        <AnalysisCard
          text={text}
          setText={setText}
          analyze={analyze}
          loading={loading}
          error={error}
        />

        <PredictionCard
          result={result}
          scorePercent={scorePercent}
        />

      </section>

      <section className="metric-strip">

        <Metric
          label="MODEL"
          value="Simple RNN"
          sub="PyTorch architecture"
        />

        <Metric
          label="VOCABULARY"
          value="12K"
          sub="most frequent tokens"
        />

        <Metric
          label="SEQUENCE"
          value="40"
          sub="tokens per input"
        />

        <Metric
          label="CLASSES"
          value="03"
          sub="negative · neutral · positive"
        />

      </section>

      <section className="how-card">

        <div>
          <div className="eyebrow">HOW IT WORKS</div>

          <h3>
            From raw text to sentiment in milliseconds.
          </h3>
        </div>

        <div className="pipeline">
          <span>Clean text</span>
          <b>→</b>
          <span>Tokenize</span>
          <b>→</b>
          <span>GloVe 50D</span>
          <b>→</b>
          <span>Simple RNN</span>
          <b>→</b>
          <span>Prediction</span>
        </div>

      </section>
    </>
  );
}

function AnalyzePage({
  text,
  setText,
  result,
  analyze,
  loading,
  error,
  scorePercent,
}) {
  return (
    <>
      <PageHeader
        eyebrow="LIVE CLASSIFICATION"
        title="Analyze text"
        description="Enter any English sentence or tweet and let the trained model classify its sentiment."
      />

      <div className="analyze-page-grid">

        <section className="big-card">

          <div className="card-top">
            <div>
              <div className="eyebrow">
                INPUT
              </div>

              <h2>
                Enter text
              </h2>
            </div>

            <span className="character-count">
              {text.length}/280
            </span>
          </div>

          <textarea
            value={text}
            maxLength={280}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type something here..."
          />

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          <div className="examples">

            {examples.map((example) => (
              <button
                key={example.name}
                onClick={() => {
                  setText(example.text);
                  analyze(example.text);
                }}
              >
                {example.name}
              </button>
            ))}

          </div>

          <button
            className="primary-button full"
            onClick={() => analyze()}
            disabled={loading}
          >
            {loading
              ? "Analyzing..."
              : "⚡ Analyze sentiment"}
          </button>

        </section>

        <PredictionCard
          result={result}
          scorePercent={scorePercent}
        />

      </div>

      {result && (
        <section className="details-card">

          <div className="eyebrow">
            API RESPONSE
          </div>

          <div className="response-grid">

            <div>
              <span>Model mode</span>
              <strong>
                {result.model_mode || "trained"}
              </strong>
            </div>

            <div>
              <span>Prediction</span>
              <strong>
                {result.sentiment}
              </strong>
            </div>

            <div>
              <span>Confidence</span>
              <strong>
                {result.confidence}%
              </strong>
            </div>

          </div>

        </section>
      )}
    </>
  );
}

function InsightsPage({
  history,
  stats,
  total,
  setHistory,
}) {
  const percent = (value) =>
    total ? Math.round((value / total) * 100) : 0;

  return (
    <>
      <PageHeader
        eyebrow="ANALYTICS"
        title="Prediction insights"
        description="Review your recent model predictions and understand the sentiment distribution."
      />

      <section className="insight-cards">

        <InsightStat
          className="positive-card"
          title="Positive"
          value={stats.positive}
          percent={percent(stats.positive)}
        />

        <InsightStat
          className="neutral-card"
          title="Neutral"
          value={stats.neutral}
          percent={percent(stats.neutral)}
        />

        <InsightStat
          className="negative-card"
          title="Negative"
          value={stats.negative}
          percent={percent(stats.negative)}
        />

        <InsightStat
          title="Total"
          value={total}
          percent={100}
        />

      </section>

      <section className="history-card">

        <div className="history-header">

          <div>
            <div className="eyebrow">
              RECENT ACTIVITY
            </div>

            <h2>
              Prediction history
            </h2>
          </div>

          {history.length > 0 && (
            <button
              className="danger-button"
              onClick={() => setHistory([])}
            >
              Clear history
            </button>
          )}

        </div>

        {history.length === 0 ? (
          <div className="empty-state">
            No predictions yet. Go to Analyze and run your
            first prediction.
          </div>
        ) : (
          <div className="table-wrap">

            <table>

              <thead>
                <tr>
                  <th>Text</th>
                  <th>Sentiment</th>
                  <th>Confidence</th>
                  <th>Time</th>
                </tr>
              </thead>

              <tbody>

                {history.map((item) => (
                  <tr key={item.id}>

                    <td className="history-text">
                      {item.text}
                    </td>

                    <td>
                      <span
                        className={`sentiment-badge ${item.sentiment}`}
                      >
                        {item.sentiment}
                      </span>
                    </td>

                    <td>
                      {item.confidence}%
                    </td>

                    <td>
                      {item.time}
                    </td>

                  </tr>
                ))}

              </tbody>

            </table>

          </div>
        )}

      </section>
    </>
  );
}

function ModelPage() {
  return (
    <>
      <PageHeader
        eyebrow="MODEL DETAILS"
        title="Simple RNN architecture"
        description="A lightweight recurrent neural network trained for three-class sentiment classification."
      />

      <section className="model-grid">

        <div className="model-main-card">

          <div className="eyebrow">
            ARCHITECTURE
          </div>

          <h2>
            Neural pipeline
          </h2>

          <div className="architecture">

            <ArchitectureStep
              number="01"
              title="Input tokens"
              text="Maximum sequence length: 40 tokens"
            />

            <ArchitectureArrow />

            <ArchitectureStep
              number="02"
              title="Embedding layer"
              text="GloVe 50-dimensional word embeddings"
            />

            <ArchitectureArrow />

            <ArchitectureStep
              number="03"
              title="Simple RNN"
              text="144 hidden units with tanh activation"
            />

            <ArchitectureArrow />

            <ArchitectureStep
              number="04"
              title="Dense layer"
              text="64 neurons + ReLU + dropout"
            />

            <ArchitectureArrow />

            <ArchitectureStep
              number="05"
              title="Output"
              text="Negative · Neutral · Positive"
            />

          </div>

        </div>

        <div className="model-side">

          <InfoCard
            label="FRAMEWORK"
            value="PyTorch"
          />

          <InfoCard
            label="EMBEDDINGS"
            value="GloVe 50D"
          />

          <InfoCard
            label="HIDDEN SIZE"
            value="144"
          />

          <InfoCard
            label="VOCABULARY"
            value="≈ 12,000"
          />

          <InfoCard
            label="SEQUENCE LENGTH"
            value="40"
          />

          <InfoCard
            label="OUTPUT CLASSES"
            value="3"
          />

        </div>

      </section>

      <section className="limitation-card">

        <div className="eyebrow">
          MODEL LIMITATION
        </div>

        <h3>
          Negation and complex context can be challenging.
        </h3>

        <p>
          Simple RNN models process sequential context but may
          struggle with sentences such as “not good” or complex
          linguistic relationships. This is a known limitation of
          the current educational model and can be improved with
          architectures such as LSTM, GRU, or Transformer-based
          models.
        </p>

      </section>
    </>
  );
}

function AnalysisCard({
  text,
  setText,
  analyze,
  loading,
  error,
}) {
  return (
    <section className="analysis-card">

      <div className="eyebrow">
        LIVE ANALYSIS
      </div>

      <h2>
        Analyze text
      </h2>

      <textarea
        value={text}
        maxLength={280}
        onChange={(e) => setText(e.target.value)}
      />

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      <div className="analysis-footer">

        <span>
          {text.length}/280 characters
        </span>

        <button
          className="primary-button"
          onClick={() => analyze()}
          disabled={loading}
        >
          {loading
            ? "Analyzing..."
            : "⚡ Analyze sentiment"}
        </button>

      </div>

    </section>
  );
}

function PredictionCard({
  result,
  scorePercent,
}) {
  const sentiment =
    result?.sentiment?.toLowerCase() || "neutral";

  const confidence =
    result?.confidence ?? 0;

  return (
    <section className={`prediction-card ${sentiment}`}>

      <div className="eyebrow">
        PREDICTION
      </div>

      {!result ? (
        <div className="prediction-empty">

          <h2>
            Ready to analyze
          </h2>

          <p>
            Enter a sentence and run the trained model.
          </p>

        </div>
      ) : (
        <>
          <h2 className="prediction-label">
            {sentiment.toUpperCase()}
          </h2>

          <div className="confidence-row">

            <strong>
              {confidence}%
            </strong>

            <span>
              confidence
            </span>

          </div>

          <Probability
            title="Negative"
            value={scorePercent("negative")}
          />

          <Probability
            title="Neutral"
            value={scorePercent("neutral")}
          />

          <Probability
            title="Positive"
            value={scorePercent("positive")}
          />
        </>
      )}

    </section>
  );
}

function Probability({ title, value }) {
  return (
    <div className="probability">

      <div className="probability-top">

        <span>
          {title}
        </span>

        <b>
          {value}%
        </b>

      </div>

      <div className="progress-track">

        <div
          className="progress-value"
          style={{
            width: `${value}%`,
          }}
        />

      </div>

    </div>
  );
}

function Metric({
  label,
  value,
  sub,
}) {
  return (
    <div className="metric">

      <span>
        {label}
      </span>

      <strong>
        {value}
      </strong>

      <small>
        {sub}
      </small>

    </div>
  );
}

function PageHeader({
  eyebrow,
  title,
  description,
}) {
  return (
    <header className="page-header">

      <div className="eyebrow">
        {eyebrow}
      </div>

      <h1>
        {title}
      </h1>

      <p>
        {description}
      </p>

    </header>
  );
}

function InsightStat({
  title,
  value,
  percent,
  className = "",
}) {
  return (
    <div className={`insight-stat ${className}`}>

      <span>
        {title}
      </span>

      <strong>
        {value}
      </strong>

      <small>
        {percent}% of predictions
      </small>

    </div>
  );
}

function ArchitectureStep({
  number,
  title,
  text,
}) {
  return (
    <div className="architecture-step">

      <span>
        {number}
      </span>

      <div>

        <strong>
          {title}
        </strong>

        <p>
          {text}
        </p>

      </div>

    </div>
  );
}

function ArchitectureArrow() {
  return (
    <div className="architecture-arrow">
      ↓
    </div>
  );
}

function InfoCard({
  label,
  value,
}) {
  return (
    <div className="info-card">

      <span>
        {label}
      </span>

      <strong>
        {value}
      </strong>

    </div>
  );
}

createRoot(
  document.getElementById("root")
).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);