import { useState, useRef, useEffect } from "react";
import "./App.css";

const SCORE_COLORS = {
  clarity: "#4ade80",
  specificity: "#60a5fa",
  ambiguity: "#f59e0b",
  tone: "#a78bfa",
};

function generateFeedback(scores) {
  const issues = [];
  if (scores.clarity < 50) issues.push("could be clearer");
  if (scores.specificity < 50) issues.push("lacks specifics");
  if (scores.ambiguity < 50) issues.push("contains ambiguity");
  if (scores.tone < 50) issues.push("tone could be improved");
  if (issues.length === 0) {
    return "Your prompt scores well across all dimensions. The optimized version below adds further polish for even better AI responses.";
  }
  return `Your prompt ${issues.join(" and ")}. See the optimized version below for a refined approach.`;
}

function ScoreBar({ label, value, color }) {
  return (
    <div className="score-row">
      <span className="score-label">{label}</span>
      <div className="score-track">
        <div className="score-fill" style={{ width: `${value}%`, background: color }} />
      </div>
      <span className="score-value">{value}</span>
    </div>
  );
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);
  const handle = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <button className="copy-btn" onClick={handle}>
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}

function OverallScore({ value }) {
  const color =
    value >= 75 ? "#4ade80" : value >= 50 ? "#f59e0b" : "#f87171";
  return (
    <div className="overall-score-row">
      <span className="overall-label">Overall</span>
      <div className="overall-ring" style={{ "--ring-color": color }}>
        <span className="overall-number" style={{ color }}>{value}</span>
        <span className="overall-denom">/100</span>
      </div>
    </div>
  );
}

function CheckResult({ msg }) {
  return (
    <div className="message assistant">
      <div className="result-card">
        {msg.scores && (
          <div className="result-section">
            <h3 className="section-title">Dimension Scores</h3>
            {Object.entries(msg.scores).map(([key, val]) => (
              <ScoreBar
                key={key}
                label={key.charAt(0).toUpperCase() + key.slice(1)}
                value={val}
                color={SCORE_COLORS[key]}
              />
            ))}
            {msg.overall != null && <OverallScore value={msg.overall} />}
          </div>
        )}
        {msg.feedback && (
          <div className="result-section">
            <h3 className="section-title">Feedback</h3>
            <p className="feedback-text">{msg.feedback}</p>
          </div>
        )}
        {msg.improved && (
          <div className="result-section">
            <div className="row-between">
              <h3 className="section-title" style={{ marginBottom: 0 }}>
                Optimized Prompt
              </h3>
              <CopyButton text={msg.improved} />
            </div>
            <div className="improved-prompt">{msg.improved}</div>
          </div>
        )}
        {!msg.scores && !msg.improved && (
          <div className="result-section">
            <p className="feedback-text" style={{ color: "var(--muted)" }}>
              Backend unavailable — make sure the Flask server is running on port 5001.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function GenerateResult({ msg }) {
  return (
    <div className="message assistant">
      <div className="result-card">
        <div className="result-section">
          <div className="row-between">
            <h3 className="section-title" style={{ marginBottom: 0 }}>
              Generated Prompt
            </h3>
            <CopyButton text={msg.prompt} />
          </div>
          <div className="improved-prompt">{msg.prompt}</div>
        </div>
        {msg.explanation && (
          <div className="result-section">
            <p className="feedback-text">{msg.explanation}</p>
          </div>
        )}
      </div>
    </div>
  );
}

function Message({ msg }) {
  if (msg.role === "user") {
    return (
      <div className="message user">
        <div className="bubble user-bubble">{msg.content}</div>
      </div>
    );
  }
  if (msg.type === "check") return <CheckResult msg={msg} />;
  if (msg.type === "generate") return <GenerateResult msg={msg} />;
  return (
    <div className="message assistant">
      <div className={`bubble assistant-bubble ${msg.type === "error" ? "error-bubble" : ""}`}>
        {msg.content}
      </div>
    </div>
  );
}

function Thinking() {
  return (
    <div className="message assistant">
      <div className="bubble assistant-bubble thinking">
        <span /><span /><span />
      </div>
    </div>
  );
}

const WELCOME = {
  check:
    "Paste your prompt below and I'll score it across clarity, specificity, ambiguity, and tone — then generate an optimized version using our fine-tuned model.",
  generate:
    "Describe your idea or task and I'll craft a polished, ready-to-use prompt for you.",
};

const INIT_MESSAGES = {
  check: [{ role: "assistant", type: "welcome", content: WELCOME.check }],
  generate: [{ role: "assistant", type: "welcome", content: WELCOME.generate }],
};

export default function App() {
  const [mode, setMode] = useState("check");
  // Separate message histories per mode — preserved when switching tabs
  const [checkMessages, setCheckMessages] = useState(INIT_MESSAGES.check);
  const [generateMessages, setGenerateMessages] = useState(INIT_MESSAGES.generate);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  const messages = mode === "check" ? checkMessages : generateMessages;
  const setMessages = mode === "check" ? setCheckMessages : setGenerateMessages;

  // Whether a prompt has been submitted in the current mode (hide input after first submit)
  const submitted = messages.some((m) => m.role === "user");

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const switchMode = (m) => {
    if (m === mode) return;
    setMode(m);
    setInput("");
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    const text = input.trim();
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setLoading(true);

    if (mode === "check") {
      try {
        const [dimsRes, optimizeRes] = await Promise.all([
          fetch("/api/model/score_dims", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt: text }),
          }),
          fetch("/api/model/optimize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt: text }),
          }),
        ]);

        if (!dimsRes.ok || !optimizeRes.ok) throw new Error("Backend error");

        const dims = await dimsRes.json();
        const opt = await optimizeRes.json();

        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            type: "check",
            scores: dims.scores,
            overall: dims.overall,
            feedback: generateFeedback(dims.scores),
            improved: opt.optimized,
          },
        ]);
      } catch {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            type: "error",
            content:
              "Could not reach the backend. Make sure the Flask server is running on port 5001.",
          },
        ]);
      }
    } else {
      try {
        const res = await fetch("/api/model/optimize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: text }),
        });
        if (!res.ok) throw new Error("Backend error");
        const data = await res.json();
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            type: "generate",
            prompt: data.optimized,
            explanation:
              "This optimized prompt was generated by our fine-tuned model to improve clarity, specificity, and effectiveness for AI systems.",
          },
        ]);
      } catch {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            type: "error",
            content:
              "Could not reach the backend. Make sure the Flask server is running on port 5001.",
          },
        ]);
      }
    }

    setLoading(false);
  };

  const handleNewConversation = () => {
    setMessages(INIT_MESSAGES[mode]);
    setInput("");
  };

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <span className="logo-mark">✦</span> PromptLab
        </div>
        <nav className="mode-toggle">
          <button
            className={`mode-btn ${mode === "check" ? "active" : ""}`}
            onClick={() => switchMode("check")}
          >
            Check Prompt
          </button>
          <button
            className={`mode-btn ${mode === "generate" ? "active" : ""}`}
            onClick={() => switchMode("generate")}
          >
            Generate Prompt
          </button>
        </nav>
      </header>

      <main className="chat-area">
        {messages.map((msg, i) => (
          <Message key={i} msg={msg} />
        ))}
        {loading && <Thinking />}
        <div ref={bottomRef} />
      </main>

      {submitted && !loading ? (
        <div className="input-area">
          <div className="new-convo-wrap">
            <button className="new-convo-btn" onClick={handleNewConversation}>
              + New Conversation
            </button>
          </div>
        </div>
      ) : !submitted ? (
        <div className="input-area">
          <div className="input-wrapper">
            <textarea
              className="chat-input"
              placeholder={
                mode === "check"
                  ? "Paste your prompt here…"
                  : "Describe your idea or task…"
              }
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              rows={3}
            />
            <button
              className="send-btn"
              onClick={handleSend}
              disabled={loading || !input.trim()}
              aria-label="Send"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </div>
          <p className="hint">
            {mode === "check"
              ? "Shift+Enter for new line · Check Prompt mode"
              : "Shift+Enter for new line · Generate Prompt mode"}
          </p>
        </div>
      ) : null}
    </div>
  );
}
