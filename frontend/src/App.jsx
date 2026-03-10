import { useState, useRef, useEffect } from "react";
import "./App.css";

function scorePrompt(prompt) {
  const words = prompt.trim().split(/\s+/);
  const len = words.length;
  const hasQuestion = /\?/.test(prompt);
  const hasDetail = len > 15;
  const vagueWords = ["something", "stuff", "things", "good", "nice", "better", "help me", "tell me"];
  const vagueCount = vagueWords.filter(w => prompt.toLowerCase().includes(w)).length;

  const clarity = Math.min(100, Math.max(20, 40 + len * 2 - vagueCount * 10));
  const specificity = Math.min(100, Math.max(15, hasDetail ? 55 + len : 25 + len * 2));
  const ambiguity = Math.min(100, Math.max(20, 90 - vagueCount * 18 - (hasQuestion ? 0 : 5)));
  const tone = Math.min(100, Math.max(30, 60 + (hasDetail ? 15 : 0) - vagueCount * 5));
  const overall = Math.round((clarity + specificity + ambiguity + tone) / 4);

  const issues = [];
  if (clarity < 60) issues.push("could be clearer");
  if (specificity < 50) issues.push("lacks specifics");
  if (vagueCount > 0) issues.push("contains vague language");

  const feedback = issues.length
    ? `Your prompt ${issues.join(" and ")}. Adding constraints, examples, or a desired output format would significantly improve results.`
    : "Your prompt is reasonably well-formed. Consider adding an output format or role framing to get even more targeted responses.";

  return { scores: { clarity, specificity, ambiguity, tone }, overall, feedback };
}

function improvePrompt(prompt) {
  const trimmed = prompt.trim();
  const lower = trimmed.toLowerCase();

  if (lower.startsWith("write") || lower.startsWith("create")) {
    return `${trimmed} Be specific, well-structured, and aimed at an expert audience. Provide step-by-step details where applicable and format the response clearly.`;
  }
  if (lower.startsWith("explain") || lower.startsWith("what is") || lower.startsWith("how")) {
    return `${trimmed} Assume I have intermediate knowledge on the topic. Use concrete examples and analogies. Keep the response concise but thorough.`;
  }
  return `You are an expert assistant. ${trimmed} Respond with clear, structured, and specific information. Use bullet points or numbered steps where helpful.`;
}

function generatePrompt(idea) {
  const trimmed = idea.trim();
  const prompt = `You are an expert in ${trimmed}. Provide a comprehensive, well-structured response that covers the key concepts, practical applications, and common pitfalls. Use clear headings, bullet points where appropriate, and concrete examples to illustrate your points. Tailor the depth of your explanation for someone with intermediate familiarity with the topic.`;
  const explanation = "This prompt uses role framing, specifies output structure, sets the audience level, and requests concrete examples — all best practices for reliable, high-quality responses.";
  return { prompt, explanation };
}


const SCORE_COLORS = {
  clarity: "#4ade80",
  specificity: "#60a5fa",
  ambiguity: "#f59e0b",
  tone: "#a78bfa",
};

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

function CheckResult({ msg }) {
  return (
    <div className="message assistant">
      <div className="result-card">
        {msg.scores && (
          <div className="result-section">
            <h3 className="section-title">Analysis</h3>
            {Object.entries(msg.scores).map(([key, val]) => (
              <ScoreBar
                key={key}
                label={key.charAt(0).toUpperCase() + key.slice(1)}
                value={val}
                color={SCORE_COLORS[key]}
              />
            ))}
            {msg.overall != null && (
              <div className="overall-score">
                Overall: <strong>{msg.overall}/100</strong>
              </div>
            )}
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
                Improved Prompt
              </h3>
              <CopyButton text={msg.improved} />
            </div>
            <div className="improved-prompt">{msg.improved}</div>
          </div>
        )}
        {!msg.scores && !msg.improved && (
          <div className="result-section">
            <p className="feedback-text" style={{ color: "var(--muted)" }}>
              Backend(s) unavailable — make sure the Flask server is running and
              VITE_ANTHROPIC_API_KEY is set in frontend/.env
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
    "Paste your prompt below and I'll score it across clarity, specificity, ambiguity, and tone — then generate an improved version using our fine-tuned model.",
  generate:
    "Describe your idea or task and I'll craft a polished, ready-to-use prompt for you.",
};

export default function App() {
  const [mode, setMode] = useState("check");
  const [messages, setMessages] = useState([
    { role: "assistant", type: "welcome", content: WELCOME.check },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const switchMode = (m) => {
    setMode(m);
    setMessages([{ role: "assistant", type: "welcome", content: WELCOME[m] }]);
    setInput("");
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    const text = input.trim();
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setLoading(true);

    // Simulate a short thinking delay
    await new Promise((r) => setTimeout(r, 700));

    if (mode === "check") {
      const { scores, overall, feedback } = scorePrompt(text);
      const improved = improvePrompt(text);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", type: "check", scores, overall, feedback, improved },
      ]);
    } else {
      const { prompt, explanation } = generatePrompt(text);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", type: "generate", prompt, explanation },
      ]);
    }

    setLoading(false);
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
    </div>
  );
}
