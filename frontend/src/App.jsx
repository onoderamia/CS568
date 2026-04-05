import { useState, useRef, useEffect } from "react";
import "./App.css";

/** Display order requested: coherence → complexity → helpfulness → correctness → verbosity */
const DIMENSION_ORDER = [
  "coherence",
  "complexity",
  "helpfulness",
  "correctness",
  "verbosity",
];

const SCORE_COLORS = {
  coherence: "#60a5fa",
  complexity: "#c084fc",
  helpfulness: "#4ade80",
  correctness: "#f59e0b",
  verbosity: "#94a3b8",
};

const DIMENSION_LABELS = {
  coherence: "Coherence",
  complexity: "Complexity",
  helpfulness: "Helpfulness",
  correctness: "Correctness",
  verbosity: "Verbosity",
};

/** value: integer 1–5 */
function ScoreBarFive({ dimKey, value, color }) {
  const pct = ((Number(value) - 1) / 4) * 100;
  const label = DIMENSION_LABELS[dimKey] || dimKey;
  return (
    <div className="score-row score-row--five">
      <span className="score-label">{label}</span>
      <div className="score-track" aria-hidden>
        <div
          className="score-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="score-value score-value--five">
        {value}
        <span className="score-out-of">/5</span>
      </span>
    </div>
  );
}

function DimensionFeedbackBody({ text }) {
  if (!text || !String(text).trim()) {
    return <p className="dimension-card__feedback">—</p>;
  }
  const lines = String(text)
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  const bulletLines = lines.filter(
    (l) => l.startsWith("- ") || l.startsWith("• "),
  );
  if (bulletLines.length >= 1) {
    return (
      <ul className="dimension-card__bullets">
        {bulletLines.map((l, i) => (
          <li key={i} className="dimension-card__bullet">
            {l.replace(/^[-•]\s+/, "")}
          </li>
        ))}
      </ul>
    );
  }
  return <p className="dimension-card__feedback">{text}</p>;
}

function DimensionFeedbackCard({ dimKey, score, explanation, color }) {
  const label = DIMENSION_LABELS[dimKey] || dimKey;
  return (
    <article className="dimension-card" style={{ "--dim-accent": color }}>
      <header className="dimension-card__head">
        <h4 className="dimension-card__title">{label}</h4>
        <span className="dimension-card__badge" title="Score 1–5">
          {score}
          <span className="dimension-card__badge-denom">/5</span>
        </span>
      </header>
      <DimensionFeedbackBody text={explanation} />
    </article>
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

/** Match backend `_split_into_sentences` (sentence-ending punctuation + whitespace). */
function splitOptimizedSentences(text) {
  const t = (text || "").trim();
  if (!t) return [];
  return t.split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter(Boolean);
}

/** value: mean dimension score 1–5 */
function OverallScore({ value }) {
  const n = Number(value);
  const color = n >= 4 ? "#4ade80" : n >= 3 ? "#f59e0b" : "#f87171";
  return (
    <div className="overall-score-row">
      <span className="overall-label">Overall (avg)</span>
      <div className="overall-ring" style={{ "--ring-color": color }}>
        <span className="overall-number" style={{ color }}>
          {value}
        </span>
        <span className="overall-denom">/5</span>
      </div>
    </div>
  );
}

function CheckResult({ msg, msgIndex, setMessages }) {
  const [selectedSentenceIdx, setSelectedSentenceIdx] = useState(null);
  const [refineBusy, setRefineBusy] = useState(false);
  const [refineError, setRefineError] = useState(null);
  const [changedIndices, setChangedIndices] = useState(new Set());

  const orderedDims =
    msg.scores && DIMENSION_ORDER.filter((d) => msg.scores[d] != null);
  const sentences = splitOptimizedSentences(msg.improved);
  const userPromptForRefine = (msg.userPrompt || "").trim();

  const patchImprovedOnly = (optimized) => {
    setMessages((prev) => {
      const next = [...prev];
      const cur = next[msgIndex];
      if (cur && cur.type === "check") {
        next[msgIndex] = { ...cur, improved: optimized };
      }
      return next;
    });
  };

  const callRefine = async (body, sentenceIdx) => {
    setRefineError(null);
    setRefineBusy(true);
    setChangedIndices(new Set());
    const oldSentences = splitOptimizedSentences(msg.improved);
    try {
      const res = await fetch("/api/model/refine_optimized", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.error || "Refine request failed");
      }
      if (typeof data.optimized !== "string") {
        throw new Error("Invalid refine response");
      }
      patchImprovedOnly(data.optimized);
      setSelectedSentenceIdx(null);

      const newSentences = splitOptimizedSentences(data.optimized);
      if (body.mode === "full") {
        setChangedIndices(new Set(newSentences.map((_, i) => i)));
      } else {
        const changed = new Set();
        for (let i = 0; i < newSentences.length; i++) {
          if (i >= oldSentences.length || newSentences[i] !== oldSentences[i]) {
            changed.add(i);
          }
        }
        if (changed.size === 0 && sentenceIdx != null) {
          changed.add(sentenceIdx);
        }
        setChangedIndices(changed);
      }
    } catch (e) {
      setRefineError(e.message || String(e));
    } finally {
      setRefineBusy(false);
    }
  };

  const onSentenceAction = (action) => {
    if (
      selectedSentenceIdx == null ||
      !msg.improved ||
      !userPromptForRefine
    ) {
      return;
    }
    callRefine(
      {
        mode: "sentence",
        original_user_prompt: userPromptForRefine,
        full_optimized: msg.improved,
        sentence_index: selectedSentenceIdx,
        action,
      },
      selectedSentenceIdx,
    );
  };

  const onFullRefresh = () => {
    if (!msg.improved || !userPromptForRefine) return;
    const initial =
      msg.initialOptimized != null && String(msg.initialOptimized).trim()
        ? String(msg.initialOptimized).trim()
        : msg.improved;
    callRefine({
      mode: "full",
      original_user_prompt: userPromptForRefine,
      full_optimized: msg.improved,
      initial_optimized: initial,
    });
  };

  return (
    <div className="message assistant">
      <div className="result-card">
        {msg.scores && orderedDims?.length > 0 && (
          <div className="result-section">
            <h3 className="section-title">Quality scores (1–5)</h3>
            <p className="section-hint">
              Scores reflect your prompt's quality across key dimensions.
              Feedback explains why it scored that way and how to improve it.
            </p>
            {orderedDims.map((key) => (
              <ScoreBarFive
                key={key}
                dimKey={key}
                value={msg.scores[key]}
                color={SCORE_COLORS[key] || "#7c6af5"}
              />
            ))}
            {msg.overall != null && <OverallScore value={msg.overall} />}
          </div>
        )}
        {msg.explanations && orderedDims?.length > 0 && (
          <div className="result-section result-section--feedback-grid">
            <h3 className="section-title">Feedback by dimension</h3>
            <div className="dimension-grid">
              {orderedDims.map((key) => (
                <DimensionFeedbackCard
                  key={key}
                  dimKey={key}
                  score={msg.scores[key]}
                  explanation={msg.explanations[key]}
                  color={SCORE_COLORS[key] || "#7c6af5"}
                />
              ))}
            </div>
          </div>
        )}
        {msg.draft_reply && (
          <details className="draft-details">
            <summary className="draft-details__summary">
              Draft reply used for scoring
            </summary>
            <div className="draft-details__body">{msg.draft_reply}</div>
          </details>
        )}
        {msg.improved && (
          <div className="result-section">
            <div className="row-between">
              <h3 className="section-title" style={{ marginBottom: 0 }}>
                Optimized Prompt
              </h3>
              <CopyButton text={msg.improved} />
            </div>
            <p className="refine-hint">
              Click a sentence, then elaborate or shorten it. Or generate a
              wholly new optimized prompt (scores above stay the same).
            </p>
            <div className="improved-prompt improved-prompt--interactive">
              {sentences.length > 0 ? (
                sentences.map((s, i) => (
                  <span
                    key={i}
                    className={[
                      "opt-sentence",
                      selectedSentenceIdx === i && "opt-sentence--selected",
                      changedIndices.has(i) && "opt-sentence--changed",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                    role="button"
                    tabIndex={0}
                    onClick={() => {
                      if (changedIndices.size > 0) {
                        setChangedIndices(new Set());
                        setSelectedSentenceIdx(null);
                        return;
                      }
                      setSelectedSentenceIdx((prev) => (prev === i ? null : i));
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        if (changedIndices.size > 0) {
                          setChangedIndices(new Set());
                          setSelectedSentenceIdx(null);
                          return;
                        }
                        setSelectedSentenceIdx((prev) => (prev === i ? null : i));
                      }
                    }}
                  >
                    {s}
                    {i < sentences.length - 1 ? " " : ""}
                  </span>
                ))
              ) : (
                <span>{msg.improved}</span>
              )}
            </div>
            <div className="refine-toolbar">
              <div className="refine-actions">
                <button
                  type="button"
                  className="refine-btn"
                  disabled={
                    refineBusy ||
                    selectedSentenceIdx == null ||
                    !userPromptForRefine
                  }
                  onClick={() => onSentenceAction("elaborate")}
                >
                  Elaborate
                </button>
                <button
                  type="button"
                  className="refine-btn"
                  disabled={
                    refineBusy ||
                    selectedSentenceIdx == null ||
                    !userPromptForRefine
                  }
                  onClick={() => onSentenceAction("concise")}
                >
                  More concise
                </button>
              </div>
              <button
                type="button"
                className="refine-btn refine-btn--primary"
                disabled={refineBusy || !msg.improved || !userPromptForRefine}
                onClick={onFullRefresh}
              >
                New full optimized prompt
              </button>
            </div>
            {!userPromptForRefine && (
              <p className="refine-error">
                Refine unavailable for this message (missing original prompt).
                Start a new check to use refinement.
              </p>
            )}
            {refineError && <p className="refine-error">{refineError}</p>}
          </div>
        )}
        {!msg.scores && !msg.improved && (
          <div className="result-section">
            <p className="feedback-text" style={{ color: "var(--muted)" }}>
              Backend unavailable — make sure the Flask server is running on
              port 5001.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function GenerateResult({ msg, msgIndex, setMessages }) {
  const [selectedSentenceIdx, setSelectedSentenceIdx] = useState(null);
  const [refineBusy, setRefineBusy] = useState(false);
  const [refineError, setRefineError] = useState(null);
  const [changedIndices, setChangedIndices] = useState(new Set());

  const sentences = splitOptimizedSentences(msg.prompt);
  const userIdea = (msg.userIdea || "").trim();

  const patchPromptOnly = (generated) => {
    setMessages((prev) => {
      const next = [...prev];
      const cur = next[msgIndex];
      if (cur && cur.type === "generate") {
        next[msgIndex] = { ...cur, prompt: generated };
      }
      return next;
    });
  };

  const callRefine = async (body, sentenceIdx) => {
    setRefineError(null);
    setRefineBusy(true);
    setChangedIndices(new Set());
    const oldSentences = splitOptimizedSentences(msg.prompt);
    try {
      const res = await fetch("/api/model/refine_generated", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Refine request failed");
      if (typeof data.generated !== "string") throw new Error("Invalid refine response");
      patchPromptOnly(data.generated);
      setSelectedSentenceIdx(null);

      const newSentences = splitOptimizedSentences(data.generated);
      if (body.mode === "full") {
        setChangedIndices(new Set(newSentences.map((_, i) => i)));
      } else {
        const changed = new Set();
        for (let i = 0; i < newSentences.length; i++) {
          if (i >= oldSentences.length || newSentences[i] !== oldSentences[i]) changed.add(i);
        }
        if (changed.size === 0 && sentenceIdx != null) changed.add(sentenceIdx);
        setChangedIndices(changed);
      }
    } catch (e) {
      setRefineError(e.message || String(e));
    } finally {
      setRefineBusy(false);
    }
  };

  const onSentenceAction = (action) => {
    if (selectedSentenceIdx == null || !msg.prompt || !userIdea) return;
    callRefine(
      {
        mode: "sentence",
        original_user_idea: userIdea,
        full_generated: msg.prompt,
        sentence_index: selectedSentenceIdx,
        action,
      },
      selectedSentenceIdx,
    );
  };

  const onFullRefresh = () => {
    if (!msg.prompt || !userIdea) return;
    const initial =
      msg.initialPrompt != null && String(msg.initialPrompt).trim()
        ? String(msg.initialPrompt).trim()
        : msg.prompt;
    callRefine({
      mode: "full",
      original_user_idea: userIdea,
      full_generated: msg.prompt,
      initial_generated: initial,
    });
  };

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
          <p className="refine-hint">
            Click a sentence, then elaborate or shorten it. Or generate a
            wholly new prompt from your idea.
          </p>
          <div className="improved-prompt improved-prompt--interactive">
            {sentences.length > 0 ? (
              sentences.map((s, i) => (
                <span
                  key={i}
                  className={[
                    "opt-sentence",
                    selectedSentenceIdx === i && "opt-sentence--selected",
                    changedIndices.has(i) && "opt-sentence--changed",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                  role="button"
                  tabIndex={0}
                  onClick={() => {
                    if (changedIndices.size > 0) {
                      setChangedIndices(new Set());
                      setSelectedSentenceIdx(null);
                      return;
                    }
                    setSelectedSentenceIdx((prev) => (prev === i ? null : i));
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      if (changedIndices.size > 0) {
                        setChangedIndices(new Set());
                        setSelectedSentenceIdx(null);
                        return;
                      }
                      setSelectedSentenceIdx((prev) => (prev === i ? null : i));
                    }
                  }}
                >
                  {s}
                  {i < sentences.length - 1 ? " " : ""}
                </span>
              ))
            ) : (
              <span>{msg.prompt}</span>
            )}
          </div>
          <div className="refine-toolbar">
            <div className="refine-actions">
              <button
                type="button"
                className="refine-btn"
                disabled={refineBusy || selectedSentenceIdx == null || !userIdea}
                onClick={() => onSentenceAction("elaborate")}
              >
                Elaborate
              </button>
              <button
                type="button"
                className="refine-btn"
                disabled={refineBusy || selectedSentenceIdx == null || !userIdea}
                onClick={() => onSentenceAction("concise")}
              >
                More concise
              </button>
            </div>
            <button
              type="button"
              className="refine-btn refine-btn--primary"
              disabled={refineBusy || !msg.prompt || !userIdea}
              onClick={onFullRefresh}
            >
              New generated prompt
            </button>
          </div>
          {!userIdea && (
            <p className="refine-error">
              Refine unavailable for this message (missing original idea).
              Start a new conversation to use refinement.
            </p>
          )}
          {refineError && <p className="refine-error">{refineError}</p>}
        </div>
      </div>
    </div>
  );
}


function Message({ msg, msgIndex, setMessages }) {
  if (msg.role === "user") {
    return (
      <div className="message user">
        <div className="bubble user-bubble">{msg.content}</div>
      </div>
    );
  }
  if (msg.type === "check")
    return (
      <CheckResult msg={msg} msgIndex={msgIndex} setMessages={setMessages} />
    );
  if (msg.type === "generate") return <GenerateResult msg={msg} msgIndex={msgIndex} setMessages={setMessages} />;
  return (
    <div className="message assistant">
      <div
        className={`bubble assistant-bubble ${msg.type === "error" ? "error-bubble" : ""}`}
      >
        {msg.content}
      </div>
    </div>
  );
}

function Thinking() {
  return (
    <div className="message assistant">
      <div className="bubble assistant-bubble thinking">
        <span />
        <span />
        <span />
      </div>
    </div>
  );
}

const WELCOME = {
  check:
    "Paste your prompt and I'll score it on helpfulness, correctness, coherence, complexity, and verbosity (1–5), explain why your prompt received each score, then show an optimized version.",
  generate:
    "Describe your idea, task, or project request and I'll turn it into a polished, ready-to-use prompt.",
};

const INIT_MESSAGES = {
  check: [{ role: "assistant", type: "welcome", content: WELCOME.check }],
  generate: [{ role: "assistant", type: "welcome", content: WELCOME.generate }],
};

export default function App() {
  const [mode, setMode] = useState("check");
  // Separate message histories per mode — preserved when switching tabs
  const [checkMessages, setCheckMessages] = useState(INIT_MESSAGES.check);
  const [generateMessages, setGenerateMessages] = useState(
    INIT_MESSAGES.generate,
  );
  const [input, setInput] = useState("");
  const [checkLoading, setCheckLoading] = useState(false);
  const [generateLoading, setGenerateLoading] = useState(false);
  const bottomRef = useRef(null);

  const messages = mode === "check" ? checkMessages : generateMessages;
  const setMessages = mode === "check" ? setCheckMessages : setGenerateMessages;
  const loading = mode === "check" ? checkLoading : generateLoading;
  const setLoading = mode === "check" ? setCheckLoading : setGenerateLoading;

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
            explanations: dims.explanations,
            draft_reply: dims.draft_reply,
            userPrompt: text,
            initialOptimized: opt.optimized,
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
    } else if (mode === "generate") {
      try {
        const res = await fetch("/api/model/generate_task_prompt", {
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
            prompt: data.generated_prompt,
            initialPrompt: data.generated_prompt,
            userIdea: text,
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
          <Message key={i} msg={msg} msgIndex={i} setMessages={setMessages} />
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
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
              >
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
