# PromptLab — AI Prompt Checker & Generator

A React + Vite web app that uses Claude to analyze and improve AI prompts.

## Quickstart

```bash
npm install
npm run dev
```

Then open http://localhost:5173

---

## Features (current)

- **Check Prompt mode** — paste any prompt and get:
  - Scores (0–10) for: Clarity, Specificity, Tone, Ambiguity, Overall
  - Hover tooltips on each score ring for dimension feedback
  - Summary assessment
  - List of key improvements made
  - A fully rewritten, optimized version of your prompt (with copy button)

- **Generate Prompt mode** — describe an idea in plain language and get:
  - A polished, ready-to-use prompt
  - Explanation of what was improved
  - Suggested use case

- **Recent history** — click any past entry to restore it

---

## Tech Stack

- React 18 + Vite
- Anthropic Claude API (claude-sonnet-4-20250514) — called directly from browser
- Pure CSS animations / no component library needed

---

## TODO for your team

### Frontend (your part)
- [ ] Hook up to your fine-tuned model once Sean & Brandon have it running
  - Replace the `ANTHROPIC_API_URL` and `model` field in `App.jsx` with your endpoint
  - The JSON schema the app expects is documented in the `SYSTEM_PROMPT` constants
- [ ] Add a loading skeleton instead of the dots animation
- [ ] Add side-by-side diff view (original vs improved prompt)
- [ ] Mobile responsiveness pass
- [ ] Dark/light theme toggle
- [ ] Export results as PDF or copy as markdown
- [ ] Add a history panel with localStorage persistence

### Integration with fine-tuned models
When Sean & Brandon's models are ready, there are two routes:
1. **Replace the API call** in `handleSubmit()` with a call to your hosted endpoint
2. **Keep Claude as fallback** and add a toggle to switch between "Claude" and "Our Model"

### Deployment (Vercel)
```bash
npm run build
# then push to GitHub and connect to Vercel
# set env var: VITE_ANTHROPIC_API_KEY if you move to server-side
```

---

## File Structure

```
src/
  App.jsx       ← everything lives here for now (split out later if needed)
  main.jsx      ← React entry point
index.html
vite.config.js
package.json
```

---

## Notes
- The scoring rubric (clarity, specificity, tone, ambiguity) matches what's described in your proposal
- The `SYSTEM_PROMPT` constants at the top of `App.jsx` define exactly what Claude outputs — easy to tweak
- The JSON schema is strict — if your fine-tuned model outputs something different, update `ResultCard` accordingly
