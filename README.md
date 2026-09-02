# PromptLab

A web app that scores and rewrites AI prompts, powered by a fine-tuned local model. Built for CS 568 (UIUC).

## Demo (linked)

[![Watch the demo](https://img.youtube.com/vi/v9mqQYmustw/0.jpg)](https://youtu.be/v9mqQYmustw)

## Features

**Check Prompt mode:** paste a prompt and get scores (0 to 10) for clarity, specificity, tone, and ambiguity, a summary assessment, and a rewritten, optimized version.

**Generate Prompt mode:** describe an idea in plain language and get a ready to use prompt, with an explanation of what was improved.

## Tech Stack

**Frontend:** React 18 + Vite

**Backend:** Flask API serving a fine-tuned [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) with custom LoRA adapters:
- a BPO (black-box prompt optimization) adapter that rewrites prompts
- a HelpSteer-trained rater adapter that scores prompts across clarity, specificity, tone, and ambiguity

Google Gemini is used for auxiliary explanation text on the scored feedback.

## Figures

![Prompt analysis](media/analysis.png)
![Latency chart](media/latency_chart.png)
![Training loss curves](media/training_loss_curves.png)

## Quickstart

```bash
npm install
npm run dev
```

Then open http://localhost:5173
