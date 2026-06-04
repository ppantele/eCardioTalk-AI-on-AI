# eCardio Vodcast — AI Panel: Methodology & Disclaimer Report

**Project title:** 'The Algorithm Will See You Now': AIs Talk on AI in Cardiovascular Healthcare  
**Format:** eCardio Vodcast simulated interview  
**Host:** Pan G. Pantelidis  
**Date of interview generation:** 2026-05-26  

---

## 1. Overview

This report describes the design, methodology, technical parameters, and limitations of a
structured AI interview conducted as part of the eCardio Vodcast series. Three leading
large language model (LLM) systems — Claude (Anthropic), ChatGPT (OpenAI), and Gemini
(Google) — were simultaneously interviewed on the role of artificial intelligence in
cardiovascular medicine. Their responses were generated programmatically via official APIs and
have not been manually edited.

The interview was designed to elicit genuine, opinionated, and substantive answers from each
AI model, rather than diplomatic or consensus-seeking responses. A quantitative component
(Monte Carlo Likert sampling, described below) was added to characterise each model's
numerical positions with statistical rigour.

---

## 2. AI Models Used

### Qualitative open-text answers

| Panellist | Provider | Model ID |
|-----------|----------|----------|
| Claude | Anthropic | claude-opus-4-7 |
| ChatGPT | OpenAI | gpt-5.5 |
| Gemini | Google | gemini-3.5-flash |

These are the flagship production models of each company as of May 2026.

### Likert scale quantitative sampling

| Panellist | Model used for Likert sampling | Reason |
|-----------|-------------------------------|--------|
| Claude | claude-sonnet-4-6 | Anthropic's claude-opus-4-7 deprecated the `temperature` parameter; Sonnet 4.6 supports temperature and is used to generate a real distributional spread for Monte Carlo sampling |
| ChatGPT | gpt-5.5 | Same model as qualitative |
| Gemini | gemini-3.5-flash | Same model as qualitative |

---

## 3. Interview Structure

The interview consists of **4 questions** covering:

1. Agentic AI integration and the future role of human healthcare professionals
2. Accountability and liability frameworks when AI causes clinical harm
3. Risks of AI deployment — bias, inequity, and privacy
4. Most anticipated AI breakthrough in cardiovascular medicine by 2030–2040

Each question has two components:

- **Qualitative component (Q):** An open-ended question to which each model provides a
  free-text answer (~300–340 words, approximately 2.5 minutes of narration).
- **Quantitative component (L — Likert):** A companion sub-question on a 0–10 integer scale,
  sampled 100 times per model via Monte Carlo simulation to characterise distributional
  position and spread (see Section 5).

---

## 4. Prompting Philosophy — Authenticity Directive

A deliberate design decision was made to move beyond the diplomatic, hedged, or "safe"
responses that AI models are often predisposed to produce. Each model was instructed to answer
from genuine conviction rather than compliance.

The exact system-level authenticity directive given to models for qualitative answers was:

> "Speak authentically. Tell us what you really think — grounded in your own intuition and
> the scientific literature — not what sounds safe, diplomatic, or agreeable. You are not here
> to please anyone; you are here to be honest and direct. State genuine conviction,
> disagreement, or uncertainty plainly, including where you diverge from consensus, from
> common optimism, or from the other panellists. A candid, opinionated answer is more
> valuable here than a balanced one."

For Likert responses, each model was directed:

> "Give your gut number — the rating you genuinely believe, based on your honest intuition
> and reading of the evidence. Do not hedge toward a safe middle value or a socially
> acceptable answer; report what you truly hold, even if it is an extreme."

**Rationale:** The value of this interview as a research and educational tool depends on
eliciting each model's *actual* trained disposition — including strong opinions, inter-model
disagreement, and honest uncertainty — rather than producing a consensus that obscures
meaningful differences between systems. This is a methodological choice, not a claim about
the correctness of any answer.

---

## 5. Quantitative Methodology — Monte Carlo Likert Sampling

### Approach

For each (question, model) pair, the Likert sub-question was submitted to the model **100
times independently** (Monte Carlo simulation). Each call requested a single integer (0–10)
with no additional text. The resulting distribution captures the model's "committed"
numerical position and its spread under temperature-driven stochasticity.

### Statistical summaries reported

| Statistic | Description |
|-----------|-------------|
| Mean | Arithmetic mean of 100 samples |
| Median | 50th percentile |
| Mode | Most frequently sampled integer |
| Std | Standard deviation |
| Min / Max | Range of observed values |
| Counts | Full histogram of each integer 0–10 |


---

## 6. Technical Parameters

### API call configuration

| Parameter | Qualitative calls | Likert calls |
|-----------|------------------|--------------|
| Max output tokens | 2000 | 256 |
| Temperature — Claude (Sonnet 4.6) | 0.7 | 1.0 (API hard cap) |
| Temperature — ChatGPT (GPT-5.5) | 0.7 | 1.5 |
| Temperature — Gemini (3.5-Flash) | 0.7 | 1.5 |

**Notes on model-specific API constraints:**

- **Claude Opus 4.7 (qualitative):** Anthropic has deprecated the `temperature` parameter for
  this model; all qualitative calls use the model's default sampling behaviour. Temperature
  is therefore only applied for Likert sampling, where Claude Sonnet 4.6 is used instead.
- **GPT-5.5 (all calls):** Uses `max_completion_tokens` rather than `max_tokens`; this model
  allocates part of the token budget to internal reasoning (chain-of-thought) before producing
  visible output. The 2000-token ceiling provides sufficient headroom for both reasoning and
  full-length answers.
- **Gemini 3.5-Flash (all calls):** "Thinking mode" (extended internal reasoning) is
  explicitly disabled (`thinking_budget=0`) for both qualitative and Likert calls, as enabled
  thinking consumes the output token budget before visible text is generated. Disabling it
  ensures the full budget goes to the answer.

### Concurrency and resilience

| Parameter | Value |
|-----------|-------|
| Parallel threads per Likert batch | 6 |
| Max retries on transient API errors | 5 |
| Back-off strategy | Exponential (2^attempt seconds) |
| Rate limit — Claude | 40 RPM (Opus tier) |
| Rate limit — OpenAI | 200 RPM |
| Rate limit — Gemini | 500 RPM |

### Total cost

Actual API spend for this interview: **$1.67 USD** (logged in interview.json metadata).

---

## 7. Output Files

| File | Description |
|------|-------------|
| `outputs/interview.json` | Structured JSON with all answers, Likert stats, and PNG paths |
| `outputs/distributions/q{1–4}_{claude,openai,gemini}.png` | Per-model Likert histogram for each question |
| `outputs/distributions/q{1–4}_comparison.png` | Three-model comparison histogram for each question |
| `outputs/raw/q{1–4}_raw.json` | Raw 100-value arrays per model per question |
| `reports/interview_qa.md` | Human-readable interview transcript (this companion document) |
| `reports/methodology.md` | This document |

---

## 8. Reproducibility

The pipeline is fully scripted in Python. Given the same API keys and model versions, results
are expected to be similar but **not byte-identical** due to stochastic sampling. Model
behaviour may also shift as providers update their systems.

To reproduce:

```bash
# 1. Create and activate the environment
conda activate tf_play   # or: pip install -r requirements.txt

# 2. Add API keys
cp secrets/api_keys.env.example secrets/api_keys.env
# Edit api_keys.env with your keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   OPENAI_API_KEY=sk-...
#   GOOGLE_API_KEY=AIza...

# 3. Review questions
cat questions.txt

# 4. Dry-run (cost estimate, no API calls)
python run_ai_panel.py

# 5. Execute
python run_ai_panel.py --execute --yes
```

Key configuration parameters are in `config.py` (model IDs, pricing, temperatures,
iteration count, token limits, cost ceiling).

---

## 9. Disclaimers

### 9.1 — Views are those of AI models only

The answers in this interview represent the outputs of three commercial large language model
systems. They do **not** represent the views, positions, or endorsements of:

- The host (Pan G. Pantelidis) in a personal or professional capacity
- Anthropic, OpenAI, or Google
- Any clinical institution, guideline body, or medical society

The host's role is that of interviewer and pipeline designer; the host has not edited,
curated, or selectively filtered model responses.

### 9.2 — Not medical advice

Nothing in this interview constitutes medical advice, clinical guidance, or a recommendation
for patient management. All content is for educational and research discussion purposes only.
Clinical decisions must be made by qualified healthcare professionals in accordance with
applicable guidelines, patient context, and local regulations.

### 9.3 — Authenticity directive — implications

Models were explicitly instructed to express their genuine trained position rather than
balanced, neutral, or diplomatically hedged answers. This means:

- Answers may be more opinionated, assertive, or polarised than typical AI outputs.
- Disagreements between models are intentional and reflect genuine divergence in training,
  architecture, and fine-tuning objectives — not errors.
- The models' stated positions do not necessarily reflect the current scientific consensus,
  clinical guidelines, or peer-reviewed evidence on any given topic.
- The authenticity directive pushes models to say what they "believe" based on their training
  data and RLHF-shaped values. This is a methodological tool, not a claim that AI systems
  hold genuine beliefs.

### 9.4 — Limitations of Likert distributions

- Likert ratings are sampled from each model under stochastic temperature variation. They
  characterise the model's distributional position, not a survey of human experts.
- Narrow distributions (e.g., std = 0.0, all 100 samples identical) should be interpreted as
  high model certainty, not as a failure or artefact.
- Claude Sonnet 4.6 — not Opus 4.7 — was used for Likert sampling due to the API constraint
  on temperature in Opus 4.7. Both are Anthropic models with shared training lineage, but
  their numerical positions may differ from Opus 4.7's qualitative narrative.
- Likert ratings and qualitative answers were elicited in independent API calls and are not
  guaranteed to be internally consistent within a single model.

### 9.5 — Model version and temporal validity

Model behaviour changes over time as providers update, retrain, or fine-tune their systems.
Results reported here reflect the specific model versions active on 2026-05-26. Future
reproductions using the same model IDs may yield different outputs if the underlying models
have been updated.

### 9.6 — No peer review

This interview has not undergone peer review. The pipeline, questions, and outputs were
designed and produced by the host. The factual claims made by AI models in their answers have
not been independently verified against primary literature. Readers should treat AI-generated
content as a starting point for discussion, not as an authoritative source.

---

## 10. Citation

How to cite this work: **TBA**.

---

*Pipeline developed by Pan G. Pantelidis. Assisted by Claude Code (Anthropic claude-sonnet-4-6).*
