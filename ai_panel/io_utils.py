"""
I/O utilities: secrets loading, questions.txt parsing, JSON export,
and --init scaffolding.
"""

import json
import os
import textwrap
from pathlib import Path

from dotenv import load_dotenv

import config


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

def load_secrets() -> dict[str, str]:
    """
    Load API keys from secrets/api_keys.env.

    Returns a dict mapping provider name to key string.
    Raises FileNotFoundError with clear instructions if the file is missing.
    Raises KeyError with the missing variable name if a required key is absent.
    """
    if not config.SECRETS_FILE.exists():
        raise FileNotFoundError(
            f"\nSecrets file not found: {config.SECRETS_FILE}\n\n"
            "To create it:\n"
            "  1. Run:  python run_ai_panel.py --init\n"
            "     This drops secrets/api_keys.env.example with the required format.\n"
            "  2. Copy it:  cp secrets/api_keys.env.example secrets/api_keys.env\n"
            "  3. Fill in your real API keys.\n"
        )
    load_dotenv(dotenv_path=config.SECRETS_FILE)

    keys: dict[str, str] = {}
    required = {
        "claude": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
    }
    for provider, env_var in required.items():
        value = os.environ.get(env_var)
        if not value:
            raise KeyError(
                f"\nMissing API key for {provider}: {env_var} not set in "
                f"{config.SECRETS_FILE}\n"
                f"Add the line:  {env_var}=your-key-here\n"
            )
        keys[provider] = value
    return keys


# ---------------------------------------------------------------------------
# Questions parsing
# ---------------------------------------------------------------------------

def parse_questions(path: Path = config.QUESTIONS_FILE) -> list[dict]:
    """
    Parse questions.txt and return a list of question dicts.

    Format::
        Q: <open-text question>
        L: <0-10 Likert sub-question>

    Lines beginning with # are ignored (use this to comment out questions).
    Blank lines are ignored.
    A Q: line without a following L: raises ValueError.

    Returns:
        List of {"question": str, "likert": str} dicts, in file order.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"\nQuestions file not found: {path}\n"
            "Run:  python run_ai_panel.py --init  to generate a starter file.\n"
        )

    questions: list[dict] = []
    pending_q: str | None = None

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("Q:"):
            if pending_q is not None:
                raise ValueError(
                    f"questions.txt line {lineno}: new Q: found before an L: "
                    f"for the previous question.\nPrevious Q: {pending_q!r}"
                )
            pending_q = line[2:].strip().strip("\"'")
        elif line.upper().startswith("L:"):
            if pending_q is None:
                raise ValueError(
                    f"questions.txt line {lineno}: L: found without a preceding Q:"
                )
            questions.append({"question": pending_q, "likert": line[2:].strip().strip("\"'")})
            pending_q = None

    if pending_q is not None:
        raise ValueError(
            f"questions.txt: final Q: has no matching L:.\nQ: {pending_q!r}"
        )

    if not questions:
        raise ValueError(
            f"No active questions found in {path}.\n"
            "Un-comment some Q:/L: blocks (remove the leading #)."
        )

    return questions


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def save_json(data: dict, path: Path = config.INTERVIEW_JSON) -> None:
    """Write data to path as indented JSON, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# --init scaffolding
# ---------------------------------------------------------------------------

EXAMPLE_SECRETS = """\
# API keys for the eCardio Vodcast pipeline.
# Copy this file to api_keys.env and fill in your real keys.
# NEVER commit api_keys.env — it is gitignored.
#
# Anthropic (Claude): https://console.anthropic.com/settings/keys
ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE

# OpenAI (ChatGPT): https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-YOUR_KEY_HERE

# Google (Gemini): https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=AIzaYOUR_KEY_HERE
"""

QUESTIONS_CONTENT = """\
# 'The Algorithm Will See You Now': AIs Talk on AI in Cardiovascular Healthcare
# eCardio Vodcast — AI Panel
#
# FORMAT:
#   Q: <open-text question for the model to answer in full>
#   L: <0-10 Likert sub-question (answered 100 times per model for a distribution)>
#   Lines beginning with # are ignored.
#
# BEFORE RUNNING THE PIPELINE:
#   Edit this file down to 4-5 questions.
#   Comment out unwanted questions by adding # before their Q: line
#   (you can leave the L: line — it will be ignored without a preceding active Q:).
#   Then run:  python run_ai_panel.py          (dry-run: prints cost estimate)
#              python run_ai_panel.py --execute  (after reviewing the estimate)

Q: How do you see artificial intelligence changing the accuracy of cardiovascular disease diagnosis over the next decade, and what does this mean for patients?
L: On a scale of 0 (not at all confident) to 10 (fully confident), how confident are you that AI will match or exceed board-certified cardiologist diagnostic accuracy for common cardiovascular diseases by 2035?

Q: AI is increasingly being applied to ECG interpretation. How close are we to AI-read ECGs being used routinely in clinical practice without physician oversight?
L: On a scale of 0 (not at all ready) to 10 (completely ready), how ready is AI-based ECG interpretation for routine unsupervised clinical deployment today?

Q: Traditional risk scores like SCORE2 and Framingham have guided prevention for decades. How fundamentally will AI change individual cardiovascular risk prediction?
L: On a scale of 0 (no improvement) to 10 (transformative improvement), how much will AI-based models improve individual cardiovascular risk prediction compared to current validated scores?

Q: Echocardiography, cardiac MRI, and CT angiography generate enormous amounts of imaging data. How will AI reshape the field of cardiac imaging analysis and reporting?
L: On a scale of 0 (no confidence) to 10 (very high confidence), how confident are you that AI will significantly reduce clinically meaningful inter-observer variability in cardiac imaging within 10 years?

Q: Heart failure is the leading cause of hospitalisation in people over 65. How can AI improve the monitoring and personalisation of heart failure therapy?
L: On a scale of 0 (no benefit) to 10 (major benefit), how beneficial will AI-guided heart failure therapy titration be for patient outcomes in the next decade?

Q: Consumer wearables like smartwatches are already detecting atrial fibrillation. Where is this going, and should clinicians trust these AI-driven detections?
L: On a scale of 0 (no trust) to 10 (full trust), how much do you currently trust consumer-wearable AI tools for generating clinically actionable atrial fibrillation detections?

Q: There is growing concern that AI trained on biased data may amplify existing health inequities. How real is this risk in cardiovascular medicine, and how should we address it?
L: On a scale of 0 (not at all concerned) to 10 (extremely concerned), how concerned are you that AI deployment in cardiovascular care will worsen health inequities if left unaddressed?

Q: AI in cardiology requires access to vast amounts of sensitive patient data. How do we balance innovation with the right to privacy?
L: On a scale of 0 (completely inadequate) to 10 (fully adequate), how adequate are current legal and technical safeguards for cardiovascular patient data used in AI development?

Q: Will practising cardiologists embrace AI decision-support tools, or will there be significant resistance? What will drive adoption or reluctance?
L: On a scale of 0 (extremely slow, >20 years) to 10 (very fast, <5 years), how quickly do you expect widespread adoption of AI decision-support tools among practising cardiologists?

Q: Regulatory agencies like the FDA and EMA are struggling to keep up with the pace of AI development. How should AI in cardiovascular medicine be regulated to protect patients without stifling innovation?
L: On a scale of 0 (far behind) to 10 (fully keeping pace), how well do current regulatory frameworks keep pace with the speed of AI development in cardiovascular medicine?

Q: AI is beginning to accelerate drug discovery. How might this change the pipeline for new cardiovascular therapies?
L: On a scale of 0 (minimal impact) to 10 (transformative impact), how transformative do you expect AI to be for cardiovascular drug discovery over the next 10 years?

Q: Precision or personalised medicine promises to tailor treatment to the individual. What role will AI play in making precision cardiology a clinical reality?
L: On a scale of 0 (marginal role) to 10 (central role), how central will AI be to the realisation of precision cardiology in routine clinical practice?

Q: Billions of people around the world have little or no access to a cardiologist. Can AI close this gap, and if so, how?
L: On a scale of 0 (no improvement) to 10 (dramatic improvement), how much do you believe AI can improve access to cardiovascular care in low-resource and underserved settings?

Q: There is a fear in medicine that AI will replace clinicians. Do you think AI will replace cardiologists, or augment them — and does the distinction matter?
L: On a scale of 0 (AI will largely replace cardiologists) to 10 (AI will purely augment, never replace), where do you see the long-term human-AI relationship in cardiology settling?

Q: Clinicians and patients need to understand why an AI made a particular recommendation. How important is explainability in cardiovascular AI, and how do we achieve it?
L: On a scale of 0 (not important) to 10 (absolutely essential), how important is AI explainability for the safe and ethical adoption of AI in clinical cardiovascular decision-making?

Q: Lifestyle modification is the cornerstone of cardiovascular prevention. How can AI be used to change human behaviour and reduce the burden of preventable heart disease?
L: On a scale of 0 (minimally effective) to 10 (highly effective), how effective do you believe AI-powered personalised interventions will be at driving lasting preventive cardiovascular behaviour change?

Q: In acute cardiovascular emergencies — heart attacks, cardiac arrest, aortic dissection — speed is life. How can AI improve outcomes in these time-critical settings?
L: On a scale of 0 (no meaningful improvement) to 10 (dramatic improvement), how much do you expect AI to improve outcomes in acute cardiovascular emergencies such as STEMI or out-of-hospital cardiac arrest within the next decade?

Q: When an AI system makes a clinical error that harms a patient, who is responsible — the developer, the hospital, the clinician, or the AI itself? How should accountability work?
L: On a scale of 0 (completely unclear) to 10 (very clearly defined), how well-defined do you believe accountability and liability frameworks are for AI-related clinical errors in cardiovascular medicine today?

Q: Healthcare systems worldwide are under financial pressure. Will AI reduce the overall cost of cardiovascular care, or simply shift costs elsewhere?
L: On a scale of 0 (will not reduce costs) to 10 (will significantly reduce costs), how likely do you think it is that widespread AI adoption will lead to a net reduction in overall cardiovascular care costs?

Q: Looking ahead to 2040, what single development in cardiovascular AI excites you most? And overall, are you optimistic or pessimistic that AI will meaningfully improve cardiovascular health outcomes for humanity?
L: On a scale of 0 (very pessimistic) to 10 (very optimistic), how optimistic are you overall that AI will meaningfully improve global cardiovascular health outcomes by 2040?
"""


def scaffold_init() -> None:
    """Write starter files for --init mode. No API calls, no cost."""
    # secrets/
    secrets_dir = config.BASE_DIR / "secrets"
    secrets_dir.mkdir(exist_ok=True)
    example_path = secrets_dir / "api_keys.env.example"
    example_path.write_text(EXAMPLE_SECRETS, encoding="utf-8")
    print(f"  Created : {example_path.relative_to(config.BASE_DIR)}")

    # questions.txt
    q_path = config.QUESTIONS_FILE
    q_path.write_text(QUESTIONS_CONTENT, encoding="utf-8")
    print(f"  Created : {q_path.relative_to(config.BASE_DIR)}")

    # outputs dirs
    config.DISTRIBUTIONS_DIR.mkdir(parents=True, exist_ok=True)
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("Next steps:")
    print("  1. Edit questions.txt — comment out questions you don't want")
    print("     (keep 4-5 to stay within budget).")
    print("  2. Copy secrets/api_keys.env.example → secrets/api_keys.env")
    print("     and fill in your real API keys.")
    print("  3. Run:  python run_ai_panel.py")
    print("     This prints a cost estimate. No tokens spent.")
    print("  4. If the estimate looks good:")
    print("     python run_ai_panel.py --execute")
    print()
