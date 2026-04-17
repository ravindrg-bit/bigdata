from __future__ import annotations

import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DATA = ROOT / "data"
DEMO = ROOT / "demo"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def draw_wrapped_lines(c: canvas.Canvas, text: str, x: int, y: int, max_chars: int = 95, leading: int = 16) -> int:
    words = text.split()
    lines = []
    cur = []

    for word in words:
        trial = " ".join(cur + [word])
        if len(trial) > max_chars and cur:
            lines.append(" ".join(cur))
            cur = [word]
        else:
            cur.append(word)
    if cur:
        lines.append(" ".join(cur))

    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def create_executive_report(fairness: dict, federated: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    report_path = DOCS / "executive_report.pdf"

    c = canvas.Canvas(str(report_path), pagesize=A4)
    width, height = A4

    c.setTitle("From Grameen to GPUs: A Hybrid Risk Engine for Banco Falabella's Mexico Thin-File Market")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(
        50,
        height - 60,
        "From Grameen to GPUs: A Hybrid Risk Engine for Banco Falabella's Mexico Thin-File Market",
    )

    y = height - 95
    c.setFont("Helvetica", 11)

    sections = [
        (
            "Problem",
            "Mexico's thin-file lending environment challenges traditional bureau-heavy underwriting. "
            "Banco Falabella can bridge this gap with store-originated acquisition and graph-aware AI risk scoring.",
        ),
        (
            "Approach",
            "The solution combines Grameen-style group dynamics, Tala-style behavioral features, "
            "and GraphSAGE embeddings in a hybrid XGBoost ensemble.",
        ),
        (
            "Results",
            f"Hybrid hold-out AUC={fairness.get('baseline', {}).get('classification_metrics', {}).get('auc', 'n/a')}, "
            f"PR-AUC={fairness.get('baseline', {}).get('classification_metrics', {}).get('pr_auc', 'n/a')}, "
            f"F1={fairness.get('baseline', {}).get('classification_metrics', {}).get('f1', 'n/a')}. "
            f"Federated AUC={federated.get('federated_metrics', {}).get('auc', 'n/a')} "
            f"vs centralized AUC={federated.get('centralized_metrics', {}).get('auc', 'n/a')}.",
        ),
        (
            "Three-Phase Rollout",
            "Month 1 uses rule-based scoring over validated store signals. Months 2-3 use hybrid-lite "
            "tabular models. Month 3+ activates full peer-graph hybrid scoring.",
        ),
        (
            "AI Scaling Roadmap",
            "Next upgrades prioritize federated retraining, near-real-time event scoring, and "
            "embedding integration into CMR mobile journeys.",
        ),
        (
            "Risk Register",
            "Primary risks include data drift, subgroup fairness regression, and operational override misuse. "
            "Mitigations include periodic fairness audits, calibration monitoring, and human-in-the-loop controls.",
        ),
        (
            "Macro Anchor",
            "Fitch's BBB- stable perspective for Falabella (Oct 2025) supports the investment thesis for "
            "disciplined AI-assisted expansion in Mexico's underbanked segments.",
        ),
    ]

    for title, body in sections:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, title)
        y -= 16
        c.setFont("Helvetica", 11)
        y = draw_wrapped_lines(c, body, 50, y, max_chars=96, leading=15)
        y -= 8
        if y < 90:
            c.showPage()
            y = height - 60

    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 45, "Generated from repository artifacts on 2026-04-13.")
    c.save()



def create_market_entry_playbook() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    playbook_path = DOCS / "market_entry_playbook.pdf"

    c = canvas.Canvas(str(playbook_path), pagesize=A4)
    width, height = A4
    c.setTitle("Banco Falabella Mexico Market-Entry Playbook")

    c.setFont("Helvetica-Bold", 15)
    c.drawString(50, height - 60, "Banco Falabella Mexico Market-Entry Playbook")

    y = height - 90
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "1. Zero-CAC Acquisition Funnel")
    y -= 18
    c.setFont("Helvetica", 11)
    y = draw_wrapped_lines(
        c,
        "Leverage existing Falabella/Sodimac traffic and verified purchase behavior to identify low-risk "
        "new-to-credit applicants before broad digital expansion.",
        50,
        y,
        max_chars=95,
    )

    y -= 14
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "2. Three-Phase Credit Expansion")
    y -= 18
    c.setFont("Helvetica", 11)
    bullets = [
        "Month 1: Rule-based risk gating (MXN 500-2,000).",
        "Months 2-3: Hybrid-lite tabular model (MXN 2,000-8,000).",
        "Month 3+: Full graph hybrid scoring (MXN 8,000-25,000).",
    ]
    for bullet in bullets:
        y = draw_wrapped_lines(c, f"- {bullet}", 60, y, max_chars=92)

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "3. Governance and Control")
    y -= 18
    c.setFont("Helvetica", 11)
    y = draw_wrapped_lines(
        c,
        "Use explainability payloads, override logs, subgroup fairness checks, and federated retraining "
        "to maintain compliance-ready model operations during scaling.",
        50,
        y,
        max_chars=95,
    )

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "4. KPI Targets")
    y -= 18
    c.setFont("Helvetica", 11)
    for line in [
        "- Portfolio AUC > 0.80",
        "- Subgroup parity gaps < 5%",
        "- End-to-end scoring latency <= 200 ms",
        "- Stable repayment behavior through phased credit-line growth",
    ]:
        y = draw_wrapped_lines(c, line, 60, y, max_chars=92)

    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 45, "One-page market-entry playbook generated on 2026-04-13.")
    c.save()



def create_demo_video() -> None:
    DEMO.mkdir(parents=True, exist_ok=True)
    video_path = DEMO / "demo_video.mp4"
    silent_video_path = DEMO / "demo_video_silent.mp4"
    script_path = DEMO / "narration_script.txt"
    narration_audio_path = DEMO / "narration.aiff"

    scenes = [
        {
            "title": "Falabella Hybrid Risk Engine Demo",
            "lines": [
                "1) Applicant intake from store + mobile signals",
                "2) Phase-based scorer selects rule / hybrid-lite / full hybrid",
                "3) Risk score, credit line, and explainability displayed",
            ],
            "duration_s": 20,
        },
        {
            "title": "Portfolio Risk Tab",
            "lines": [
                "Network view highlights borrower communities",
                "Color encodes neighborhood risk pressure",
                "Scatter checks default vs repayment latency signal",
            ],
            "duration_s": 20,
        },
        {
            "title": "Borrower Lookup Tab",
            "lines": [
                "Manual form accepts CURP hash, INE flag, store visits, CoDi behavior",
                "Model returns risk score, decision, and credit line range",
                "SHAP waterfall explains top risk drivers",
            ],
            "duration_s": 20,
        },
        {
            "title": "Ethics Audit Tab",
            "lines": [
                "Displays subgroup parity and equal opportunity gaps",
                "Shows post-mitigation fairness status",
                "Supports model card download for governance",
            ],
            "duration_s": 20,
        },
        {
            "title": "Federated Mode",
            "lines": [
                "FedAvg simulation across four regional shards",
                "Dashboard can switch to federated model scoring",
                "Benchmark compares federated vs centralized AUC",
            ],
            "duration_s": 20,
        },
        {
            "title": "Decision Operations",
            "lines": [
                "Loan officer override is logged with timestamp and reason",
                "Audit artifacts stored for compliance reviews",
                "End-to-end architecture supports phased market entry",
            ],
            "duration_s": 20,
        },
    ]

    script_lines = []
    fps = 1
    width, height = 1280, 720

    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 54)
        body_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 34)
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    writer = imageio.get_writer(str(silent_video_path), fps=fps)
    for scene in scenes:
        script_lines.append(scene["title"])
        script_lines.extend([f"- {line}" for line in scene["lines"]])
        script_lines.append("")

        frame = Image.new("RGB", (width, height), color=(18, 35, 58))
        draw = ImageDraw.Draw(frame)

        draw.text((60, 60), scene["title"], fill=(245, 245, 245), font=title_font)
        y = 180
        for line in scene["lines"]:
            draw.text((90, y), f"- {line}", fill=(220, 230, 240), font=body_font)
            y += 80

        arr = np.array(frame)
        repeats = int(scene["duration_s"] * fps)
        for _ in range(repeats):
            writer.append_data(arr)

    writer.close()
    script_path.write_text("\n".join(script_lines), encoding="utf-8")

    try:
        subprocess.run(
            ["say", "-v", "Samantha", "-r", "165", "-f", str(script_path), "-o", str(narration_audio_path)],
            check=True,
            capture_output=True,
            text=True,
        )

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-i",
                str(silent_video_path),
                "-i",
                str(narration_audio_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if silent_video_path.exists():
            silent_video_path.unlink()
    except Exception:
        if silent_video_path.exists():
            silent_video_path.replace(video_path)



def main() -> None:
    fairness = load_json(DATA / "fairness_report.json")
    federated = load_json(DATA / "federated_report.json")

    create_executive_report(fairness=fairness, federated=federated)
    create_market_entry_playbook()
    create_demo_video()

    print("Generated:")
    print("- docs/executive_report.pdf")
    print("- docs/market_entry_playbook.pdf")
    print("- demo/demo_video.mp4")
    print("- demo/narration_script.txt")


if __name__ == "__main__":
    main()
