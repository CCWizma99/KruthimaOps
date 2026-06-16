"""
FloodGuard SL - PDF Report Generator
Generates a professional flood-risk report from a stored prediction.

This module intentionally imports ReportLab inside the function so the API can
still start with a clear error message if the dependency is missing.
"""
from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Optional
from xml.sax.saxutils import escape


def _safe(value: Any, fallback: str = "-") -> str:
    if value is None:
        return fallback
    text = str(value)
    return text if text.strip() else fallback


def _risk_description(level: str) -> str:
    level = (level or "").upper()
    return {
        "LOW": "Low flood risk. Continue normal monitoring and keep basic awareness.",
        "MEDIUM": "Moderate flood risk. Monitor rainfall changes and avoid unnecessary travel near low-lying water paths.",
        "HIGH": "High flood risk. Prepare emergency contacts, protect valuables, and review safe evacuation options.",
        "EXTREME": "Extreme flood risk. Immediate caution is required. Prioritise safety, avoid flood-prone routes, and follow official warnings.",
    }.get(level, "Risk level unavailable. Treat this prediction as decision-support information only.")


def _priority_label(level: str) -> str:
    level = (level or "").upper()
    return {
        "LOW": "Green / Routine Monitoring",
        "MEDIUM": "Yellow / Watch Zone",
        "HIGH": "Orange / Preparedness Zone",
        "EXTREME": "Red / Critical Attention",
    }.get(level, "Unclassified")


def _recommendations(level: str) -> list[str]:
    level = (level or "").upper()
    base = [
        "Use this report as decision-support, not as an official disaster warning.",
        "Check official weather and Disaster Management Centre updates before taking action.",
    ]
    if level == "LOW":
        return [
            "Continue routine monitoring of rainfall and water levels.",
            "Keep community communication channels active.",
            *base,
        ]
    if level == "MEDIUM":
        return [
            "Monitor rainfall updates more frequently over the next 24-72 hours.",
            "Avoid unnecessary travel through low-lying roads and river-bank areas.",
            "Prepare emergency contacts and important documents.",
            *base,
        ]
    if level == "HIGH":
        return [
            "Prepare a household or community evacuation plan.",
            "Move valuable items, electronics, and documents to higher ground.",
            "Identify nearest evacuation support and hospital access routes.",
            "Inform vulnerable groups such as elderly people, children, and patients.",
            *base,
        ]
    if level == "EXTREME":
        return [
            "Avoid entering flooded roads, bridges, and waterlogged zones.",
            "Move to a safer location if official instructions or local conditions require it.",
            "Keep phones charged and maintain contact with family/community leaders.",
            "Prioritise life safety over property protection.",
            *base,
        ]
    return base


def _p(text: Any, style: Any) -> Any:
    """ReportLab Paragraph with escaping and predictable wrapping."""
    from reportlab.platypus import Paragraph

    return Paragraph(escape(_safe(text)), style)


def build_prediction_report_pdf(
    prediction: Dict[str, Any],
    model_metadata: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Return a PDF report as bytes for a stored prediction row."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            HRFlowable,
        )
    except ImportError as exc:  # pragma: no cover - depends on deployment environment
        raise RuntimeError(
            "ReportLab is not installed. Run: pip install reportlab "
            "or add 'reportlab==4.2.5' to production/requirements.txt."
        ) from exc

    model_metadata = model_metadata or {}
    metrics = metrics or {}

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title="FloodGuard SL Prediction Report",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#075985"),
        spaceBefore=10,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="SmallMuted",
        parent=styles["BodyText"],
        fontSize=8.2,
        leading=10.5,
        textColor=colors.HexColor("#64748b"),
        wordWrap="CJK",
    ))
    styles.add(ParagraphStyle(
        name="BodyClean",
        parent=styles["BodyText"],
        fontSize=9.3,
        leading=12.5,
        textColor=colors.HexColor("#1e293b"),
        wordWrap="CJK",
    ))
    styles.add(ParagraphStyle(
        name="CellLabel",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=8.6,
        leading=11.2,
        textColor=colors.HexColor("#0f172a"),
        wordWrap="CJK",
    ))
    styles.add(ParagraphStyle(
        name="CellValue",
        parent=styles["BodyText"],
        fontSize=8.6,
        leading=11.2,
        textColor=colors.HexColor("#1e293b"),
        wordWrap="CJK",
    ))
    styles.add(ParagraphStyle(
        name="CellValueWhite",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=8.8,
        leading=11.5,
        textColor=colors.white,
        wordWrap="CJK",
    ))
    styles.add(ParagraphStyle(
        name="ActionNumber",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        alignment=1,
        textColor=colors.white,
    ))

    pred_id = _safe(prediction.get("prediction_id"))
    district = _safe(prediction.get("district"))
    level = _safe(prediction.get("risk_level"), "UNKNOWN").upper()
    score = prediction.get("risk_score")
    score_text = f"{float(score):.4f}" if score is not None else "-"
    ts_raw = prediction.get("timestamp")
    try:
        ts_text = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        ts_text = _safe(ts_raw)

    risk_colors = {
        "LOW": colors.HexColor("#16a34a"),
        "MEDIUM": colors.HexColor("#ca8a04"),
        "HIGH": colors.HexColor("#ea580c"),
        "EXTREME": colors.HexColor("#dc2626"),
    }
    risk_color = risk_colors.get(level, colors.HexColor("#475569"))

    story = []
    story.append(Paragraph("FloodGuard SL - Prediction Report", styles["ReportTitle"]))
    story.append(Paragraph(
        "AI-powered flood risk decision-support report generated from the production monitoring log.",
        styles["SmallMuted"],
    ))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1")))
    story.append(Spacer(1, 8))

    # Compact four-column summary. Values are Paragraphs, so long UUIDs wrap instead of overlapping.
    generated_text = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    summary_data = [
        [
            _p("Prediction ID", styles["CellLabel"]),
            _p(pred_id, styles["CellValue"]),
            _p("Generated", styles["CellLabel"]),
            _p(generated_text, styles["CellValue"]),
        ],
        [
            _p("District", styles["CellLabel"]),
            _p(district, styles["CellValue"]),
            _p("Prediction Time", styles["CellLabel"]),
            _p(ts_text, styles["CellValue"]),
        ],
        [
            _p("Risk Score", styles["CellLabel"]),
            _p(score_text, styles["CellValue"]),
            _p("Risk Level", styles["CellValueWhite"]),
            _p(level, styles["CellValueWhite"]),
        ],
        [
            _p("Priority", styles["CellLabel"]),
            _p(_priority_label(level), styles["CellValue"]),
            _p("Latency", styles["CellLabel"]),
            _p(f"{_safe(prediction.get('latency_ms'))} ms", styles["CellValue"]),
        ],
    ]
    summary_table = Table(summary_data, colWidths=[2.8 * cm, 5.4 * cm, 3.1 * cm, 4.8 * cm], repeatRows=0)
    summary_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f1f5f9")),
        ("BACKGROUND", (2, 2), (3, 2), risk_color),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(summary_table)

    story.append(Paragraph("Risk Interpretation", styles["SectionTitle"]))
    story.append(Paragraph(escape(_risk_description(level)), styles["BodyClean"]))

    story.append(Paragraph("Input Evidence Snapshot", styles["SectionTitle"]))
    inputs = [
        ["Rainfall 7 days", f"{_safe(prediction.get('rainfall_7d'))} mm"],
        ["Flood occurrence", _safe(prediction.get("flood_occurrence"))],
        ["Inundation area", f"{_safe(prediction.get('inundation_area'))} sqm"],
        ["Safe to live response", _safe(prediction.get("is_good_to_live"))],
        ["Latitude / Longitude", f"{_safe(prediction.get('latitude'))}, {_safe(prediction.get('longitude'))}"],
        ["Validation warnings", _safe(prediction.get("warning_text"), "None")],
    ]
    input_data = [[_p(label, styles["CellLabel"]), _p(value, styles["CellValue"])] for label, value in inputs]
    input_table = Table(input_data, colWidths=[4.8 * cm, 11.3 * cm])
    input_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e0f2fe")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(input_table)

    story.append(Paragraph("Recommended Actions", styles["SectionTitle"]))
    rec_rows = [[_p(str(i), styles["ActionNumber"]), _p(rec, styles["CellValue"])] for i, rec in enumerate(_recommendations(level), start=1)]
    rec_table = Table(rec_rows, colWidths=[0.9 * cm, 15.2 * cm])
    rec_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (0, -1), risk_color),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(rec_table)

    story.append(Paragraph("Model & MLOps Trace", styles["SectionTitle"]))
    model_names = model_metadata.get("model_names", [])
    if isinstance(model_names, (list, tuple)):
        model_names_text = ", ".join(str(name) for name in model_names) or "-"
    else:
        model_names_text = _safe(model_names)
    mlops_rows = [
        ["Model version", _safe(model_metadata.get("version"))],
        ["Base pipeline", _safe(model_metadata.get("base_pipeline"))],
        ["Base models", model_names_text],
        ["OOF MAE", _safe(model_metadata.get("oof_mae"))],
        ["OOF RMSE", _safe(model_metadata.get("oof_rmse"))],
        ["OOF EV", _safe(model_metadata.get("oof_ev"))],
        ["Total API predictions", _safe(metrics.get("total_predictions"))],
        ["Average latency", f"{_safe(metrics.get('avg_latency_ms'))} ms"],
    ]
    mlops_data = [[_p(label, styles["CellLabel"]), _p(value, styles["CellValue"])] for label, value in mlops_rows]
    mlops_table = Table(mlops_data, colWidths=[4.8 * cm, 11.3 * cm])
    mlops_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(mlops_table)

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Disclaimer: FloodGuard SL is a competition prototype and decision-support tool. "
        "It must not replace official disaster warnings, field observations, or government instructions.",
        styles["SmallMuted"],
    ))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
