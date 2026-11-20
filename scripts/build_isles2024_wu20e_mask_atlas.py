#!/usr/bin/env python3
"""Build a PRIVATE 15-case anatomical-support review atlas from existing QC PNGs.

Use only within authorized patient-data workspace. Never commit images or PDF
output to a public repository. No automated clinical decisions or approvals.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

PAGE_W, PAGE_H = landscape(A4)
NAVY = colors.HexColor("#183047")
ACCENT = colors.HexColor("#117A88")
GRAY = colors.HexColor("#5A6876")


def coverage_from_log(log: Path) -> dict[str, float]:
    text = log.read_text(encoding="utf-8")
    values = {}
    for case, pct in re.findall(r"\[\d+/149\] (sub_stroke\d{4}) (?:COMPUTED|CACHED) support=([\d.]+)%", text):
        values[case] = float(pct)
    if len(values) != 149:
        raise ValueError(f"Expected 149 patient support QC lines; found {len(values)}")
    return values


def write_atlas(images: Path, log: Path, destination: Path) -> int:
    coverage = coverage_from_log(log)
    source_images = sorted(images.glob("*_support_QC.png"))
    if len(source_images) != 15:
        raise ValueError(f"Expected exactly 15 independent review images, got {len(source_images)}")
    actual = [im.name.removesuffix("_support_QC.png") for im in source_images]
    if len(set(actual)) != 15 or not set(actual).issubset(set(coverage)):
        raise ValueError("Overlay identities conflict with the audit log")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(destination), pagesize=landscape(A4), pageCompression=1)
    pdf.setTitle("ISLES2024 WU20E | PRIVATE anatomical-support reviewer atlas")
    pdf.setAuthor("ISLES2024 project | reviewer package; no approval")
    pdf.setSubject("Technical mask QC, not diagnosis or certified brain extraction")

    # Cover
    pdf.setFillColor(NAVY)
    pdf.rect(0, PAGE_H - 135, PAGE_W, 135, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 26)
    pdf.drawString(45, PAGE_H - 70, "ISLES2024 | Anatomical-support QC")
    pdf.setFont("Helvetica", 15)
    pdf.drawString(45, PAGE_H - 105, "WU-20E independent reviewer atlas")
    pdf.setFillColor(NAVY)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(46, PAGE_H - 188, "PRIVATE PATIENT-IMAGE REVIEW - UNAPPROVED")
    pdf.setFillColor(GRAY)
    details = [
        "15 prespecified priority overlays selected from 149 NCCT patient volumes.",
        "Each figure compares three source CT slices and the provisional support overlay.",
        "The support is a head-tissue proxy: it is NOT a validated clinical brain mask.",
        "Coverage fraction is relative to the acquired image volume; scan FOV can vary.",
        "Reviewer must inspect source images where needed; these three slices are not proof",
        "that all relevant brain tissue is retained across the complete native volume.",
        "No lesion labels, outcome measures or past model scores informed this review.",
    ]
    for i, line in enumerate(details):
        pdf.drawString(46, PAGE_H - 229 - 29 * i, line)
    pdf.setFillColor(colors.HexColor("#B32432"))
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(46, 84, "ALL REVIEW DECISIONS CURRENTLY PENDING. NO TRAINING AUTHORIZATION.")
    pdf.setFont("Helvetica", 9)
    pdf.setFillColor(GRAY)
    pdf.drawString(46, 58, "Do not publish, share openly, or upload to the public GitHub repository.")
    pdf.showPage()

    for idx, path in enumerate(source_images, 1):
        case = path.name.removesuffix("_support_QC.png")
        fraction = coverage[case]
        pdf.setFillColor(NAVY)
        pdf.setFont("Helvetica-Bold", 17)
        pdf.drawString(38, PAGE_H - 35, f"Case {idx:02d}/15 | {case}")
        pdf.setFont("Helvetica", 10)
        pdf.setFillColor(GRAY)
        pdf.drawString(38, PAGE_H - 56,
                       f"Image-derived head-tissue support: {fraction:.2f}% of image volume | unapproved proxy")
        if fraction < 10.0:
            pdf.setFillColor(colors.HexColor("#B32432"))
            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawRightString(PAGE_W - 35, PAGE_H - 35, "PRIORITY: low support coverage")
        im_reader = ImageReader(str(path))
        w, h = im_reader.getSize()
        width = min(756, PAGE_W - 72)
        height = width * h / w
        available_height = PAGE_H - 130
        if height > available_height:
            height = available_height
            width = height * w / h
        left = (PAGE_W - width) / 2
        bottom = 68
        pdf.drawImage(im_reader, left, bottom, width=width, height=height,
                      preserveAspectRatio=True, mask='auto')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        pdf.setFillColor(GRAY)
        pdf.setFont("Helvetica", 8)
        pdf.drawString(38, 39, f"Image SHA256 {digest}")
        pdf.drawString(38, 24, "Reviewer: record APPROVED / REJECTED / PENDING and reasons in the PRIVATE CSV; no auto-approval")
        pdf.showPage()

    pdf.setFillColor(NAVY)
    pdf.setFont("Helvetica-Bold", 21)
    pdf.drawString(42, PAGE_H - 58, "Independent review and decision record")
    pdf.setFont("Helvetica", 12)
    pdf.setFillColor(GRAY)
    instructions = [
        "1. Review all 15 priority overlays and inspect original native volumes where necessary.",
        "2. Record per-case tissue preservation, extracranial inclusion and corrections in the existing CSV.",
        "3. Document reviewer name, clinical/technical role, organization and ISO date externally.",
        "4. Review the clipping/replacement/normalization policy independently of prior model outcomes.",
        "5. Require a separate, verifiable signed policy record before any staging authorization.",
        "6. Review only the intended acute imaging inputs, never lesion reference maps or OOF scores.",
        "7. These 15 slices alone cannot establish anatomical validity across all 149 patients.",
    ]
    for idx, line in enumerate(instructions):
        pdf.drawString(44, PAGE_H - 104 - 38 * idx, line)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(colors.HexColor("#B32432"))
    pdf.drawString(44, PAGE_H - 423, "This PDF is a review aid. It is not an approval form or clinical attestation.")
    pdf.setFillColor(GRAY)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(44, PAGE_H - 460, "No training may start while decisions remain pending or until full release gates are met.")
    pdf.showPage()
    pdf.save()
    return len(source_images)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    count = write_atlas(a.images, a.log, a.output)
    print(f"PRIVATE atlas complete: {count} cases -> {a.output}")


if __name__ == "__main__":
    main()
