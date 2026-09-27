"""
faceeval.reporting.generator
==============================
Assembles ``ExperimentReport`` objects and exports them to CSV, JSON,
and PDF formats.

Report structure
----------------
An ``ExperimentReport`` packages:
  - All ``EvaluationResult`` objects
  - All ``BehavioralFingerprint`` objects
  - All ``FailureCluster`` objects
  - All ``DeploymentRanking`` objects
  - All ``FairnessReport`` objects
  - Figure specs (paths to exported figures)
  - A text summary suitable for the thesis abstract

CSV exports
-----------
  evaluation_results.csv   — one row per (model × perturbation × severity)
  adc_summary.csv          — one row per (model × perturbation type)
  deployment_ranking.csv   — one row per (model × scenario)
  failure_clusters.csv     — one row per cluster
  fingerprint_scores.csv   — one row per (model × dimension)

JSON exports
------------
  experiment_report.json   — full report as nested JSON
  fingerprints.json        — fingerprint vectors and scores
  deployment_rankings.json — ranked scores per scenario

PDF report
----------
A structured PDF with all figures, tables, and text.
Uses ReportLab when available, falls back to an HTML export otherwise.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from faceeval.core.types import (
    BehavioralFingerprint,
    DeploymentRanking,
    EvaluationResult,
    ExperimentConfig,
    ExperimentReport,
    ExperimentRun,
    FailureCluster,
    FairnessReport,
    FigureSpec,
    RunID,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report assembler
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    Assembles and exports experiment reports.

    Parameters
    ----------
    output_dir:
        Root directory for all report files.
    run_id:
        The experiment run ID (used for filenames).
    """

    def __init__(
        self,
        output_dir: str = "reports/",
        run_id: RunID = "unknown",
    ) -> None:
        self._root = Path(output_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def build_report(
        self,
        run: ExperimentRun,
        figure_specs: list[FigureSpec] | None = None,
        fairness_reports: list[FairnessReport] | None = None,
    ) -> ExperimentReport:
        """
        Assemble an ``ExperimentReport`` from a completed ``ExperimentRun``.

        Parameters
        ----------
        run:
            Completed experiment run (status should be COMPLETED).
        figure_specs:
            Specs for all generated figures.
        fairness_reports:
            Per-model fairness reports.

        Returns
        -------
        ExperimentReport
        """
        summary = self._build_summary(run)

        report = ExperimentReport(
            run_id=run.run_id,
            experiment_name=run.config.experiment_name,
            config=run.config,
            evaluation_results=run.evaluation_results,
            fingerprints=run.fingerprints,
            failure_clusters=run.failure_clusters,
            deployment_rankings=run.deployment_rankings,
            fairness_reports=fairness_reports or [],
            figures=figure_specs or [],
            generated_at=datetime.utcnow(),
            summary_text=summary,
        )

        logger.info(
            "Report assembled: %d results, %d fingerprints, %d rankings, %d figures.",
            len(report.evaluation_results),
            len(report.fingerprints),
            len(report.deployment_rankings),
            len(report.figures),
        )
        return report

    # ------------------------------------------------------------------
    # CSV exports
    # ------------------------------------------------------------------

    def export_evaluation_results_csv(
        self, results: list[EvaluationResult]
    ) -> str:
        """Export evaluation results as CSV. Returns file path."""
        path = self._root / f"evaluation_results_{self._run_id}.csv"
        if not results:
            return str(path)

        fieldnames = [
            "run_id", "model_name", "model_paradigm", "dataset_name",
            "perturbation_type", "perturbation_severity",
            "accuracy", "auc", "eer", "f1_score", "precision", "recall",
            "num_samples",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                row = {
                    "run_id": r.run_id,
                    "model_name": r.model_name,
                    "model_paradigm": r.model_paradigm.value,
                    "dataset_name": r.dataset_name,
                    "perturbation_type": (
                        r.perturbation_spec.perturbation_type.value
                        if r.perturbation_spec else "clean"
                    ),
                    "perturbation_severity": (
                        r.perturbation_spec.severity
                        if r.perturbation_spec else 0.0
                    ),
                    "accuracy": round(r.accuracy, 6),
                    "auc": round(r.auc, 6),
                    "eer": round(r.eer, 6),
                    "f1_score": round(r.f1_score, 6),
                    "precision": round(r.precision, 6),
                    "recall": round(r.recall, 6),
                    "num_samples": r.num_samples,
                }
                writer.writerow(row)

        logger.info("Evaluation CSV exported: %s", path)
        return str(path)

    def export_fingerprint_csv(
        self, fingerprints: list[BehavioralFingerprint]
    ) -> str:
        """Export fingerprint dimension scores as CSV."""
        path = self._root / f"fingerprints_{self._run_id}.csv"
        if not fingerprints:
            return str(path)

        with path.open("w", newline="", encoding="utf-8") as f:
            dim_names = fingerprints[0].dimension_names
            fieldnames = ["model_name", "paradigm"] + dim_names
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for fp in fingerprints:
                row: dict[str, Any] = {
                    "model_name": fp.model_name,
                    "paradigm": fp.paradigm.value,
                }
                dim_idx = {d: i for i, d in enumerate(fp.dimension_names)}
                for dim in dim_names:
                    idx = dim_idx.get(dim)
                    row[dim] = round(fp.vector[idx], 6) if idx is not None else ""
                writer.writerow(row)

        logger.info("Fingerprint CSV exported: %s", path)
        return str(path)

    def export_deployment_csv(
        self, rankings: list[DeploymentRanking]
    ) -> str:
        """Export deployment rankings as CSV."""
        path = self._root / f"deployment_rankings_{self._run_id}.csv"
        if not rankings:
            return str(path)

        fieldnames = [
            "scenario", "rank", "model_name", "paradigm", "total_score",
            "score_accuracy", "score_robustness", "score_calibration",
            "score_fairness", "score_latency", "score_memory", "score_computational_cost",
            "recommendation",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for ranking in rankings:
                for score in ranking.ranked_scores:
                    row = {
                        "scenario": score.scenario.value,
                        "rank": score.rank,
                        "model_name": score.model_name,
                        "paradigm": score.paradigm.value,
                        "total_score": round(score.total_score, 6),
                        "recommendation": score.recommendation,
                    }
                    for dim in ["accuracy","robustness","calibration",
                                "fairness","latency","memory","computational_cost"]:
                        row[f"score_{dim}"] = round(score.component_scores.get(dim, 0.0), 6)
                    writer.writerow(row)

        logger.info("Deployment CSV exported: %s", path)
        return str(path)

    def export_failure_cluster_csv(
        self, clusters: list[FailureCluster]
    ) -> str:
        """Export failure cluster summaries as CSV."""
        path = self._root / f"failure_clusters_{self._run_id}.csv"
        if not clusters:
            path.touch()   # create empty file so callers can verify export
            return str(path)

        fieldnames = [
            "cluster_id", "model_name", "size",
            "dominant_failure_mode", "dominant_perturbation_type",
            "mean_confidence", "mean_severity", "description",
        ]
        import numpy as np
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for cl in clusters:
                confs = [c.confidence for c in cl.cases if c.confidence]
                sevs  = [c.perturbation_spec.severity for c in cl.cases if c.perturbation_spec]
                row = {
                    "cluster_id": cl.cluster_id,
                    "model_name": cl.model_name,
                    "size": cl.size,
                    "dominant_failure_mode": cl.dominant_failure_mode.value,
                    "dominant_perturbation_type": (
                        cl.dominant_perturbation_type.value
                        if cl.dominant_perturbation_type else ""
                    ),
                    "mean_confidence": round(float(np.mean(confs)), 4) if confs else "",
                    "mean_severity": round(float(np.mean(sevs)), 4) if sevs else "",
                    "description": cl.description,
                }
                writer.writerow(row)

        logger.info("Failure cluster CSV exported: %s", path)
        return str(path)

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def export_json(self, report: ExperimentReport) -> str:
        """Export full report as JSON. Returns file path."""
        path = self._root / f"experiment_report_{self._run_id}.json"
        data = {
            "run_id": report.run_id,
            "experiment_name": report.experiment_name,
            "generated_at": report.generated_at.isoformat(),
            "summary": report.summary_text,
            "config": report.config.to_dict(),
            "num_evaluation_results": len(report.evaluation_results),
            "num_fingerprints": len(report.fingerprints),
            "num_deployment_rankings": len(report.deployment_rankings),
            "num_figures": len(report.figures),
            "deployment_rankings": [r.to_dict() for r in report.deployment_rankings],
            "fingerprints": [fp.to_dict() for fp in report.fingerprints],
            "figures": [f.to_dict() for f in report.figures],
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("JSON report exported: %s", path)
        return str(path)

    # ------------------------------------------------------------------
    # PDF report
    # ------------------------------------------------------------------

    def export_pdf(
        self,
        report: ExperimentReport,
        figure_dir: str = "",
    ) -> str:
        """
        Export a structured PDF report.

        Falls back to HTML if reportlab is not installed.
        """
        try:
            return self._export_pdf_reportlab(report, figure_dir)
        except ImportError:
            logger.info("reportlab not installed — exporting HTML report instead.")
            return self._export_html(report, figure_dir)

    def _export_html(self, report: ExperimentReport, figure_dir: str) -> str:
        """HTML fallback for PDF export."""
        path = self._root / f"report_{self._run_id}.html"
        lines = [
            "<!DOCTYPE html><html><head>",
            "<meta charset='utf-8'>",
            f"<title>FaceEval-X Report — {report.experiment_name}</title>",
            "<style>body{font-family:sans-serif;max-width:900px;margin:auto;padding:2em}",
            "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:6px}",
            "th{background:#f0f0f0}img{max-width:100%;margin:1em 0}</style></head><body>",
            f"<h1>FaceEval-X Experiment Report</h1>",
            f"<h2>{report.experiment_name}</h2>",
            f"<p><b>Run ID:</b> {report.run_id}</p>",
            f"<p><b>Generated:</b> {report.generated_at.isoformat()}</p>",
            f"<h3>Summary</h3><p>{report.summary_text}</p>",
        ]

        # Deployment rankings table
        if report.deployment_rankings:
            lines.append("<h3>Deployment Rankings</h3>")
            for ranking in report.deployment_rankings:
                lines.append(f"<h4>Scenario: {ranking.scenario.value}</h4>")
                lines.append("<table><tr><th>Rank</th><th>Model</th><th>Paradigm</th><th>Score</th><th>Recommendation</th></tr>")
                for score in ranking.ranked_scores:
                    lines.append(
                        f"<tr><td>{score.rank}</td><td>{score.model_name}</td>"
                        f"<td>{score.paradigm.value}</td>"
                        f"<td>{score.total_score:.4f}</td>"
                        f"<td>{score.recommendation[:80]}</td></tr>"
                    )
                lines.append("</table>")

        # Fingerprint table
        if report.fingerprints:
            lines.append("<h3>Behavioral Fingerprint Scores</h3>")
            dims = report.fingerprints[0].dimension_names[:8]
            lines.append(
                "<table><tr><th>Model</th>" +
                "".join(f"<th>{d.replace('robustness_','').replace('_',' ')[:10]}</th>" for d in dims) +
                "</tr>"
            )
            for fp in report.fingerprints:
                dim_idx = {d: i for i, d in enumerate(fp.dimension_names)}
                cells = "".join(
                    f"<td>{fp.vector[dim_idx[d]]:.3f}</td>" if d in dim_idx else "<td>-</td>"
                    for d in dims
                )
                lines.append(f"<tr><td>{fp.model_name}</td>{cells}</tr>")
            lines.append("</table>")

        # Figures
        if report.figures and figure_dir:
            lines.append("<h3>Figures</h3>")
            for fig_spec in report.figures[:12]:  # limit for brevity
                png = fig_spec.output_path_png
                if png and Path(png).exists():
                    lines.append(f"<figure><img src='{png}'><figcaption>{fig_spec.caption}</figcaption></figure>")

        lines.append("</body></html>")
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("HTML report exported: %s", path)
        return str(path)

    def _export_pdf_reportlab(self, report: ExperimentReport, figure_dir: str) -> str:
        """Full PDF export using ReportLab."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
        )
        from reportlab.lib import colors
        from reportlab.lib.units import inch

        path = self._root / f"report_{self._run_id}.pdf"
        doc = SimpleDocTemplate(str(path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph(f"FaceEval-X Experiment Report", styles["Title"]))
        story.append(Paragraph(f"{report.experiment_name}", styles["Heading1"]))
        story.append(Paragraph(f"Run ID: {report.run_id}", styles["Normal"]))
        story.append(Paragraph(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Summary", styles["Heading2"]))
        story.append(Paragraph(report.summary_text, styles["Normal"]))
        story.append(Spacer(1, 0.2*inch))

        # Deployment table
        if report.deployment_rankings:
            story.append(Paragraph("Deployment Rankings", styles["Heading2"]))
            for ranking in report.deployment_rankings[:2]:
                story.append(Paragraph(f"Scenario: {ranking.scenario.value}", styles["Heading3"]))
                data = [["Rank", "Model", "Paradigm", "Score"]]
                for s in ranking.ranked_scores:
                    data.append([str(s.rank), s.model_name, s.paradigm.value, f"{s.total_score:.4f}"])
                t = Table(data, colWidths=[0.6*inch, 1.5*inch, 1.3*inch, 1.0*inch])
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0,0), (-1,0), colors.grey),
                    ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
                    ("GRID", (0,0), (-1,-1), 0.5, colors.black),
                    ("FONTSIZE", (0,0), (-1,-1), 8),
                ]))
                story.append(t)
                story.append(Spacer(1, 0.15*inch))

        doc.build(story)
        logger.info("PDF report exported: %s", path)
        return str(path)

    # ------------------------------------------------------------------
    # Export all at once
    # ------------------------------------------------------------------

    def export_all(
        self,
        report: ExperimentReport,
        config: ExperimentConfig,
        figure_dir: str = "",
    ) -> dict[str, str]:
        """
        Run all exports based on ``config`` flags.

        Returns a dict mapping export type → file path.
        """
        paths: dict[str, str] = {}

        if config.export_csv:
            paths["evaluation_csv"] = self.export_evaluation_results_csv(report.evaluation_results)
            paths["fingerprint_csv"] = self.export_fingerprint_csv(report.fingerprints)
            paths["deployment_csv"] = self.export_deployment_csv(report.deployment_rankings)
            paths["failure_cluster_csv"] = self.export_failure_cluster_csv(report.failure_clusters)

        if config.export_json:
            paths["json"] = self.export_json(report)

        if config.generate_pdf_report:
            paths["pdf_or_html"] = self.export_pdf(report, figure_dir)

        logger.info("All exports complete: %s", list(paths.keys()))
        return paths

    # ------------------------------------------------------------------
    # Summary text
    # ------------------------------------------------------------------

    def _build_summary(self, run: ExperimentRun) -> str:
        config = run.config
        model_names = config.all_model_names
        n_conditions = len(run.evaluation_results)
        n_models = len(model_names)
        best_model = ""
        if run.deployment_rankings:
            best_model = run.deployment_rankings[0].best_model

        duration = f"{run.duration_seconds:.1f}s" if run.duration_seconds else "N/A"

        return (
            f"FaceEval-X experiment '{config.experiment_name}' "
            f"evaluated {n_models} recognition models "
            f"({', '.join(model_names[:4])}{'...' if n_models > 4 else ''}) "
            f"across {n_conditions} (model × perturbation × severity) conditions. "
            f"Dataset(s): {', '.join(config.dataset_names) or 'N/A'}. "
            f"Run completed in {duration}. "
            + (f"Top-ranked model: {best_model}." if best_model else "")
        )
