"""
MarineMine Pro — Report Generation Routes
"""

from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from sqlalchemy.orm import Session
from app.config import REPORTS_DIR
from app.database import get_db
from app.models.db_models import GeneratedReportModel
from app.schemas.api_schemas import ReportGenerateRequest
from app.services.report_service import ReportService
from app.services.state_manager import StateManager

router = APIRouter(prefix="/api/reports", tags=["Reports"])
state = StateManager.get_instance()


@router.post("/generate")
def generate_analytical_report(req: ReportGenerateRequest, db: Session = Depends(get_db)):
    """Generate comprehensive report in Markdown, CSV, or Excel."""
    df = state.get_active_df()
    fmt = req.format.lower()

    if fmt == "markdown":
        content = ReportService.generate_markdown(
            df=df,
            dataset_name=state.dataset_name,
            preprocessing_report=state.preprocessing_report,
            clustering_results=state.clustering_results,
            classification_results=state.classification_results,
            prediction_results=state.prediction_results,
            association_rules_summary=state.association_rules_summary,
        )
        file_name = f"MarineMine_Report_{state.dataset_name}.md"
        target_path = REPORTS_DIR / file_name
        target_path.write_text(content, encoding="utf-8")

        if state.dataset_id:
            try:
                rec = GeneratedReportModel(
                    dataset_id=state.dataset_id,
                    report_type=req.report_type,
                    format="markdown",
                    file_path=str(target_path),
                )
                db.add(rec)
                db.commit()
            except Exception:
                db.rollback()

        return {
            "message": "Report generated successfully.",
            "format": "markdown",
            "file_name": file_name,
            "preview": content[:1200],
            "download_url": f"/api/reports/download?file_name={file_name}",
        }

    elif fmt == "csv":
        file_name = f"Cleaned_{state.dataset_name}"
        if not file_name.endswith(".csv"):
            file_name += ".csv"
        target_path = ReportService.export_csv(df, file_name=file_name)
        return {
            "message": "CSV dataset exported successfully.",
            "format": "csv",
            "file_name": file_name,
            "download_url": f"/api/reports/download?file_name={file_name}",
        }

    elif fmt == "excel":
        file_name = f"Export_{state.dataset_name.split('.')[0]}.xlsx"
        target_path = ReportService.export_excel(df, file_name=file_name)
        return {
            "message": "Excel workbook generated successfully.",
            "format": "excel",
            "file_name": file_name,
            "download_url": f"/api/reports/download?file_name={file_name}",
        }

    raise HTTPException(status_code=400, detail=f"Unsupported format '{fmt}'.")


@router.get("/download")
def download_generated_report(file_name: str = Query(..., description="File name in reports directory")):
    """Download generated artifact from the reports directory."""
    target_path = REPORTS_DIR / file_name
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Requested report file not found.")

    media_type = "application/octet-stream"
    if file_name.endswith(".csv"):
        media_type = "text/csv"
    elif file_name.endswith(".md") or file_name.endswith(".txt"):
        media_type = "text/markdown"
    elif file_name.endswith(".xlsx"):
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    return FileResponse(
        path=str(target_path),
        filename=file_name,
        media_type=media_type,
    )

