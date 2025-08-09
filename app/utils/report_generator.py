# report_generator.py
import os
import json
import sys
from typing import List, Union
from app.config import REPORT_DIR
from app.exception_logging.exception import CustomException

try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except:
    WEASYPRINT_AVAILABLE = False


class ReportGenerator:
    def _load_summary(self, summary_path: Union[str, dict]):
        if isinstance(summary_path, dict):
            return summary_path
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                return json.load(f)
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    def generate_html_report(self, summaries: List[Union[str, dict]], output_path=None):
        try:
            if output_path is None:
                output_path = os.path.join(REPORT_DIR, "all_models_report.html")

            html = ["<html><head><title>Model Evaluation Report</title></head><body>"]
            html.append("<h1>AutoML Model Evaluation Report</h1>")

            for summary in summaries:
                data = self._load_summary(summary)
                html.append(f"<h2>{data['model_name']}</h2>")
                html.append("<pre>" + json.dumps(data["metrics"], indent=2) + "</pre>")
                for k, path in data.get("artifacts", {}).items():
                    if path and os.path.exists(path):
                        html.append(f"<h3>{k}</h3><img src='{path}' width='500'>")

            html.append("</body></html>")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                f.write("\n".join(html))
            return output_path
        except Exception as e:
            raise CustomException(e, sys)

    def generate_pdf_report(self, summaries: List[Union[str, dict]], output_path=None):
        if not WEASYPRINT_AVAILABLE:
            raise RuntimeError("WeasyPrint not installed")
        html_path = self.generate_html_report(summaries, output_path="temp.html")
        HTML(html_path).write_pdf(output_path or os.path.join(REPORT_DIR, "all_models_report.pdf"))
