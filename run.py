from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import numpy as np
import pandas as pd
import os
from PIL import Image
import json
from google.cloud import storage
from io import BytesIO
from app.config import GCS_BUCKET_NAME
from app.utils.data_loader import DataLoader
import traceback

app = Flask(__name__)

app.secret_key = "asfashdaskhkashk"

@app.route("/")
def home():
    return render_template('index.html')



@app.route("/upload_data", methods=["POST"])
def get_columns():
    try:
        file = request.files["file"]
        bucket_name = GCS_BUCKET_NAME
        blob_name = f"uploads/{file.filename}"
        loader = DataLoader(file_source=None, target_column="")
        loader.upload_files_to_gcs(file, bucket_name, blob_name)

        # Download file from GCS to memory for column extraction
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        data = blob.download_as_bytes()
        df = pd.read_csv(BytesIO(data))
        column_names = list(df.columns)

        return jsonify(column_names)

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    


@app.route("/run_automl", methods=["POST"])
def run_automl():
    try:
        file = request.files.get("file")
        target_column = request.form.get("target_column")
        features_to_drop = request.form.getlist("drop_columns[]")

        bucket_name = GCS_BUCKET_NAME
        blob_name = f"uploads/{file.filename}"
        loader = DataLoader(file_source=None, target_column="")
        loader.upload_files_to_gcs(file, bucket_name, blob_name)
        dataset_name = os.path.splitext(file.filename)[0]

        # Download file from GCS for processing
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        data = blob.download_as_bytes()
        save_path = BytesIO(data)
        save_path.seek(0)  # Reset pointer to start

        from app.main import main as run_pipeline
        result = run_pipeline(save_path, target_column, features_to_drop, dataset_name)

        session["automl_results"] = result

        return jsonify({
            "status": "success",
            "report_url": url_for("view_report")
        })

    except Exception as e:
        print("Error in /run_automl route:")
        traceback.print_exc()  # This will show the exact error in Flask console
        return jsonify({"error": str(e)}), 500
    
    
@app.route("/view_report", methods=["GET"])
def view_report():
    results = session.get("automl_results")
    if not results:
        return "No report available. Please run AutoML first.", 400
    return render_template("report.html", results=results)










if __name__ == "__main__":
    #port=int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=8080)  # Set debug=True for development purposes


