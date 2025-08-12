from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import numpy as np
import pandas as pd
import os
from PIL import Image
import json
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
        save_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(save_path)

        # Use DataLoader to load data
        loader = DataLoader(file_path=save_path, target_column="")  # Temporarily empty target
        data = pd.read_csv(save_path)  # Just to avoid empty target_column issue
        column_names = list(data.columns)

        return jsonify(column_names)

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    


@app.route("/run_automl", methods=["POST"])
def run_automl():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        target_column = request.form.get("target_column")
        if not target_column:
            return jsonify({"error": "No target column selected"}), 400

        features_to_drop = request.form.getlist("drop_columns[]")  # checkbox values

        save_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(save_path)

        # Call your main pipeline here
        # You can import and call your main() with parameters
        # For example:
        # main_pipeline(file_path=save_path, target_column=target_column, features_to_drop=drop_columns)

        from app.main import main as run_pipeline
        result = run_pipeline(save_path, target_column, features_to_drop)

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
    app.run(host="0.0.0.0", port=5000, debug= True)  # Set debug=True for development purposes


