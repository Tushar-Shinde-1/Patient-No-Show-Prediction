"""
app.py — Flask backend for the Patient No-Show Predictive System.
Serves the dashboard, prediction page, and REST API endpoints.

Endpoints:
  GET  /                   — Dashboard
  GET  /predict            — Single-patient predict page
  GET  /api/metrics        — Model performance metrics (JSON)
  POST /api/predict        — Single-patient prediction (JSON)
  POST /api/batch-predict  — Bulk patient prediction (JSON array)
"""

from flask import Flask, render_template, request, jsonify
import joblib
import json
import os
import pandas as pd
import numpy as np

app = Flask(__name__)

# ── Load model & metrics once at startup ─────────────────────────────────────
MODEL_PATH = os.path.join('models', 'best_model_pipeline.joblib')
METRICS_PATH = os.path.join('models', 'metrics.json')

model = None
metrics = None
preprocessor = None
features = None
cat_features = None
num_features = None

def _load_artifacts():
    global model, metrics, preprocessor, features, cat_features, num_features
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        if isinstance(data, dict):
            model = data.get('model')
            preprocessor = data.get('preprocessor')
            features = data.get('features')
            cat_features = data.get('categorical_features')
            num_features = data.get('numeric_features')
        else:
            model = data
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)

_load_artifacts()


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def dashboard():
    return render_template('index.html')


@app.route('/predict')
def predict_page():
    return render_template('predict.html')


@app.route('/api/metrics')
def api_metrics():
    if metrics is None:
        return jsonify({'error': 'Metrics not found. Run model_pipeline.py first.'}), 404
    return jsonify(metrics)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Run model_pipeline.py first.'}), 500

    try:
        data = request.get_json(force=True)

        appt_day_of_week = int(data.get('AppointmentDayOfWeek', 0))
        input_df = pd.DataFrame([{
            'Age':                  int(data.get('Age', 30)),
            'Gender':               data.get('Gender', 'F'),
            'Neighbourhood':        data.get('Neighbourhood', 'CENTRO'),
            'Scholarship':          int(data.get('Scholarship', 0)),
            'Hipertension':         int(data.get('Hipertension', 0)),
            'Diabetes':             int(data.get('Diabetes', 0)),
            'Alcoholism':           int(data.get('Alcoholism', 0)),
            'Handcap':              int(data.get('Handcap', 0)),
            'SMS_received':         int(data.get('SMS_received', 0)),
            'WaitingTime':          int(data.get('WaitingTime', 5)),
            'AppointmentDayOfWeek': appt_day_of_week,
            'AppointmentMonth':     int(data.get('AppointmentMonth', 5)),
            'IsWeekend':            1 if appt_day_of_week >= 5 else 0,
        }])

        if preprocessor is not None:
            X_enc = preprocessor.transform(input_df)
            cat_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features))
            encoded_names = num_features + cat_names
            X_df = pd.DataFrame(X_enc, columns=encoded_names)
            X_final = X_df[features].values
        else:
            X_final = input_df

        prob = float(model.predict_proba(X_final)[0][1])
        pred = int(model.predict(X_final)[0])

        if prob < 0.35:
            risk = 'Low'
        elif prob < 0.65:
            risk = 'Medium'
        else:
            risk = 'High'

        return jsonify({
            'probability': round(prob, 4),
            'prediction': pred,
            'label': 'No-Show' if pred == 1 else 'Will Attend',
            'risk': risk,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/batch-predict', methods=['POST'])
def api_batch_predict():
    """
    Bulk prediction endpoint.
    Accepts a JSON array of patient objects and returns predictions for each.

    Request body:
        [ { patient fields... }, { patient fields... }, ... ]

    Response:
        {
          "total": N,
          "predictions": [ { probability, prediction, label, risk }, ... ],
          "summary": { total, no_show_count, attend_count, risk_breakdown }
        }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded. Run model_pipeline.py first.'}), 500

    try:
        patients = request.get_json(force=True)
        if not isinstance(patients, list):
            return jsonify({'error': 'Request body must be a JSON array of patient objects.'}), 400
        if len(patients) == 0:
            return jsonify({'error': 'Empty patient list supplied.'}), 400
        if len(patients) > 10000:
            return jsonify({'error': 'Batch size exceeds maximum of 10,000 records per request.'}), 400

        results = []
        for data in patients:
            appt_day_of_week = int(data.get('AppointmentDayOfWeek', 0))
            input_df = pd.DataFrame([{
                'Age':                  int(data.get('Age', 30)),
                'Gender':               data.get('Gender', 'F'),
                'Neighbourhood':        data.get('Neighbourhood', 'CENTRO'),
                'Scholarship':          int(data.get('Scholarship', 0)),
                'Hipertension':         int(data.get('Hipertension', 0)),
                'Diabetes':             int(data.get('Diabetes', 0)),
                'Alcoholism':           int(data.get('Alcoholism', 0)),
                'Handcap':              int(data.get('Handcap', 0)),
                'SMS_received':         int(data.get('SMS_received', 0)),
                'WaitingTime':          int(data.get('WaitingTime', 5)),
                'AppointmentDayOfWeek': appt_day_of_week,
                'AppointmentMonth':     int(data.get('AppointmentMonth', 5)),
                'IsWeekend':            1 if appt_day_of_week >= 5 else 0,
            }])

            if preprocessor is not None:
                X_enc = preprocessor.transform(input_df)
                cat_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features))
                encoded_names = num_features + cat_names
                X_df = pd.DataFrame(X_enc, columns=encoded_names)
                X_final = X_df[features].values
            else:
                X_final = input_df.values

            prob = float(model.predict_proba(X_final)[0][1])
            pred = int(model.predict(X_final)[0])

            if prob < 0.35:
                risk = 'Low'
            elif prob < 0.65:
                risk = 'Medium'
            else:
                risk = 'High'

            results.append({
                'patient_id':  data.get('PatientId', None),
                'probability': round(prob, 4),
                'prediction':  pred,
                'label':       'No-Show' if pred == 1 else 'Will Attend',
                'risk':        risk,
            })

        no_show_count = sum(1 for r in results if r['prediction'] == 1)
        summary = {
            'total':         len(results),
            'no_show_count': no_show_count,
            'attend_count':  len(results) - no_show_count,
            'risk_breakdown': {
                'Low':    sum(1 for r in results if r['risk'] == 'Low'),
                'Medium': sum(1 for r in results if r['risk'] == 'Medium'),
                'High':   sum(1 for r in results if r['risk'] == 'High'),
            }
        }

        return jsonify({'total': len(results), 'predictions': results, 'summary': summary})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)
