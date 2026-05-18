from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model and preprocessor artifacts
try:
    preprocessor = joblib.load('fraud_preprocessor.joblib')
    model = joblib.load('best_fraud_xgboost_model.joblib')
except FileNotFoundError:
    print("❌ Error: Could not find model or preprocessor files. Make sure they are in the same folder as app.py")

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    probability_text = None
    alert_class = ""
    
    if request.method == 'POST':
        try:
            # 1. Capture form data submitted from the UI HTML page
            form_data = {
                'amount': float(request.form['amount']),
                'transaction_hour': int(request.form['transaction_hour']),
                'merchant_category': request.form['merchant_category'],
                'foreign_transaction': int(request.form['foreign_transaction']),
                'location_mismatch': int(request.form['location_mismatch']),
                'device_trust_score': int(request.form['device_trust_score']),
                'velocity_last_24h': int(request.form['velocity_last_24h']),
                'cardholder_age': int(request.form['cardholder_age'])
            }
            
            # 2. Convert raw form entries into a Pandas DataFrame format
            input_df = pd.DataFrame([form_data])
            
            # 3. Process the raw fields using your fitted pipeline transformer
            processed_input = preprocessor.transform(input_df)
            
            # 4. Generate prediction arrays and confidence statistics
            prob = model.predict_proba(processed_input)[0][1] # Probability of being fraud
            pred = model.predict(processed_input)[0]          # 0 or 1
            
            # 5. Build dynamic text outputs for your front-end display
            probability_text = f"Fraud Risk Confidence: {prob * 100:.2f}%"
            
            if pred == 1:
                prediction_text = "🚨 TRANSACTION BLOCKED: High Risk of Fraud detected!"
                alert_class = "danger"
            else:
                prediction_text = "✅ TRANSACTION APPROVED: Low Risk transaction profile."
                alert_class = "success"
                
        except Exception as e:
            prediction_text = f"Error processing request: {str(e)}"
            alert_class = "warning"

    return render_template('index1.html', 
                           prediction=prediction_text, 
                           probability=probability_text, 
                           alert_class=alert_class)

if __name__ == '__main__':
    app.run(debug=True)