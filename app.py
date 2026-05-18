from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('gradient_boost_model.pkl')

# The 41 columns your model expects
MODEL_COLUMNS = [
    'OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 
    'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 
    'MSZoning_C (all)', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 
    'Utilities_AllPub', 'Utilities_NoSeWa', 'BldgType_1Fam', 'BldgType_2fmCon', 
    'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'Heating_Floor', 
    'Heating_GasA', 'Heating_GasW', 'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 
    'KitchenQual_Ex', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 
    'SaleCondition_Abnorml', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 
    'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial', 
    'LandSlope_Gtl', 'LandSlope_Mod', 'LandSlope_Sev'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get numbers from the HTML form
    input_data = {col: 0 for col in MODEL_COLUMNS}
    
    input_data['OverallQual'] = int(request.form.get('OverallQual', 5))
    input_data['GrLivArea'] = float(request.form.get('GrLivArea', 1500))
    input_data['YearBuilt'] = int(request.form.get('YearBuilt', 2000))
    # ... add other numeric fields as needed ...

    # 2. Handle Categorical logic (One-Hot Encoding)
    zoning = request.form.get('MSZoning')
    kitchen = request.form.get('KitchenQual')
    
    if f"MSZoning_{zoning}" in input_data:
        input_data[f"MSZoning_{zoning}"] = 1
    if f"KitchenQual_{kitchen}" in input_data:
        input_data[f"KitchenQual_{kitchen}"] = 1
    
    # 3. Create DataFrame and Predict
    df = pd.DataFrame([input_data])[MODEL_COLUMNS]
    prediction = model.predict(df)[0]
    
    return render_template('index.html', prediction_text=f'Estimated Price: ${prediction:,.2f}')

if __name__ == "__main__":
    app.run(debug=True)