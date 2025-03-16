from flask import Flask, render_template, request, jsonify, redirect, flash
import pandas as pd
import smtplib
import joblib
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load the data and model
DATA_PATH = "Cleaned_data.csv"
MODEL_PATH = "RandomForestModel.pkl"

data = pd.read_csv(DATA_PATH)
pipe = joblib.load(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/dataInsights')
def dataInsights():
    return render_template('dataInsights.html')

@app.route('/api/house-price-insights', methods=['GET'])
def house_price_insights():
    """API endpoint to fetch house price insights."""
    avg_price_by_region = data.groupby("region_name")["price"].mean().sort_values(ascending=False).head(5)
    avg_price_per_sqft_by_region = data.groupby("region_name")["value_per_sqft"].mean().sort_values(ascending=False).head(5)

    response = {
        "regions": avg_price_by_region.index.tolist(),
        "avg_prices": avg_price_by_region.values.tolist(),
        "avg_price_per_sqft": avg_price_per_sqft_by_region.values.tolist()
    }
    return jsonify(response)

@app.route('/prediction')
def prediction():
    """Render prediction form with dropdown options."""
    locations = sorted(data["locality_name"].unique())
    regions = sorted(data["region_name"].unique())
    house_types = sorted(data["house_type"].unique())
    return render_template('prediction.html', locations=locations, regions=regions, house_types=house_types)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle house price prediction request."""
    try:
        location = request.form.get('location')
        region = request.form.get('region')
        house_type = request.form.get('house-type')
        area = request.form.get('area', type=float)
        total_rooms = request.form.get('total_rooms', type=int)
        total_beds = request.form.get('total_beds', type=int)
        age = request.form.get('age', type=int)

        if not all([location, region, house_type, area, total_rooms, total_beds, age]):
            return jsonify({"error": "All fields are required!"}), 400

        input_data = pd.DataFrame([[location, region, area, house_type, total_rooms, total_beds, age]],
                                  columns=['locality_name', 'region_name', 'area', 'house_type', 'total_rooms', 'total_beds', 'age'])

        # Predict house price
        prediction = pipe.predict(input_data)[0]
        return jsonify({"predicted_price": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
