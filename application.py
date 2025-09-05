from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize Flask application
application = Flask(__name__)
app = application


# Route for Home Page
@app.route("/")
def index():
    """
    Renders the landing page (index.html).
    """
    return render_template("index.html")
 


# Route for Prediction
@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint(): 
    """
    Handles input form submission for prediction.
    - On GET: Renders the form (home.html).
    - On POST: Accepts form data, preprocesses it, and returns predictions.
    """
    if request.method == "GET":
        # Show input form page when user visits /predictdata directly
        return render_template("home.html")
    else:
        # Collect data submitted from the HTML form
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = float(request.form.get('reading_score')),
            writing_score = float(request.form.get('writing_score'))
        )

        # Convert user input into DataFrame for preprocessing
        pred_df = data.get_data_as_data_frame()
        print(pred_df)                # Debug: print input DataFrame
        print("Before Prediction")    # Debugging checkpoint

        # Initialize prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")       # Debugging checkpoint

        # Generate prediction
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")     # Debugging checkpoint

        # Render the same page with prediction result
        return render_template('home.html', results=results[0])


# Run Flask Application
if __name__=="__main__":
    # Run app on all network interfaces with debugging enabled
    app.run(host="0.0.0.0")  
