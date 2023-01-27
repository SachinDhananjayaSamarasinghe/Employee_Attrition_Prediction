from flask import Flask,render_template,url_for,request
import numpy as np
import pandas as pd 
import jsonify
import requests
import pickle
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
	return render_template('AttritionPrediction.html')

@app.route('/predict', methods = ["POST"])
def predict():
    if request.method == "POST":
        Age = request.form["Age"]
        Gender = request.form["Gender"]
        MaritalStatus = request.form["MaritalStatus"]
        DistanceFromHome = request.form["DistanceFromHome"]
        TotalWorkingYears = request.form["TotalWorkingYears"]
        NumCompaniesWorked = request.form["NumCompaniesWorked"]
        MonthlyIncome = request.form["MonthlyIncome"]
        JobRole = request.form["JobRole"]
        YearsAtCompany = request.form["YearsAtCompany"]
        YearsWithCurrManager = request.form["YearsWithCurrManager"]
        YearsInCurrentRole = request.form["YearsInCurrentRole"]
        YearsSinceLastPromotion = request.form["YearsSinceLastPromotion"]
        JobSatisfaction = request.form["JobSatisfaction"]
        EnvironmentSatisfaction = request.form["EnvironmentSatisfaction"]
        RelationshipSatisfaction = request.form["RelationshipSatisfaction"]
        WorkLifeBalance = request.form["WorkLifeBalance"]
     
        data = [Age, Gender, MaritalStatus, DistanceFromHome, TotalWorkingYears, NumCompaniesWorked, MonthlyIncome, JobRole, YearsAtCompany, YearsWithCurrManager, 
                                        YearsInCurrentRole, YearsSinceLastPromotion, JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance]
        data = [np.array(data)]
        pred = model.predict_proba(data)
        output = '{0:.{1}f}'.format(pred[0][0]*100,2)
        print(output)
        if output>str(50):
            print("Employee will not leave the company.")
            return render_template('NegativeOutcome.html', prediction_text="{}%".format(output))
        else:
            print("Employee will leave the company.")
            return render_template('PositiveOutcome.html', prediction_text="{}%".format(output))

if __name__ == "__main__":
    app.run(debug = True)
    