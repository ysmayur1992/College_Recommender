from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.pipeline.prediction_pipeline import CustomData1,PredictPipeline,CustomData2

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('homepage.html') 

@app.route('/predict_for_10th',methods=['GET','POST'])
def predict_data():
    if request.method=='GET':
        return render_template('input1.html')
    else:
        data=CustomData1(
            Maths=int(request.form.get('Maths')),
            Science=int(request.form.get('Science')),
            History=int(request.form.get('History')),
            Geography=int(request.form.get('Geography')),
            Language=int(request.form.get('Language')),
            Gender = request.form.get('Gender'),
            Locality = request.form.get('Locality'),
            Budget = int(request.form.get('Budget'))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        res=int(predict_pipeline.predict1(pred_df)[0])
        results = ""
        print("after Predictions {0}".format(res))
        if(res == 0):
            results = "Congratulations! Based on Your Input Scores You will be selected for Section D College! Enroll now to get flat 25% off on enrollment fee"
        elif(res == 1):
            results = "Congratulations! Based on Your Input Scores You will be selected for Section C College! Enroll now to get flat 25% off on enrollment fee"
        elif(res == 2):
            results = "Congratulations! Based on Your Input Scores You will be selected for Section B College! Enroll now to get flat 25% off on enrollment fee"
        else:
            results = "Congratulations! Based on Your Input Scores You will be selected for Section A College! Enroll now to get flat 25% off on enrollment fee"
        return render_template('input1.html',results=results)
    

@app.route('/predict_for_12th',methods=['GET','POST'])
def predict_college():
    if request.method=='GET':
        return render_template('input2.html')
    else:
        data=CustomData2(
            Physics=int(request.form.get('Physics')),
            Maths=int(request.form.get('Maths')),
            Chemistry=int(request.form.get('Chemistry')),
            Biology=int(request.form.get('Biology')),
            Entrance=int(request.form.get('Entrance')),
            Locality = request.form.get('Locality'),
            prev_stream = request.form.get('prev_stream'),
            Budget = int(request.form.get('Budget'))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        res=predict_pipeline.predict2(pred_df)
        results = ""
        print("after Predictions")

        if(res == 0):
            results = "Congratulations! Based on Your Input Scores You will be selected for Section C College! Enroll now to get flat 25% off on enrollment fee"
        elif(res == 1):
            results = "Congratulations! Based on Your Input Scores You will be selected for Section B College! Enroll now to get flat 25% off on enrollment fee"
        elif(res == 2):
            results = "Congratulations! Based on Your Input Scores You will be selected for Diploma Role! Enroll now to get flat 25% off on enrollment fee"
        else:
            results = "Congratulations! Based on Your Input Scores You will be selected for Section A Science College! Enroll now to get flat 25% off on enrollment fee"

        return render_template('input2.html',results=results)
    

if __name__=="__main__":
    app.run(host="0.0.0.0")