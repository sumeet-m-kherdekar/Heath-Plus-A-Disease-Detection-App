import joblib
from flask import Flask, render_template, request
import numpy as np
app = Flask(__name__)

@app.route('/')
def enter1():
    return render_template('index.html')

@app.route('/main_page')
def main_page():
    return render_template('main.html')
    
@app.route('/liver')
def liver():
    return render_template('liver.html')
@app.route('/kidney')
def kidney():
    return render_template('kidney.html')
@app.route('/heart')
def heart():
    return render_template('heart.html')
@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/bmi')
def bmi():
    return render_template('bmi.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/bmi_pred',methods=['POST','GET'])
def bmi_pred():
    my_dict = request.form 
    result = round(float(my_dict['wt'])/(float(my_dict['ht'])**2)*10000,1)
    return render_template('bmi.html', pred = result)

@app.route('/kidney_pred',methods=['POST','GET'])
def kidney_pred():
    try:
        my_dict = request.form 
        arr = [my_dict['age'],my_dict['bp'],my_dict['sg'],my_dict['al'],my_dict['su'],my_dict['rbc'],my_dict['pc'],my_dict['pcc'],my_dict['ba']]
        arr = [np.array(arr)]
        loaded_model = joblib.load('Kidney_model.h5')
        result = loaded_model.predict(arr)
        if result[0] == 1:
            return render_template('kidney.html', pred = 'Disease detected, you should consult a doctor !')
        else:
            return render_template('kidney.html', pred = 'All good, You Are Healthy.')
    except:
            return render_template('kidney.html', pred = 'Please enter all values')

@app.route('/heart_pred',methods=['POST','GET'])
def heart_pred():
    try:
        my_dict = request.form 
        arr = [my_dict['age'],my_dict['gender'],my_dict['cp'],my_dict['rbp'],my_dict['sc'],my_dict['fbs'],my_dict['recg'],my_dict['mhr'],my_dict['ang'],my_dict['op'],my_dict['ep'],my_dict['mv'],my_dict['thal']]
        arr = [np.array(arr)]
        loaded_model = joblib.load('Heart_model.h5')
        result = loaded_model.predict(arr)
        if result[0] == 1:
            return render_template('heart.html', pred = 'Disease detected, you should consult a doctor !')
        else:
            return render_template('heart.html', pred = 'All good, You Are Healthy.')
    except:
            return render_template('heart.html', pred = 'Please enter all values')
        

@app.route('/diabetes_pred',methods=['POST','GET'])
def diabetes_pred():
    try:
        my_dict = request.form 
        arr = [my_dict['pr'],my_dict['glu'],my_dict['bp'],my_dict['st'],my_dict['in'],my_dict['bmi'],my_dict['dpf'],my_dict['age']]
        arr = [np.array(arr)]
        loaded_model = joblib.load('Diabetes_model.h5')
        result = loaded_model.predict(arr)
        if result[0] == 1:
            return render_template('diabetes.html', pred = 'Disease detected, you should consult a doctor !')
        else:
            return render_template('diabetes.html', pred = 'All good, you are Healthy.')
    except:
            return render_template('diabetes.html', pred = 'Please enter all values')


@app.route('/liver_pred',methods=['POST','GET'])
def liver_pred():
    try:
        my_dict = request.form 
        arr = [my_dict['age'],my_dict['gender'],my_dict['tb'],my_dict['db'],my_dict['ap'],my_dict['ala'],my_dict['as'],my_dict['tp'],my_dict['alb'],my_dict['agr']]
        arr = [np.array(arr)]
        loaded_model = joblib.load('Liver_model.h5')
        result = loaded_model.predict(arr)
        if result[0] == 1:
            return render_template('liver.html', pred = 'Disease detected, you should consult a doctor !')
        else:
            return render_template('liver.html', pred = 'All good, You Are Healthy.')
    except:
            return render_template('liver.html', pred = 'Please enter all values')


if __name__ == "__main__":
    app.run(debug=True, port=5000)
