import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application
## import ridge regressor model and standard scaler
ridge_model=pickle.load(open("models/ridge1.pkl","rb"))
# pickle file object is created
standard_scaler=pickle.load(open('models/scaler1.pkl','rb'))

# import pickle
# scaler = pickle.load(open('models/scaler1.pkl', 'rb'))
# print(type(scaler))


# 1 : route for home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST','GET'])
def predict_datapoint():
    if request.method=='POST':
        # 1 get the data from the form
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        #2 convert data into numpy arr
        input_data=np.array([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])

        #3 scale/Standarized the data
        new_scaled_data=standard_scaler.transform(input_data)
        #4 get prediction from the model
        fwi_prediction=ridge_model.predict(new_scaled_data)
        # ans in list formate to hmko list ka phla element chahiye

        return render_template('home.html',result=fwi_prediction[0])
    else:
        return render_template('home.html')




if __name__=="__main__":
    app.run(host='0.0.0.0')