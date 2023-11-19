import numpy as np
import pandas as pd
import pickle
from flask import Flask , request , url_for , render_template , app , jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = ['area', 'PN', 'WC', 'floor', 'Tien_ich']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Xác định các cột định danh và định lượng
categorical_features = ['Tinh/TP', 'Quan/Huyen']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Kết hợp các transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])



def load_model(path , way = 1):
    if way == 1:
        total = pickle.load(open(path, "rb"))
    ## Or way two:
    elif way == 2:
        with open(path, "rb") as f:
            total = pickle.load(f)
    
    model = total["model"]

    return model


app = Flask(__name__)



@app.route("/", methods = ['GET' , 'POST'])
def home():
    return render_template('index.html')

@app.route("/predict" , methods = ['POST'])
def predict():
    bedrooms = request.form['PN']
    restrooms = request.form['WC']
    floors = request.form['Floors']
    provinces = request.form['Provinces']
    tien_ich = request.form['Tien_ich']
    area = request.form['Areas']
    quan_huyen = request.form['Quan/Huyen']


    input_values = pd.DataFrame([[provinces , quan_huyen , bedrooms , restrooms , area , floors , tien_ich]] , 
                                columns= ["Tinh/TP"	,"Quan/Huyen"	,"PN"	,"WC",	"area",	"floor"	,"Tien_ich"])
    # input_values = preprocessor(input_values)

    output = model.predict(input_values)[0]
    # print(output)
    return render_template('predict.html' , prediction_text = f"~{output:.1f}")

# [["Hà Nội" , "Đống Đa" , 2 , 1 , 12 , 2 , 2]]


if __name__ == "__main__":
    # print(model1 , model)
    path = "model.pkl" 
    model= load_model(path  , 1)
    app.run(debug= True)
# print(model)