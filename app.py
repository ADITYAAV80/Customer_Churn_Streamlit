import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from torch import tensor
import streamlit as st

class NNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(in_features=12,out_features=3)
        self.dense1 = nn.Linear(in_features=3,out_features=2)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(in_features=2,out_features=1)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self,x):

        x = self.input(x)
        x = self.relu(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.sigmoid(x)

        return x

if __name__=="__main__":

    model = NNet()
    model.load_state_dict(torch.load("model_complete.pth"))

    with open("label_encoding_gender.pkl","rb") as file:
        le = pickle.load(file)

    with open("one_hot_encoding_geography.pkl","rb") as file:
        ohe = pickle.load(file)

    with open("standar_scalar.pkl","rb") as file:
        ss = pickle.load(file)
    

    st.title("Customer Churn Prediction")

    st.header("Enter the data to predict")

    st.write("This data set contains details of a bank's customers and the target variable is a binary variable reflecting the fact whether the customer left the bank (closed his account) or he continues to be a customer.")

    geography = st.selectbox("Geography",ohe.categories_[0])
    gender = st.selectbox("Gender",le.classes_)
    age = st.slider("Age",18,92)
    balance = st.number_input("Balance")
    credit_score = st.number_input("Credit Score")
    estimated_salary = st.number_input("Estimated Salary")
    tenure = st.slider("Tenure",0,10)
    number_of_products = st.slider("Number of products",1,4)
    has_cr_card = st.selectbox("Has Credit Card",[0,1])
    is_active_member = st.selectbox("Is Active Memeber",[0,1])


    input_data = {
            "CreditScore":credit_score,
            "Geography":geography,
            "Gender":gender,
            "Age":age,
            "Tenure":tenure,
            "Balance":balance,
            "NumOfProducts":number_of_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember":is_active_member,
            "EstimatedSalary": estimated_salary
    }

    input_df = pd.DataFrame([input_data])
    input_df["Gender"] = le.transform(input_df['Gender'])
    geo_data = ohe.transform(input_df[["Geography"]])
    geo_columns = ohe.get_feature_names_out(["Geography"])
    geography_df = pd.DataFrame(geo_data.toarray(), columns=geo_columns)
    final_df = pd.concat([geography_df,input_df.drop(["Geography"],axis=1)],axis=1)
    x_test = ss.transform(final_df)
    x_test_torch = tensor(x_test,dtype=torch.float32)
    prediction = model(x_test_torch).round()

    if prediction > 0.5:
        st.write("Person is going to leave")
    else:
        st.write("Person is going to stay")