from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
import pickle 
import pandas as pd 
import numpy as np
import tensorflow as tf
import streamlit as st

#load trained model
model=load_model('model.h5')

#load encoder & scaler 
with open('geo_encoder.pkl','rb') as file:
    label_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

st.title('Customer churn prediction')

#user inputs
geography=st.selectbox('Geography',label_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,120)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Est.Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has credit card',[0,1])
is_active_member=st.selectbox('Is active member',[0,1])

input_data=({
    'CreditScore':[credit_score],
    'Geography':[geography],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

geo_encoded=label_encoder_geo.transform([input_data['Geography']]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=label_encoder_geo.get_feature_names_out(['Geography']))    

input_data_df=pd.DataFrame(input_data)
input_data_df=pd.concat([input_data_df.drop("Geography",axis=1),geo_encoded_df],axis=1)

scaled_input_data=scaler.transform(input_data_df)


#Predict
prediction=model.predict(scaled_input_data)
predict_probability=prediction[0][0]

st.write(f"Churn probability: {predict_probability:.2f}")
if predict_probability>0.5:
    st.write(f'{predict_probability} Customer is likely to churn')
else:
    st.write(f'{predict_probability} Customer is not likely to churn') 