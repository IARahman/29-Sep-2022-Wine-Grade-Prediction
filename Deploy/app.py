from pickletools import long1
import streamlit as st
import pandas as pd
import pickle
import feature_engine

model = pickle.load(open('Wine_Classifier.pkl','rb'))

st.header('Wine Grade Prediction')
type                    = st.selectbox('Type of wine:', ['White', 'Red'])
fixed_acidity           = st.number_input('Tartaric acid              (gram / liter):')
volatile_acidity        = st.number_input('Acetic acid                 (gram / liter)')
citric_acid             = st.number_input('Citric acid                (gram / liter):')
residual_sugar          = st.number_input('Residual sugar             (gram / liter):')
chlorides               = st.number_input('Natrium Chloride           (gram / liter):')
free_sulfur_dioxide     = st.number_input('Free Sulphur Dioxide   (miligram / liter):')
total_sulfur_dioxide    = st.number_input('Total  Sulphur Dioxide (miligram / liter):')
density                 = st.number_input('Density                (gram / mililiter):')
pH                      = st.number_input('Acidity                              (pH):')
sulphates               = st.number_input('Kalium Sulphate            (gram / liter):')
alcohol                 = st.number_input('Alcohol content                       (%):')

if st.button('Submit'):
    '''
    cat_cols = ['type']
    num_cols = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
    
    cat_df = pd.DataFrame([[type]], columns = cat_cols)
    num_df = pd.DataFrame([[fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol]], columns = num_cols)
    
    X = pd.concat([cat_df,num_df], axis = 1)
    '''

    X = pd.DataFrame([[type,fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol]],
    columns = ['type','fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'])

    pred = model.predict(X)

    st.text(f'Wine grade: {pred[0]}')