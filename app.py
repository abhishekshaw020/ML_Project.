import streamlit as st
import pickle
st.title("MPG ML PROJECT")
st.header("Creating my own MPG predictor model")
displacement=st.number_input('Displacement',value=300,placeholder='enter a value for displacement')
horsepower=st.number_input('Horsepower',value=130,placeholder='enter a value for horsepower')
weight=st.number_input('weight',value=3000,placeholder='enter a value for weight')
acceleration= st.number_input('Acceleration',value=12,placeholder='enter a value for acceleration')

loaded_model=pickle.load(open("mpg_regression.sav",'rb'))
prediction=loaded_model.predict([[displacement,horsepower,weight,acceleration]])
st.subheader(f'Predicted mpg value for above parameter is {prediction[0]}')
st.write(prediction)