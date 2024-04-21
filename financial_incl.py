import pandas as pd
import streamlit as st
import joblib
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Financial_inclusion_dataset.csv')

st.markdown("<h1 style = 'color: #DC6B19; text-align: center; font-size: 60px; font-family: Monospace'>FINANCIAL BANK ACCOUNT PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #898121; text-align: center; font-family: Serif '>Built by Era King</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)

#get an image
st.image('pngwing.com (5).png', caption = 'Built by Era King')

#Add Project problem statement
st.markdown("<h2 style = 'color: #898121; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)

st.markdown("The study aims to predict customer behavior in opening bank accounts, leveraging machine learning techniques. By analyzing demographic data, transaction history, and customer interactions, the model seeks to identify patterns that influence account opening decisions. Understanding these factors can help banks tailor their marketing strategies, improve customer targeting, and enhance overall customer experience. The goal is to develop a predictive model that accurately forecasts the likelihood of customers opening an account, enabling banks to make data-driven decisions and optimize their customer acquisition strategies.</p>", unsafe_allow_html=True)

# Sidebar design (to put what you want on the side)
st.sidebar.image('pngwing.com (6).png')

# markdown is for space
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.header('Project Data')
st.dataframe(data, use_container_width= True)

age = st.sidebar.number_input("age_of_respondent", placeholder='insert your numbers...')
household_size = st.sidebar.number_input("household_size", placeholder='insert your numbers...')
job_type = st.sidebar.selectbox ('job_type', data['job_type'].unique())
education = st.sidebar.selectbox ('education_level', data['education_level'].unique())
marital_status = st.sidebar.selectbox ('marital_status', data['marital_status'].unique())
country = st.sidebar.selectbox ('country', data['country'].unique())
location_type = st.sidebar.selectbox ('location_type', data['location_type'].unique())

# user input dataframe
user_input = pd.DataFrame()
user_input['age_of_respondent'] = [age]
user_input['household_size'] = [household_size]
user_input['job_type'] = [job_type]
user_input['education_level'] = [education]
user_input['marital_status'] = [marital_status]
user_input['country'] = [country]
user_input['location_type'] = [location_type]

st.markdown("<b>", unsafe_allow_html = True)
st.header('input Variable')
st.dataframe(user_input, use_container_width = True)

job_type = joblib.load('job_type_encoder.pkl')
education = joblib.load('education_level_encoder.pkl')
marital_status = joblib.load('marital_status_encoder.pkl')
country = joblib.load('country_encoder.pkl')
location_type = joblib.load('location_type_encoder.pkl')

# tarnsform user input according to training scale and encoding
user_input['job_type'] = job_type.transform(user_input[['job_type']])
user_input['education_level'] = education.transform(user_input[['education_level']])
user_input['marital_status'] = marital_status.transform(user_input[['marital_status']])
user_input['country'] = country.transform(user_input[['country']])
user_input['location_type'] = location_type.transform(user_input[['location_type']])

st.dataframe(user_input)
model = joblib.load('financialinclusionpredictionmodel.pkl')
predict = model.predict(user_input)

if st.button('Check your Account Status'):
    if predict[0] == 'No':
        st.error(f"Unfortunately, you do not have a Bank Account.")
        st.image('pngwing.com (-).png', width = 300)
    else:
        st.success(f"Great!....You have a Bank Account.")
        st.image('pngwing.com (!).png', width = 300)
        

