
# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Import machine learning libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder  # For data preprocessing
from sklearn.ensemble import RandomForestClassifier  # For prediction model
from sklearn.model_selection import train_test_split  # For splitting datasets
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score, roc_curve
)  # For evaluation metrics
from imblearn.over_sampling import SMOTE  # For balancing data

# Import collections library (optional, used for debugging resampled dataset)
from collections import Counter

# Load and Preprocess the Dataset
@st.cache_data
def load_and_preprocess_data():
    # Load the dataset
    data = pd.read_csv('./Employee-Attrition.csv')
    data.drop_duplicates(inplace=True)
    data.columns = data.columns.str.lower().str.replace(" ", "_")
    
    # Drop irrelevant columns
    drop_columns = ['employeecount', 'standardhours', 'over18']
    data.drop(columns=drop_columns, axis=1, inplace=True)
    
    # Feature engineering
    def categorize_tenure(years):
        if years < 3:
            return 'Short-Term'
        elif 3 <= years <= 7:
            return 'Mid-Term'
        else:
            return 'Long-Term'

    data['tenurecategory'] = data['yearsatcompany'].apply(categorize_tenure)
    data['performancescore'] = (
        0.4 * data['jobsatisfaction'] +
        0.3 * data['performancerating'] +
        0.3 * data['jobinvolvement']
    )
    data['workloadindicator'] = data['overtime'] * 1

    # One-hot encoding for department and jobrole
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
    one_hot_encoded = enc.fit_transform(data[['department', 'jobrole']])
    data = pd.concat([data, one_hot_encoded], axis=1)
    data.drop(columns=['department', 'jobrole'], inplace=True)

    return data

# Load and preprocess the data
data = load_and_preprocess_data()

# Extract one-hot encoded department and jobrole columns
department_columns = [col for col in data.columns if col.startswith('department_')]
department_names = [col.replace('department_', '') for col in department_columns]

jobrole_columns = [col for col in data.columns if col.startswith('jobrole_')]
jobrole_names = [col.replace('jobrole_', '') for col in jobrole_columns]

# Sidebar Navigation
menu = st.sidebar.radio("Select a section:", ["Dashboard","Visualization" ,"Predict Attrition"])

# Dashboard Section
# Dashboard Section
if menu == "Dashboard":
    st.header("Welcome to Employee Insights")
    st.markdown("""
        <style>
        .stApp{
            background: linear-gradient(to right, #fbc2eb, #a6c1ee);

            color: #333;       
        }       
        </style>

        """,unsafe_allow_html=True)
    st.write("This section provides a quick overview of employee data and attrition trends.")
    st.subheader("Employee Metrics")
    st.markdown("- Employee satisfaction levels")
    st.markdown("- Departmental performance insights")

#Visualization Section

elif menu == "Visualization":
    st.header("Employee Insights - Visualization")
    st.write("Here, you can filter the employee data and identify trends.")

    # Dropdown for Department filter
    selected_department = st.selectbox("Filter by Department:", options=["All"] + department_names)

    # Apply filter for department
    if selected_department != "All":
        filtered_data = data[data[f'department_{selected_department}'] == 1]
    else:
        filtered_data = data

    # Dropdown for Job Role filter
    selected_jobrole = st.selectbox("Filter by Job Role:", options=["All"] + jobrole_names)

    # Apply filter for job role
    if selected_jobrole != "All":
        filtered_data = filtered_data[filtered_data[f'jobrole_{selected_jobrole}'] == 1]

    st.write("Filtered Data Shape:", filtered_data.shape)
    st.dataframe(filtered_data)
    
        # High-Risk Employees Section
    st.subheader("High-Risk Employees ⚠️")

    # Debugging: Test without filters
    high_risk_employees = data[data['attrition'] == "Yes"]
    if high_risk_employees.empty:
        st.write("No high-risk employees in the entire dataset.")
    else:
        st.write(high_risk_employees[['employeenumber', 'performancescore', 'jobsatisfaction']])

    # High Job Satisfaction
    st.subheader("High Job Satisfaction 😊")
    high_job_satisfaction = filtered_data[filtered_data['jobsatisfaction'] >= 4]
    st.dataframe(high_job_satisfaction[['employeenumber', 'jobsatisfaction', 'attrition']])

    # High Performance Score
    st.subheader("High Performance Score 💪")
    high_performance_score = filtered_data[filtered_data['performancescore'] >= 3]
    st.dataframe(high_performance_score[['employeenumber', 'performancescore', 'jobsatisfaction']])

# Prediction Section
else:
    st.header("Predict Employee Attrition Risk")

    # Load the trained model and scaler
    with open('random_forest_model.pkl', 'rb') as file:
        rf_model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('feature_columns.pkl', 'rb') as file:
        feature_columns = pickle.load(file)

    # User inputs for prediction
    overtime = st.selectbox("Overtime:", options=["Yes", "No"])
    stock_option_level = st.selectbox("Stock Option Level:", options=[0, 1, 2, 3])
    monthly_income_options = sorted(data['monthlyincome'].unique())
    monthly_income = st.selectbox("Monthly Income:", options=monthly_income_options)
    distance_from_home_options = sorted(data['distancefromhome'].unique())
    distance_from_home = st.selectbox("Distance from Home (in miles):", options=distance_from_home_options)
    job_satisfaction = st.slider("Job Satisfaction:", min_value=1, max_value=4, value=3)
    job_involvement = st.slider("Job Involvement:", min_value=1, max_value=4, value=3)

    # Prepare input data
    if st.button("Predict"):
        input_data = pd.DataFrame(
            [[overtime, stock_option_level, monthly_income,  
              distance_from_home, job_satisfaction, job_involvement]],
            columns=['overtime', 'stockoptionlevel', 'monthlyincome',  
                     'distancefromhome', 'jobsatisfaction', 'jobinvolvement']
        )

        # Encode categorical variables
        input_data['overtime'] = 1 if overtime == "Yes" else 0
        

        # Add missing features with default values
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Reorder columns to match the model
        input_data = input_data[feature_columns]
        scaled_input = scaler.transform(input_data)

        # Make prediction
        prediction = rf_model.predict(scaled_input)
        st.success(f"Attrition Risk Prediction: {'High Risk' if prediction[0] == 1 else 'Low Risk'}")
