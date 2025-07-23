
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def load_data():
    df = pd.read_csv("adult 3.csv")
    df = df.replace('?', pd.NA).dropna()
    encoders = {}
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'gender', 'native-country']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    income_le = LabelEncoder()
    df['income'] = income_le.fit_transform(df['income'])
    encoders['income'] = income_le
    return df, encoders

df, encoders = load_data()
X = df.drop('income', axis=1)
y = df['income']
model = RandomForestClassifier()
model.fit(X, y)

st.title("💰 Employee Salary Prediction App")
st.write("Predict whether an employee earns **more than $50K per year** based on their profile.")

with st.form("employee_form"):
    st.header("Enter Employee Details")
    age = st.number_input("Age", min_value=18, max_value=90, value=30)
    workclass = st.selectbox("Workclass", encoders['workclass'].classes_)
    education = st.selectbox("Education", encoders['education'].classes_)
    marital_status = st.selectbox("Marital Status", encoders['marital-status'].classes_)
    occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
    relationship = st.selectbox("Relationship", encoders['relationship'].classes_)
    race = st.selectbox("Race", encoders['race'].classes_)
    gender = st.selectbox("Gender", encoders['gender'].classes_)
    hours_per_week = st.number_input("Hours Per Week", min_value=1, max_value=99, value=40)
    native_country = st.selectbox("Native Country", encoders['native-country'].classes_)
    capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
    education_num = st.number_input("Education Num", min_value=1, max_value=20, value=10)
    submitted = st.form_submit_button("Predict Income")

if submitted:
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [encoders['workclass'].transform([workclass])[0]],
        'education': [encoders['education'].transform([education])[0]],
        'marital-status': [encoders['marital-status'].transform([marital_status])[0]],
        'occupation': [encoders['occupation'].transform([occupation])[0]],
        'relationship': [encoders['relationship'].transform([relationship])[0]],
        'race': [encoders['race'].transform([race])[0]],
        'gender': [encoders['gender'].transform([gender])[0]],
        'hours-per-week': [hours_per_week],
        'native-country': [encoders['native-country'].transform([native_country])[0]],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'education-num': [education_num]
    })
    prediction = model.predict(input_data)
    predicted_income = encoders['income'].inverse_transform(prediction)[0]
    if predicted_income == '>50K':
        st.success("✅ Predicted: Income is > $50K")
    else:
        st.warning("⚠️ Predicted: Income is <= $50K")
