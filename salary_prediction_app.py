import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def load_and_train_model():
    df = pd.read_csv("adult 3.csv")
    df = df.replace('?', pd.NA).dropna()

    le_dict = {}
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                        'relationship', 'race', 'gender', 'native-country']

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    le_income = LabelEncoder()
    df['income'] = le_income.fit_transform(df['income'])

    X = df.drop('income', axis=1)
    y = df['income']

    model = RandomForestClassifier()
    model.fit(X, y)

    return model, le_dict, X.columns

# Load model and encoders
model, le_dict, feature_names = load_and_train_model()

# Streamlit UI
st.title("💰 Employee Salary Prediction")
st.write("Predict whether an employee earns >$50K/year based on their profile.")

with st.form("employee_form"):
    st.header("Enter Employee Details")

    age = st.number_input("Age", min_value=18, max_value=90)
    workclass = st.selectbox("Workclass", le_dict['workclass'].classes_)
    education = st.selectbox("Education", le_dict['education'].classes_)
    marital_status = st.selectbox("Marital Status", le_dict['marital-status'].classes_)
    occupation = st.selectbox("Occupation", le_dict['occupation'].classes_)
    relationship = st.selectbox("Relationship", le_dict['relationship'].classes_)
    race = st.selectbox("Race", le_dict['race'].classes_)
    gender = st.selectbox("Gender", le_dict['gender'].classes_)
    native_country = st.selectbox("Native Country", le_dict['native-country'].classes_)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
        'age': [age],
        'workclass': [le_dict['workclass'].transform([workclass])[0]],
        'education': [le_dict['education'].transform([education])[0]],
        'marital-status': [le_dict['marital-status'].transform([marital_status])[0]],
        'occupation': [le_dict['occupation'].transform([occupation])[0]],
        'relationship': [le_dict['relationship'].transform([relationship])[0]],
        'race': [le_dict['race'].transform([race])[0]],
        'gender': [le_dict['gender'].transform([gender])[0]],
        'native-country': [le_dict['native-country'].transform([native_country])[0]]
    }

    input_df = pd.DataFrame(input_dict)
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("✅ Predicted: >$50K")
    else:
        st.warning("⚠️ Predicted: <=$50K")
