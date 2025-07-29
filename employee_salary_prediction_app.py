# employee_salary_prediction_app.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Set Streamlit Page configuration
st.set_page_config(
    page_title="Employee Salary Prediction App",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Employee Salary Prediction Management")
st.markdown("""
This app predicts whether an employee earns more than 50K per year based on demographic and work-related features.
- Upload your employee data in CSV format.
- Explore the data visually.
- Run classification and view model performance.
""")

# File upload section
uploaded_file = st.file_uploader("Upload your employee data CSV", type=["csv"])
if uploaded_file is not None:
    # 1. Data Loading
    df = pd.read_csv(uploaded_file)
    st.subheader("1️⃣ Raw Data Preview")
    st.dataframe(df.head())

    # 2. Data Cleaning
    st.subheader("2️⃣ Data Cleaning")

    # Fill missing values
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        else:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    st.write("Filled missing values with mode (categorical) or median (numerical).")

    # Replace ambiguous values for select columns
    for col in ['workclass', 'occupation']:
        if col in df_clean.columns:
            df_clean[col].replace('?', 'Others', inplace=True)
    st.dataframe(df_clean.head())

    # 3. Visualization
    st.subheader("3️⃣ Data Visualization")
    if "income" in df_clean.columns:
        fig, ax = plt.subplots()
        palette = sns.color_palette("pastel")
        sns.countplot(x="income", data=df_clean, palette=palette, ax=ax)
        ax.set_title("Income Class Distribution")
        st.pyplot(fig)
    else:
        st.warning("No 'income' column detected for target variable.")

    # 4. Encoding
    st.subheader("4️⃣ Encoding Categorical Variables")
    cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    if "income" in cat_cols:
        cat_cols.remove("income")
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        encoders[col] = le

    # Encode target variable
    target_map = {'<=50K': 0, '>50K': 1}
    if "income" in df_clean.columns:
        df_clean["income"] = df_clean["income"].map(target_map)
        st.write("All categorical variables encoded.")
        
        # Show class distribution
        st.write("**Class Distribution:**")
        st.write(pd.Series(df_clean["income"]).value_counts())
    else:
        st.error("Your data must include an 'income' column with values '<=50K' or '>50K'.")
        st.stop()

    # 5. Model Training & Prediction
    st.subheader("5️⃣ Model Training & Evaluation")

    if "income" in df_clean.columns:
        X = df_clean.drop("income", axis=1)
        y = df_clean["income"]

        # Check if we have both classes
        if len(y.unique()) < 2:
            st.error("Dataset contains only one class. Please use data with both income categories (<=50K and >50K).")
            st.stop()

        # Use stratified split to ensure both classes in train/test sets
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            # Fallback if stratification fails
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.write(f"**Model Accuracy:** {acc:.2%}")

        # Show classification report with error handling
        unique_test_classes = sorted(y_test.unique())
        class_names = ['<=50K', '>50K']
        
        if len(unique_test_classes) == 1:
            st.warning(f"⚠️ Warning: Test set contains only one class: {class_names[unique_test_classes[0]]}")
            st.code("Classification report unavailable - insufficient class diversity in test split.")
        else:
            st.code(classification_report(y_test, y_pred, target_names=class_names))

        # Confusion Matrix
        st.write("**Confusion Matrix**")
        try:
            fig2, ax2 = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(
                model, X_test, y_test, display_labels=['<=50K', '>50K'], cmap='Blues', ax=ax2
            )
            st.pyplot(fig2)
        except ValueError:
            st.warning("Confusion matrix unavailable due to single class in test set.")

        # Optional: Predict on custom input
        st.subheader("🔮 Predict Income for New Employee")
        st.write("Enter employee details to predict income category:")
        
        input_data = {}
        col1, col2 = st.columns(2)
        
        with col1:
            for i, col in enumerate(X.columns[:len(X.columns)//2]):
                if col in encoders:
                    unique_classes = list(encoders[col].classes_)
                    val = st.selectbox(f"Select {col}", unique_classes, key=col)
                    input_data[col] = encoders[col].transform([val])[0]
                else:
                    val = st.number_input(f"Enter {col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()), key=col)
                    input_data[col] = val
        
        with col2:
            for col in X.columns[len(X.columns)//2:]:
                if col in encoders:
                    unique_classes = list(encoders[col].classes_)
                    val = st.selectbox(f"Select {col}", unique_classes, key=col)
                    input_data[col] = encoders[col].transform([val])[0]
                else:
                    val = st.number_input(f"Enter {col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()), key=col)
                    input_data[col] = val
        
        if st.button("🎯 Predict Income Class"):
            input_df = pd.DataFrame([input_data])
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]
            
            if pred == 0:
                st.success(f"**Prediction: <=50K** (Confidence: {prob[0]:.2%})")
            else:
                st.success(f"**Prediction: >50K** (Confidence: {prob[1]:.2%})")
    else:
        st.error("Your data must include an 'income' column with values '<=50K' or '>50K'.")
else:
    st.info("📤 Awaiting CSV upload. Please upload your employee data file.")
    
    # Show sample data format
    st.subheader("📋 Expected CSV Format")
    sample_data = {
        'age': [39, 50, 38],
        'workclass': ['State-gov', 'Self-emp-not-inc', 'Private'],
        'education': ['Bachelors', 'Bachelors', 'HS-grad'],
        'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced'],
        'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners'],
        'race': ['White', 'White', 'White'],
        'gender': ['Male', 'Male', 'Male'],
        'hours-per-week': [40, 13, 40],
        'income': ['<=50K', '<=50K', '<=50K']
    }
    st.dataframe(pd.DataFrame(sample_data))
