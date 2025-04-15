import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt

# --- App Title ---
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("💼 Employee Attrition Prediction Dashboard")

# --- Main Interaction Choice ---
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["Upload CSV and Analyze", "Enter Details Manually"]
)

# Store unique categorical values and min/max for numerical features globally
categorical_unique_values = {}
numerical_feature_ranges = {}


def manual_input_form(df, scaler, model, le, X_cols):
    st.subheader("🧠 Predict Attrition Manually")
    st.markdown("Enter employee features to predict whether they may leave the company.")

    input_data = {}
    with st.form("manual_form"):
        for col in df.columns:
            if col == "Attrition":
                continue
            if col in categorical_unique_values:
                input_data[col] = st.selectbox(
                    f"{col}", categorical_unique_values[col])
            elif df[col].dtype == "object":
                input_data[col] = st.selectbox(f"{col}", df[col].unique())
            elif col in numerical_feature_ranges:  # Use stored min/max
                min_val = numerical_feature_ranges[col]['min']
                max_val = numerical_feature_ranges[col]['max']
                mean_val = numerical_feature_ranges[col]['mean']
                input_data[col] = st.number_input(f"{col}", min_val, max_val, mean_val)
            else:
                input_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=X_cols, fill_value=0)
        input_scaled = scaler.transform(input_encoded)
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        pred_label = le.inverse_transform([prediction])[0]

        st.success(f"🎯 Prediction: **{pred_label}**")
        st.info(f"📈 Probability of Attrition: **{prob:.2%}**")


if analysis_type == "Upload CSV and Analyze":
    # --- Upload Data ---
    st.sidebar.header("Step 1: Upload Your Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your employee dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 Preview of Uploaded Data")
        st.write(df.head())

        target_col = "Attrition"
        df[target_col].fillna("No", inplace=True)
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_encoded = pd.get_dummies(X, drop_first=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        # Store unique values from categorical columns
        categorical_cols = X.select_dtypes(include="object").columns
        for col in categorical_cols:
            categorical_unique_values[col] = df[col].unique().tolist()

        # Store min, max, and mean for numerical columns
        numerical_cols = X.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            numerical_feature_ranges[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean())
            }

        # Check if the dataset is large enough for a meaningful split
        if len(df) > 50:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
            )
            split_data = True
        else:
            X_train, X_test, y_train, y_test = X_scaled, X_scaled, y_encoded, y_encoded
            st.warning(
                "⚠️ Dataset too small for proper train-test split. Using the same data for training & testing.")
            split_data = False

        use_smote = st.sidebar.checkbox("Apply SMOTE", value=True)

        if use_smote and split_data and len(np.unique(y_train)) > 1:
            try:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            except ValueError as e:
                st.error(f"❌ SMOTE Error: {e}. SMOTE could not be applied. "
                         f"Ensure sufficient samples in minority class after split.")
                use_smote = False
        elif use_smote and not split_data:
            st.warning("⚠️ SMOTE is not applied because the dataset was not split.")
            use_smote = False

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        st.subheader("📊 Model Evaluation")
        y_pred = model.predict(X_test)
        st.write(pd.DataFrame(classification_report(
            y_test, y_pred, output_dict=True)).transpose())

        st.subheader("🧮 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        st.write(pd.DataFrame(cm, index=["Actual No", "Actual Yes"],
                 columns=["Predicted No", "Predicted Yes"]))

        st.subheader("📈 SHAP Explainability")
        if st.button("Show SHAP Summary Plot"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
            shap.summary_plot(shap_values[1], X_encoded, plot_type="bar", show=False)
            st.pyplot(plt.gcf())
            plt.clf()

        manual_input_form(df, scaler, model, le, X_encoded.columns)
    else:
        st.warning("Please upload a dataset with an 'Attrition' column to proceed.")

elif analysis_type == "Enter Details Manually":
    # Placeholder: Load a sample dataset or model if needed for manual input
    # This is necessary to get the column names, scalers, etc.
    try:
        df = pd.read_csv("sample_employee_data.csv")  # Replace with your sample data
        target_col = "Attrition"
        df[target_col].fillna("No", inplace=True)
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_encoded = pd.get_dummies(X, drop_first=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        # Train a model (or load a pre-trained one)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_scaled, y_encoded)

        manual_input_form(df, scaler, model, le, X_encoded.columns)
    except FileNotFoundError:
        st.error("Could not load sample data. Please upload a CSV file first to use manual input.")

st.markdown("---")
st.caption("Made with ❤️ by Arpit Singh")