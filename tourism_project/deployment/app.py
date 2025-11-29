import os
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Config
REPO_ID = "surnellas/Visit-With-Us"
MODEL_FILENAME = "best_tourism_model_v1.joblib"
DATA_FILENAME = "tourism.csv"
CLASSIFICATION_THRESHOLD = 0.45

st.title("Visit-With-Us — Wellness Package Purchase Prediction")
st.write(
    "Enter customer details below. The model predicts the probability that the customer "
    "will purchase the Wellness Tourism Package."
)

# Feature lists (used by the model)
numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

# Try to download dataset from HF to extract sensible options and ranges
defaults = {}
options = {}
try:
    local_data = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=DATA_FILENAME, token=os.environ.get("HF_TOKEN"))
    template_df = pd.read_csv(local_data)
    # Convert object columns to category for safer unique values
    for c in categorical_features:
        if c in template_df.columns:
            options[c] = sorted(template_df[c].astype(str).unique().tolist())
    for n in numeric_features:
        if n in template_df.columns:
            defaults[n] = {
                "min": int(template_df[n].min()),
                "max": int(template_df[n].max()),
                "mean": float(template_df[n].median()),
            }
except Exception:
    # Fallback defaults if we cannot download dataset
    options = {
        "TypeofContact": ["Company Invited", "Self Enquiry"],
        "Occupation": ["Salaried", "Small Business", "Free Lancer", "Other"],
        "Gender": ["Male", "Female"],
        "ProductPitched": ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"],
        "MaritalStatus": ["Single", "Married", "Divorced", "Unmarried"],
        "Designation": ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
    }
    defaults = {
        "Age": {"min": 18, "max": 80, "mean": 35},
        "CityTier": {"min": 1, "max": 3, "mean": 2},
        "DurationOfPitch": {"min": 1, "max": 60, "mean": 10},
        "NumberOfPersonVisiting": {"min": 1, "max": 10, "mean": 3},
        "NumberOfFollowups": {"min": 0, "max": 12, "mean": 3},
        "PreferredPropertyStar": {"min": 1, "max": 5, "mean": 3},
        "NumberOfTrips": {"min": 0, "max": 20, "mean": 2},
        "Passport": {"min": 0, "max": 1, "mean": 1},
        "PitchSatisfactionScore": {"min": 1, "max": 5, "mean": 3},
        "OwnCar": {"min": 0, "max": 1, "mean": 1},
        "NumberOfChildrenVisiting": {"min": 0, "max": 5, "mean": 0},
        "MonthlyIncome": {"min": 0, "max": 200000, "mean": 30000},
    }

# UI inputs for numeric features
st.sidebar.header("Numeric inputs")
user_inputs = {}
for n in numeric_features:
    conf = defaults.get(n, {"min": 0, "max": 1000, "mean": 0})
    step = 1 if isinstance(conf["mean"], int) or n != "MonthlyIncome" else 1
    if n in ["Passport", "OwnCar"]:
        # Use selectbox for binary features
        user_inputs[n] = st.sidebar.selectbox(n, options=[0, 1], index=int(conf["mean"]))
    else:
        # number_input with reasonable range
        if n == "MonthlyIncome":
            user_inputs[n] = st.sidebar.number_input(
                n,
                min_value=int(conf["min"]),
                max_value=int(conf["max"]) if conf["max"] > 0 else 1_000_000,
                value=int(conf["mean"]),
                step=step
            )
        else:
            user_inputs[n] = st.sidebar.number_input(
                n,
                min_value=int(conf["min"]),
                max_value=int(conf["max"]) if conf["max"] > 0 else 10000,
                value=int(conf["mean"]),
                step=1,
            )

# UI inputs for categorical features
st.sidebar.header("Categorical inputs")
for c in categorical_features:
    vals = options.get(c)
    if vals:
        user_inputs[c] = st.sidebar.selectbox(c, vals)
    else:
        # If we don't know categories, allow free text
        user_inputs[c] = st.sidebar.text_input(c, "")

# Assemble input as DataFrame (matching training columns)
input_df = pd.DataFrame([user_inputs])

# Ensure categorical dtype for relevant cols
for c in categorical_features:
    if c in input_df.columns:
        input_df[c] = input_df[c].astype("category")

# st.subheader("Input preview")
# st.write(input_df.T)

st.subheader("Input preview")
preview_df = pd.DataFrame({
    "Feature": input_df.columns,
    "Value": [str(v) for v in input_df.iloc[0].values],
})
st.table(preview_df)

# Load model (download from HF hub)
model = None
load_error = None
try:
    model_path = hf_hub_download(repo_id=REPO_ID, repo_type="model", filename=MODEL_FILENAME, token=os.environ.get("HF_TOKEN"))
    model = joblib.load(model_path)
except Exception as e:
    load_error = str(e)

if load_error:
    st.error("Failed to load model from Hugging Face Hub. Check HF_TOKEN and network.\n\n" + load_error)
else:
    if st.button("Predict purchase probability"):
        # Ensure ordering of columns matches model's expected features
        ordered_cols = numeric_features + categorical_features
        # Some environments may store y columns as dataframes; ensure all columns present
        missing = [c for c in ordered_cols if c not in input_df.columns]
        if missing:
            st.error(f"Missing features required by model: {missing}")
        else:
            X_input = input_df[ordered_cols].copy()
            proba = model.predict_proba(X_input)[:, 1][0]
            pred = int(proba >= CLASSIFICATION_THRESHOLD)

            st.metric("Purchase Probability", f"{proba:.3f}")
            st.metric("Predicted Purchase", "Yes" if pred == 1 else "No")
            st.write(
                "Notes: probability threshold = "
                + str(CLASSIFICATION_THRESHOLD)
                + ". Adjust threshold for sensitivity/precision tradeoff."
            )
