import streamlit as st
import pandas as pd
import joblib


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ==========================================
# Load Model
# ==========================================

@st.cache_resource
def load_model():

    model = joblib.load("model.pkl")

    return model


model = load_model()


# ==========================================
# Title
# ==========================================

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Enter transaction details below to check whether "
    "the transaction is likely to be fraudulent."
)

st.divider()


# ==========================================
# Transaction Information
# ==========================================

st.subheader("Transaction Information")

col1, col2, col3 = st.columns(3)


with col1:

    cc_num = st.number_input(
        "Credit Card Number",
        min_value=0,
        value=123456789012345,
        step=1
    )

    amt = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0,
        step=0.01
    )

    category = st.selectbox(
        "Transaction Category",
        [
            "misc_net",
            "grocery_pos",
            "entertainment",
            "gas_transport",
            "misc_pos",
            "grocery_net",
            "shopping_net",
            "shopping_pos",
            "food_dining",
            "personal_care",
            "health_fitness",
            "home",
            "kids_pets",
            "travel"
        ]
    )


with col2:

    merchant = st.text_input(
        "Merchant",
        "fraud_example_merchant"
    )

    gender = st.selectbox(
        "Gender",
        ["M", "F"]
    )

    zip_code = st.number_input(
        "Customer ZIP Code",
        min_value=0,
        value=10001,
        step=1
    )

    city = st.text_input(
        "City",
        "New York"
    )


with col3:

    state = st.text_input(
        "State",
        "NY"
    )

    job = st.text_input(
        "Job",
        "Software engineer"
    )

    city_pop = st.number_input(
        "City Population",
        min_value=0,
        value=100000,
        step=1
    )


# ==========================================
# Location Information
# ==========================================

st.divider()

st.subheader("Location Information")

col1, col2, col3 = st.columns(3)


with col1:

    lat = st.number_input(
        "Customer Latitude",
        value=40.7128,
        format="%.6f"
    )

    long = st.number_input(
        "Customer Longitude",
        value=-74.0060,
        format="%.6f"
    )


with col2:

    merch_lat = st.number_input(
        "Merchant Latitude",
        value=40.7306,
        format="%.6f"
    )

    merch_long = st.number_input(
        "Merchant Longitude",
        value=-73.9352,
        format="%.6f"
    )


with col3:

    merch_zipcode = st.number_input(
        "Merchant ZIP Code",
        min_value=0,
        value=10001,
        step=1
    )

    unix_time = st.number_input(
        "Unix Transaction Time",
        min_value=0,
        value=1325376018,
        step=1
    )


# ==========================================
# Prediction
# ==========================================

st.divider()

predict_button = st.button(
    "🔍 Detect Fraud",
    type="primary",
    use_container_width=True
)


if predict_button:

    # Create input DataFrame
    input_data = pd.DataFrame({

        "cc_num": [cc_num],

        "merchant": [merchant],

        "category": [category],

        "amt": [amt],

        "gender": [gender],

        "city": [city],

        "state": [state],

        "zip": [zip_code],

        "lat": [lat],

        "long": [long],

        "city_pop": [city_pop],

        "job": [job],

        "unix_time": [unix_time],

        "merch_lat": [merch_lat],

        "merch_long": [merch_long],

        "merch_zipcode": [merch_zipcode]
    })


    # Make prediction
    prediction = model.predict(input_data)[0]


    # Probability
    probability = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            input_data
        )[0]

        probability = probabilities[1]


    # ==========================================
    # Display Result
    # ==========================================

    st.divider()

    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            "🚨 FRAUDULENT TRANSACTION DETECTED"
        )

        if probability is not None:

            st.write(
                f"Fraud Probability: "
                f"**{probability * 100:.2f}%**"
            )

    else:

        st.success(
            "✅ TRANSACTION APPEARS TO BE LEGITIMATE"
        )

        if probability is not None:

            st.write(
                f"Fraud Probability: "
                f"**{probability * 100:.2f}%**"
            )


    # Show entered data

    st.subheader("Transaction Details")

    st.dataframe(
        input_data,
        use_container_width=True
    )