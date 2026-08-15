import streamlit as st
import pandas as pd
import joblib


# ==========================================
# LOAD TRAINED MODEL AND PREPROCESSOR
# ==========================================

preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("delivery_time_model.pkl")


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Delivery Time Predictor",
    page_icon="🚚",
    layout="centered"
)


# ==========================================
# TITLE
# ==========================================

st.title("🚚 Food Delivery Time Predictor")

st.write(
    "Predict food delivery time using a trained Neural Network."
)

st.divider()


# ==========================================
# INPUT SECTION
# ==========================================

st.header("📦 Enter Order Details")


market_id = st.selectbox(
    "Market ID",
    [1, 2, 3, 4, 5, 6]
)


store_primary_category = st.text_input(
    "Store Primary Category",
    value="american"
)


order_protocol = st.selectbox(
    "Order Protocol",
    [1, 2, 3, 4, 5, 6, 7]
)


total_items = st.number_input(
    "Total Items",
    min_value=1,
    value=3
)


subtotal = st.number_input(
    "Subtotal",
    min_value=0,
    value=250
)


num_distinct_items = st.number_input(
    "Number of Distinct Items",
    min_value=1,
    value=2
)


min_item_price = st.number_input(
    "Minimum Item Price",
    min_value=0,
    value=50
)


max_item_price = st.number_input(
    "Maximum Item Price",
    min_value=0,
    value=150
)


total_onshift_partners = st.number_input(
    "Total Onshift Partners",
    min_value=0,
    value=10
)


total_busy_partners = st.number_input(
    "Total Busy Partners",
    min_value=0,
    value=5
)


total_outstanding_orders = st.number_input(
    "Total Outstanding Orders",
    min_value=0,
    value=5
)


order_hour = st.slider(
    "Order Hour",
    min_value=0,
    max_value=23,
    value=20
)


order_day_of_week = st.slider(
    "Day of Week (0 = Monday, 6 = Sunday)",
    min_value=0,
    max_value=6,
    value=4
)


order_month = st.slider(
    "Order Month",
    min_value=1,
    max_value=12,
    value=2
)


is_weekend = st.selectbox(
    "Is Weekend?",
    [0, 1]
)


# ==========================================
# PREDICTION
# ==========================================

if st.button("🚀 Predict Delivery Time"):

    # Create dataframe with exactly the
    # same feature names used during training

    input_data = pd.DataFrame([{

        "market_id": market_id,

        "store_primary_category":
            store_primary_category,

        "order_protocol":
            order_protocol,

        "total_items":
            total_items,

        "subtotal":
            subtotal,

        "num_distinct_items":
            num_distinct_items,

        "min_item_price":
            min_item_price,

        "max_item_price":
            max_item_price,

        "total_onshift_partners":
            total_onshift_partners,

        "total_busy_partners":
            total_busy_partners,

        "total_outstanding_orders":
            total_outstanding_orders,

        "order_hour":
            order_hour,

        "order_day_of_week":
            order_day_of_week,

        "order_month":
            order_month,

        "is_weekend":
            is_weekend

    }])


    # ======================================
    # PREPROCESS INPUT
    # ======================================

    processed_data = preprocessor.transform(
        input_data
    )


    # ======================================
    # MAKE PREDICTION
    # ======================================

    prediction = model.predict(
        processed_data
    )[0]


    # ======================================
    # DISPLAY RESULT
    # ======================================

    st.success(
        f"🚚 Estimated Delivery Time: "
        f"{prediction:.2f} minutes"
    )

    st.info(
        f"⏱️ Approximately "
        f"{prediction / 60:.2f} hours"
    )