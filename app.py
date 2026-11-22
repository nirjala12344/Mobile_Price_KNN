
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Mobile Price Prediction",
    page_icon="📱",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp {
    background-color: #f4f7fc;
}

.main-title {
    font-size: 36px;
    font-weight: bold;
    color: #173b70;
}

.subtitle {
    color: #64748b;
    font-size: 16px;
}

.result-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    border-left: 6px solid #2563eb;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
    text-align: center;
}

.result-title {
    color: #64748b;
    font-size: 18px;
}

.result-value {
    color: #2563eb;
    font-size: 30px;
    font-weight: bold;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
    height: 45px;
    font-weight: bold;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background-color: #173b70;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return joblib.load("knn_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("Model not found. Please run python train_model.py first.")
    st.stop()

# ---------------- SIDEBAR ----------------

st.sidebar.title("📱 Mobile Price AI")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Price Prediction", "Model Performance", "About Project"]
)

st.sidebar.divider()

st.sidebar.info(
    "Machine Learning Project\n\n"
    "Algorithm: K-Nearest Neighbors (KNN)"
)

# ---------------- HOME PAGE ----------------

if page == "Home":

    st.markdown(
        '<div class="main-title">Mobile Price Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Smart mobile price classification using Machine Learning</div>',
        unsafe_allow_html=True
    )

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Welcome!")

        st.write(
            "This application predicts the price category of a mobile "
            "phone based on its hardware specifications."
        )

        st.write(
            "Enter mobile features such as RAM, battery power, camera, "
            "memory and screen details to get a prediction."
        )

        if st.button("Start Price Prediction"):
            st.session_state["page"] = "Price Prediction"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="result-card">
            <h1>📱</h1>
            <h3>Smart Prediction</h3>
            <p>Powered by KNN Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Project Features")

    c1, c2, c3 = st.columns(3)

    c1.info("📊 Machine Learning\n\nKNN Classification")
    c2.info("📱 Mobile Details\n\n20 Input Features")
    c3.info("🎯 Prediction\n\n4 Price Categories")

# ---------------- PRICE PREDICTION PAGE ----------------

elif page == "Price Prediction":

    st.markdown(
        '<div class="main-title">Mobile Price Prediction</div>',
        unsafe_allow_html=True
    )

    st.write("Enter the mobile phone specifications below.")

    st.divider()

    with st.form("prediction_form"):

        st.subheader("Battery and Performance")

        c1, c2, c3 = st.columns(3)

        with c1:
            battery_power = st.number_input(
                "Battery Power (mAh)", 500, 2000, 1500
            )

            ram = st.number_input(
                "RAM (MB)", 256, 4000, 2000
            )

        with c2:
            clock_speed = st.number_input(
                "Clock Speed", 0.5, 3.0, 1.5
            )

            n_cores = st.number_input(
                "Number of Cores", 1, 8, 4
            )

        with c3:
            mobile_wt = st.number_input(
                "Mobile Weight (g)", 80, 200, 150
            )

            m_dep = st.number_input(
                "Mobile Depth", 0.1, 1.0, 0.5
            )

        st.divider()

        st.subheader("Camera and Memory")

        c1, c2, c3 = st.columns(3)

        with c1:
            fc = st.number_input("Front Camera (MP)", 0, 19, 5)
            pc = st.number_input("Primary Camera (MP)", 0, 20, 10)

        with c2:
            int_memory = st.number_input("Internal Memory (GB)", 2, 64, 32)
            px_height = st.number_input("Pixel Height", 0, 1960, 800)

        with c3:
            px_width = st.number_input("Pixel Width", 500, 2000, 1200)
            talk_time = st.number_input("Talk Time (hours)", 2, 20, 10)

        st.divider()

        st.subheader("Connectivity and Screen")

        c1, c2, c3 = st.columns(3)

        with c1:
            blue = st.selectbox("Bluetooth", [0, 1])
            dual_sim = st.selectbox("Dual SIM", [0, 1])
            four_g = st.selectbox("4G Support", [0, 1])

        with c2:
            three_g = st.selectbox("3G Support", [0, 1])
            wifi = st.selectbox("WiFi", [0, 1])
            touch_screen = st.selectbox("Touch Screen", [0, 1])

        with c3:
            sc_h = st.number_input("Screen Height (cm)", 5, 20, 12)
            sc_w = st.number_input("Screen Width (cm)", 0, 18, 6)

        st.divider()

        predict_button = st.form_submit_button(
            "🔍 Predict Mobile Price",
            use_container_width=True
        )

    if predict_button:

        input_data = pd.DataFrame([{
            "battery_power": battery_power,
            "blue": blue,
            "clock_speed": clock_speed,
            "dual_sim": dual_sim,
            "fc": fc,
            "four_g": four_g,
            "int_memory": int_memory,
            "m_dep": m_dep,
            "mobile_wt": mobile_wt,
            "n_cores": n_cores,
            "pc": pc,
            "px_height": px_height,
            "px_width": px_width,
            "ram": ram,
            "sc_h": sc_h,
            "sc_w": sc_w,
            "talk_time": talk_time,
            "three_g": three_g,
            "touch_screen": touch_screen,
            "wifi": wifi
        }])

        try:
            input_data = input_data[model.feature_names_in_]

            prediction = model.predict(input_data)[0]

            price_ranges = {
                0: "Low Cost",
                1: "Medium Cost",
                2: "High Cost",
                3: "Very High Cost"
            }

            st.divider()
            st.subheader("Prediction Result")

            st.markdown(f"""
            
<style>

/* Main website background */
.stApp {
    background-color: #f4f7fc !important;
    color: #1e293b !important;
}

/* Main text colour */
.stApp p,
.stApp span,
.stApp label,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stMarkdown {
    color: #1e293b !important;
}

/* Main title */
.main-title {
    font-size: 36px;
    font-weight: bold;
    color: #173b70 !important;
}

/* Subtitle */
.subtitle {
    color: #475569 !important;
    font-size: 16px;
}

/* Result card */
.result-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    border-left: 6px solid #2563eb;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
    text-align: center;
}

.result-title {
    color: #475569 !important;
    font-size: 18px;
}

.result-value {
    color: #2563eb !important;
    font-size: 30px;
    font-weight: bold;
}

/* Buttons */
div.stButton > button,
div.stFormSubmitButton > button {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 8px;
    border: none;
    height: 45px;
    font-weight: bold;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background-color: #173b70 !important;
    color: white !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #e8effa !important;
}

</style>
""", unsafe_allow_html=True)


            st.success("Prediction completed successfully!")

            st.caption(
                "Note: This result is a predicted price category, "
                "not an exact mobile selling price."
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------- MODEL PERFORMANCE PAGE ----------------

elif page == "Model Performance":

    st.title("📊 Model Performance")

    st.write("View the evaluation results of your trained KNN model.")

    st.divider()

    try:
        data = pd.read_csv("dataset/train.csv")

        st.metric("Total Dataset Records", len(data))

        if st.button("Show Performance Graphs"):

            st.subheader("Confusion Matrix")

            try:
                st.image(
                    "confusion_matrix.png",
                    caption="KNN Confusion Matrix",
                    use_container_width=True
                )
            except Exception:
                st.warning("Run train_model.py to generate the confusion matrix.")

            st.subheader("Performance Comparison")

            try:
                st.image(
                    "performance_graph.png",
                    caption="KNN Performance Graph",
                    use_container_width=True
                )
            except Exception:
                st.warning("Run train_model.py to generate the performance graph.")

    except Exception:
        st.warning("Dataset file not found.")

# ---------------- ABOUT PAGE ----------------

elif page == "About Project":

    st.title("ℹ️ About Project")

    st.write("""
    ### Mobile Price Range Classification Using KNN

    This project uses Machine Learning to predict the price
    category of mobile phones based on their specifications.

    **Algorithm Used:** K-Nearest Neighbors (KNN)

    **Programming Language:** Python

    **Libraries Used:**
    - Pandas
    - NumPy
    - Scikit-learn
    - Matplotlib
    - Streamlit
    - Joblib

    **Price Categories:**
    - Category 0: Low Cost
    - Category 1: Medium Cost
    - Category 2: High Cost
    - Category 3: Very High Cost

    The application provides an interactive interface
    for mobile price classification.
    """)

st.sidebar.caption("Mobile Price Prediction | KNN Project")
