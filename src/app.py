import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "early_prediction_model.pkl"

st.set_page_config(
    page_title="Student Performance | ML Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 28px;
        border-radius: 14px;
        background: linear-gradient(120deg, #12345a, #286090);
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }

    .hero p {
        color: #e4edf7;
        margin-bottom: 0;
        font-size: 16px;
    }

    .section-heading {
        font-size: 20px;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    /* Dashboard metric cards */
div[data-testid="stMetric"] {
    background: #1e293b !important;
    padding: 22px !important;
    border-radius: 14px !important;
    border: 1px solid #334155 !important;
}

/* Metric labels */
div[data-testid="stMetricLabel"] p {
    color: #cbd5e1 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* Metric values */
div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* Dashboard overview heading */
.dashboard-heading {
    color: #60a5fa !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    margin-bottom: 16px !important;
}
    }

    .footer {
        text-align: center;
        color: #777;
        font-size: 13px;
        padding-top: 25px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("🎓 Student Analytics")

    st.write("Machine Learning Prediction Dashboard")

    st.divider()

    st.subheader("About the Project")

    st.write(
        "This application estimates a student's final academic "
        "grade using a trained Random Forest regression model."
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Algorithm:** Random Forest Regressor")
    st.write("**Prediction type:** Regression")
    st.write("**Target:** Final Grade (G3)")
    st.write("**Grade range:** 0–20")

    st.divider()

    st.caption(
        "Developed as a Machine Learning internship project."
    )


# --------------------------------------------------
# 5. APPLICATION HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🎓 Student Performance Prediction System</h1>
    <p>
        An intelligent Machine Learning application for estimating
        student academic performance using personal, social and
        educational information.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 6. PROJECT OVERVIEW
# --------------------------------------------------

st.markdown(
    '<h2 class="dashboard-heading">Dashboard Overview</h2>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Machine Learning Model", "Random Forest")

with col2:
    st.metric("Prediction Target", "G3 Grade")

with col3:
    st.metric("Maximum Grade", "20 / 20")

st.divider()


# --------------------------------------------------
# 7. LOAD MODEL SAFELY
# --------------------------------------------------

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please check that "
        "early_prediction_model.pkl exists inside the models folder."
    )
    st.stop()

try:
    model = load_model()

except Exception as error:
    st.error("Unable to load the trained model.")
    st.exception(error)
    st.stop()


# --------------------------------------------------
# 8. STUDENT INPUT FORM
# --------------------------------------------------

st.subheader("Student Information")

st.write(
    "Enter the student's information below to generate "
    "an estimated final academic grade."
)

with st.form("student_prediction_form"):

    # PERSONAL INFORMATION

    st.markdown("### 1. Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Gender", ["F", "M"])
        age = st.slider("Age", 15, 22, 17)

    with col2:
        address = st.selectbox("Address", ["U", "R"])
        famsize = st.selectbox("Family Size", ["GT3", "LE3"])
        Pstatus = st.selectbox("Parents' Status", ["T", "A"])

    with col3:
        Medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4])
        Fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4])

    st.divider()

    # FAMILY AND EDUCATION

    st.markdown("### 2. Family and Educational Background")

    col1, col2 = st.columns(2)

    with col1:
        Mjob = st.selectbox(
            "Mother's Job",
            ["teacher", "health", "services", "at_home", "other"]
        )

        Fjob = st.selectbox(
            "Father's Job",
            ["teacher", "health", "services", "at_home", "other"]
        )

        guardian = st.selectbox(
            "Guardian",
            ["mother", "father", "other"]
        )

        reason = st.selectbox(
            "Reason for Choosing School",
            ["home", "reputation", "course", "other"]
        )

    with col2:
        traveltime = st.selectbox("Travel Time", [1, 2, 3, 4])
        studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4])
        failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])
        absences = st.number_input(
            "Number of Absences",
            min_value=0,
            max_value=100,
            value=0
        )

    st.divider()

    # STUDENT SUPPORT

    st.markdown("### 3. Student Support and Activities")

    col1, col2 = st.columns(2)

    with col1:
        schoolsup = st.selectbox("Extra School Support", ["yes", "no"])
        famsup = st.selectbox("Family Educational Support", ["yes", "no"])
        paid = st.selectbox("Extra Paid Classes", ["yes", "no"])
        activities = st.selectbox("Extra Activities", ["yes", "no"])

    with col2:
        nursery = st.selectbox("Attended Nursery", ["yes", "no"])
        higher = st.selectbox("Wants Higher Education", ["yes", "no"])
        internet = st.selectbox("Internet Access", ["yes", "no"])
        romantic = st.selectbox("In a Relationship", ["yes", "no"])

    st.divider()

    # LIFESTYLE

    st.markdown("### 4. Lifestyle and Well-being")

    col1, col2, col3 = st.columns(3)

    with col1:
        famrel = st.slider("Family Relationship", 1, 5, 4)
        freetime = st.slider("Free Time", 1, 5, 3)

    with col2:
        goout = st.slider("Going Out", 1, 5, 3)
        health = st.slider("Health", 1, 5, 3)

    with col3:
        Dalc = st.slider("Weekday Alcohol Consumption", 1, 5, 1)
        Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 1)

    st.divider()

    predict_button = st.form_submit_button(
        "🚀 Predict Student Performance",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# 9. PREDICTION
# --------------------------------------------------

if predict_button:

    input_data = pd.DataFrame([{
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences
    }])

    try:
        prediction = float(model.predict(input_data)[0])

        if not pd.notna(prediction):
            raise ValueError("The model returned an invalid prediction.")

        prediction = max(0, min(20, prediction))

        # RESULT DISPLAY

        st.divider()
        st.header("📊 Prediction Results")

        result_col1, result_col2 = st.columns([1, 2])

        with result_col1:
            st.metric(
                "Estimated Final Grade",
                f"{prediction:.2f} / 20"
            )

            st.progress(prediction / 20)

        with result_col2:
            st.subheader("Performance Interpretation")

            if prediction >= 15:
                st.success("High predicted performance")

            elif prediction >= 10:
                st.info("Moderate predicted performance")

            else:
                st.warning(
                    "Potential academic risk — consider additional support."
                )

            st.write(
                "This interpretation is based on the estimated grade "
                "and is not a formal academic assessment."
            )

        with st.expander("View submitted student information"):
            st.dataframe(
                input_data.T.rename(columns={0: "Input Value"}),
                use_container_width=True
            )

        st.caption(
            "This is a machine learning estimate, not a guaranteed "
            "examination result. Predictions may be affected by "
            "dataset limitations and model uncertainty."
        )

    except Exception as error:
        st.error("An error occurred while generating the prediction.")
        st.exception(error)


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()

st.markdown("""
<div class="footer">
    Student Performance Prediction System |
    Built with Python, Scikit-learn and Streamlit
</div>
""", unsafe_allow_html=True)