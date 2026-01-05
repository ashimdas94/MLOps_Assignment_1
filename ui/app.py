import streamlit as st
import requests
import time

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide"
)

st.title("Heart Disease Prediction")
st.write(
    "Enter patient details to get a real-time prediction from the "
    "FastAPI service deployed on Kubernetes."
)

st.divider()

# -----------------------------
# Two-column page layout
# -----------------------------
left_col, right_col = st.columns([2, 1])  # Inputs wider than results

# -----------------------------
# LEFT: Input Form
# -----------------------------
with left_col:
    with st.form("prediction_form"):
        st.subheader("Patient Details")

        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 1, 120, 50)
        with c2:
            sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])

        c3, c4 = st.columns(2)
        with c3:
            cp = st.number_input("Chest Pain Type (0–3)", 0, 3, 1)
        with c4:
            trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)

        c5, c6 = st.columns(2)
        with c5:
            chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        with c6:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

        c7, c8 = st.columns(2)
        with c7:
            restecg = st.number_input("Resting ECG (0–2)", 0, 2, 0)
        with c8:
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)

        c9, c10 = st.columns(2)
        with c9:
            exang = st.selectbox("Exercise Induced Angina", [0, 1])
        with c10:
            oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)

        c11, c12 = st.columns(2)
        with c11:
            slope = st.number_input("Slope (0–2)", 0, 2, 1)
        with c12:
            ca = st.number_input("Major Vessels (0–3)", 0, 3, 0)

        thal = st.number_input("Thalassemia (1–3)", 1, 3, 2)

        submit = st.form_submit_button("🔮 Predict")

# -----------------------------
# RIGHT: Prediction Result
# -----------------------------
with right_col:
    st.subheader("Prediction Result")

    if submit:
        payload = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }

        try:
            start_time = time.time()
            response = requests.post(API_URL, json=payload, timeout=5)
            latency_ms = round((time.time() - start_time) * 1000, 2)

            if response.status_code == 200:
                result = response.json()

                prediction = result["prediction"]
                probability = result["probability"]
                message = result["message"]

                if prediction == 1:
                    st.error("⚠️ Heart Disease Detected")
                else:
                    st.success("✅ No Heart Disease Detected")

                st.markdown(f"**Message:** {message}")
                st.metric("Prediction Probability", f"{probability:.2f}")
                st.caption(f"Latency: {latency_ms} ms")

            else:
                st.error(f"API Error ({response.status_code})")
                st.code(response.text)

        except requests.exceptions.ConnectionError:
            st.error("❌ API not reachable. Is port-forward running?")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
    else:
        st.info("Submit the form to see prediction results here.")

st.divider()
st.caption(
    "FastAPI on Kubernetes | Prometheus & Grafana Monitoring | Streamlit UI"
)
