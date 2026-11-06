
import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

st.title("Spambase Classification")
st.write("Comparison of five Machine Learning classifiers")

# Upload test data
uploaded_file = st.file_uploader("Upload test_data.csv", type="csv")

# Select model
models = {
    "Logistic Regression": "Logistic_Regression.joblib",
    "Decision Tree": "Decision_Tree.joblib",
    "KNN": "KNN.joblib",
    "Naive Bayes": "Naive_Bayes.joblib",
    "Support Vector Machine": "Support_Vector_Machine.joblib"
}

model_name = st.selectbox("Select Model", list(models.keys()))

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Test Data")
    st.dataframe(data.head())

    # Separate inputs and target
    X_test = data.drop(columns=["spam"])
    y_test = data["spam"]

    # Load selected trained model
    model = joblib.load("model/" + models[model_name])

    # Make predictions
    predicted = model.predict(X_test)

    # Calculate AUC
    if model_name == "Support Vector Machine":
        scores = model.decision_function(X_test)
    else:
        scores = model.predict_proba(X_test)[:, 1]

    # Display evaluation metrics
    st.subheader("Evaluation Metrics")

    results = {
        "Accuracy": accuracy_score(y_test, predicted),
        "AUC": roc_auc_score(y_test, scores),
        "Precision": precision_score(y_test, predicted),
        "Recall": recall_score(y_test, predicted),
        "F1": f1_score(y_test, predicted),
        "MCC": matthews_corrcoef(y_test, predicted)
    }

    st.dataframe(pd.DataFrame([results]).round(4))

    # Display confusion matrix
    st.subheader("Confusion Matrix")

    matrix = confusion_matrix(y_test, predicted, labels=[0, 1])

    matrix_table = pd.DataFrame(
        matrix,
        index=["Actual Not Spam", "Actual Spam"],
        columns=["Predicted Not Spam", "Predicted Spam"]
    )

    st.dataframe(matrix_table)

    st.success("Evaluation completed successfully")
