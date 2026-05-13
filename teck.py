import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Title
st.title("KNN Classification and Regression on Iris Dataset")

# Load Dataset
iris = load_iris()

x = iris.data
y = iris.target

# Dataset info
st.subheader("Dataset Preview")
df = pd.DataFrame(x, columns=iris.feature_names)
df["target"] = y
st.dataframe(df.head())

# Sidebar options
st.sidebar.header("Model Parameters")

n_neighbors = st.sidebar.slider("Select K value", 1, 15, 5)

metric = st.sidebar.selectbox(
    "Select Distance Metric",
    ["euclidean", "manhattan", "minkowski"]
)

# For Minkowski distance
p_value = st.sidebar.slider("Select p value", 1, 5, 2)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# -------------------------------
# KNN CLASSIFICATION
# -------------------------------
st.header("KNN Classification")

k_classifier = KNeighborsClassifier(
    n_neighbors=n_neighbors,
    metric=metric,
    p=p_value
)

# Train model
k_classifier.fit(X_train, y_train)

# Prediction
y_pred_class = k_classifier.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred_class)

st.success(f"Classification Accuracy: {accuracy:.2f}")

# Show predictions
class_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred_class
})

st.subheader("Classification Predictions")
st.dataframe(class_df)

# -------------------------------
# KNN REGRESSION
# -------------------------------
st.header("KNN Regression")

k_regressor = KNeighborsRegressor(
    n_neighbors=n_neighbors,
    metric=metric,
    p=p_value
)

# Train model
k_regressor.fit(X_train, y_train)

# Prediction
y_pred_reg = k_regressor.predict(X_test)

# Regression Metrics
mse = mean_squared_error(y_test, y_pred_reg)
mae = mean_absolute_error(y_test, y_pred_reg)
r2 = r2_score(y_test, y_pred_reg)

st.subheader("Regression Evaluation Metrics")

st.write(f"Mean Squared Error (MSE): {mse:.4f}")
st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
st.write(f"R2 Score: {r2:.4f}")

# Show predictions
reg_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred_reg
})

st.subheader("Regression Predictions")
st.dataframe(reg_df)