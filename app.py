import streamlit as st
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

# Title for the app
st.title("Iris Data Explorer")

# Load the Iris dataset from scikit-learn
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target

# Show the first few rows of the dataframe
st.subheader("First rows of the Iris dataset")
st.write(df.head())

# Display summary statistics for numeric columns
st.subheader("Summary statistics")
st.write(df.describe())

# Allow user to select numeric columns for visualization
numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
selected_columns = st.multiselect(
    "Select numeric columns for visualization",
    numeric_columns,
    default=numeric_columns[:2],
)

if selected_columns:
    st.subheader("Histogram of selected columns")
    for col in selected_columns:
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=20, alpha=0.7)
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)

    # If at least two columns are selected, draw a scatter plot of the first two
    if len(selected_columns) >= 2:
        x_col, y_col = selected_columns[:2]
        st.subheader(f"Scatter plot: {x_col} vs {y_col}")
        fig, ax = plt.subplots()
        ax.scatter(df[x_col], df[y_col], c=df["target"], cmap="viridis", alpha=0.7)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        st.pyplot(fig)
else:
    st.write("Please select at least one numeric column to visualize.")
