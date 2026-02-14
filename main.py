import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="IT Salary Predictor",
                   page_icon="💼",
                   layout="wide")

st.title("💼 IT Salary Prediction System")
st.markdown("Predict salary using Machine Learning with visual analytics")

# ------------------------------
# LOAD DATA
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Salary Data.csv")
    df.drop_duplicates(inplace=True)

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

    return df

df = load_data()

# ------------------------------
# DATA OVERVIEW
# ------------------------------
# st.subheader("📊 Dataset Overview")
# col1, col2 = st.columns(2)

# with col1:
#     st.write("Shape of Dataset:", df.shape)

# with col2:
#     st.write("First 5 Rows")
#     st.dataframe(df.head())

# ------------------------------
# VISUALIZATION SECTION
# ------------------------------
# ------------------------------
# VISUALIZATION SECTION
# ------------------------------
st.subheader("📈 Data Visualization")

col1, col2 = st.columns(2)

# Salary Distribution
with col1:
    fig1, ax1 = plt.subplots(figsize=(5,3.5))
    sns.histplot(df['Salary'], kde=True, ax=ax1, color="#4CAF50")
    ax1.set_title("Salary Distribution", fontsize=11)
    ax1.set_xlabel("Salary", fontsize=9)
    ax1.set_ylabel("Frequency", fontsize=9)
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)

# Experience vs Salary
with col2:
    fig2, ax2 = plt.subplots(figsize=(5,3.5))
    sns.scatterplot(x=df['Years of Experience'],
                    y=df['Salary'],
                    ax=ax2,
                    color="#2196F3")
    ax2.set_title("Experience vs Salary", fontsize=11)
    ax2.set_xlabel("Years of Experience", fontsize=9)
    ax2.set_ylabel("Salary", fontsize=9)
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)



# Correlation Heatmap
# Correlation Heatmap
st.subheader("🔥 Correlation Heatmap")

numeric_df = df.select_dtypes(include=np.number)

fig3, ax3 = plt.subplots(figsize=(4.5,3.2))  # Smaller canvas
sns.heatmap(numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.4,
            cbar_kws={"shrink": 0.7},  # Smaller color bar
            ax=ax3)

ax3.set_title("Feature Correlation Matrix", fontsize=10)
ax3.tick_params(labelsize=8)
plt.tight_layout()

st.pyplot(fig3)


# ------------------------------
# PREPROCESSING
# ------------------------------
# ------------------------------
# PREPROCESSING
# ------------------------------

# Encode Gender
df['Gender'] = df['Gender'].map({"Male": 0, "Female": 1})

# One Hot Encoding for Education & Job Title
df = pd.get_dummies(df, columns=['Education Level', 'Job Title'], drop_first=True)

X = df.drop(['Salary'], axis=1)
y = df['Salary']

# Scale numerical columns
scaler = StandardScaler()
X[['Age', 'Years of Experience']] = scaler.fit_transform(
    X[['Age', 'Years of Experience']]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


# ------------------------------
# MODEL PERFORMANCE
# ------------------------------
st.subheader("📊 Model Performance")

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("R² Score", round(r2, 3))
col2.metric("MAE", f"₹ {round(mae,2)}")

# ------------------------------
# SIDEBAR INPUT SECTION
# ------------------------------
st.sidebar.header("🔎 Enter Employee Details")

age = st.sidebar.slider("Age", 18, 60, 25)
experience = st.sidebar.slider("Years of Experience", 0, 40, 3)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
education = st.sidebar.selectbox(
    "Education Level",
    ["Bachelor's", "Master's", "PhD"]
)

job_titles = df.filter(like="Job Title_").columns
job_titles_list = [col.replace("Job Title_", "") for col in job_titles]

job = st.sidebar.selectbox("Job Title", job_titles_list)


# gender_val = 0 if gender == "Male" else 1

# edu_masters = 1 if education == "Master's" else 0
# edu_phd = 1 if education == "PhD" else 0

# scaled_values = scaler.transform([[age, experience]])
# age_scaled = scaled_values[0][0]
# exp_scaled = scaled_values[0][1]

# input_data = pd.DataFrame([[
#     age_scaled,
#     exp_scaled,
#     gender_val,
#     edu_masters,
#     edu_phd
# ]], columns=X.columns)
# Encode gender
gender_val = 0 if gender == "Male" else 1

# Education Encoding
edu_columns = [col for col in X.columns if "Education Level_" in col]
job_columns = [col for col in X.columns if "Job Title_" in col]

input_dict = {}

# Scale numerical
scaled_values = scaler.transform([[age, experience]])
input_dict['Age'] = scaled_values[0][0]
input_dict['Years of Experience'] = scaled_values[0][1]
input_dict['Gender'] = gender_val

# Set education columns
for col in edu_columns:
    input_dict[col] = 1 if col == f"Education Level_{education}" else 0

# Set job title columns
for col in job_columns:
    input_dict[col] = 1 if col == f"Job Title_{job}" else 0

# Create dataframe
input_data = pd.DataFrame([input_dict])
input_data = input_data.reindex(columns=X.columns, fill_value=0)




# ------------------------------
# PREDICTION BUTTON
# ------------------------------
if st.sidebar.button("Predict Salary 💰"):
    prediction = model.predict(input_data)[0]
    st.sidebar.success(f"Predicted Monthly Salary: ₹ {round(prediction,2)}")

    fig4, ax4 = plt.subplots(figsize=(3,2))  # Smaller canvas
    ax4.bar(["Predicted Salary"], [prediction], color="#FF9800")
    ax4.set_ylabel("Salary", fontsize=8)
    ax4.set_title("Predicted Monthly Salary", fontsize=10)
    ax4.tick_params(labelsize=8)
    ax4.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    st.pyplot(fig4)


