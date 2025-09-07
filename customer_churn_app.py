import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
st.set_page_config(page_title="Bank Customer Churn Predictor", page_icon="🏦")

# Seed for reproducibility
RANDOM_SEED = 42

# Function to generate synthetic customer churn dataset
def generate_data(n_samples=1000, random_state=RANDOM_SEED):
    np.random.seed(random_state)
    data = pd.DataFrame()

    # Features
    data['CreditScore'] = np.random.randint(350, 850, size=n_samples)
    data['Age'] = np.random.randint(18, 90, size=n_samples)
    data['Tenure'] = np.random.randint(0, 11, size=n_samples)  # Years with bank
    data['Balance'] = np.round(np.random.uniform(0, 250000, size=n_samples), 2)
    data['NumOfProducts'] = np.random.randint(1, 5, size=n_samples)
    data['HasCreditCard'] = np.random.choice([0, 1], size=n_samples)
    data['IsActiveMember'] = np.random.choice([0, 1], size=n_samples)
    data['EstimatedSalary'] = np.round(np.random.uniform(10000, 150000, size=n_samples), 2)

    # Creating the target 'Exited' (churn) - somewhat correlated with some features
    # Simplified rule-based approach to simulate churn
    churn_prob = (
        (data['CreditScore'] < 600) * 0.3 +
        (data['Balance'] == 0) * 0.3 +
        (data['IsActiveMember'] == 0) * 0.2 +
        (data['Tenure'] < 3) * 0.1
    )
    churn_prob += np.random.normal(0, 0.05, size=n_samples)  # noise

    data['Exited'] = (churn_prob > 0.4).astype(int)

    return data

# Prepare data for modeling
def prepare_data(df):
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # Scaling numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

@st.cache(allow_output_mutation=True)
def train_model():
    df = generate_data()
    X, y, scaler = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)

    # Test accuracy (for info)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, scaler, acc

def main():
    st.title("Bank Customer Churn Prediction System")
    st.write("""
        Use this app to predict whether a bank customer is likely to churn (leave) or stay.
        Provide the customer's information below and click **Predict**
    """)

    # Train model once and cache it
    model, scaler, accuracy = train_model()

    st.sidebar.header("Input Customer Data")

    def user_input_features():
        credit_score = st.sidebar.slider('Credit Score', 350, 850, 600)
        age = st.sidebar.slider('Age', 18, 90, 30)
        tenure = st.sidebar.slider('Tenure (years with bank)', 0, 10, 3)
        balance = st.sidebar.number_input('Account Balance', min_value=0.0, max_value=250000.0, value=50000.0, step=1000.0)
        num_of_products = st.sidebar.slider('Number of Bank Products', 1, 4, 1)
        has_credit_card = st.sidebar.selectbox('Has Credit Card', options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active_member = st.sidebar.selectbox('Is Active Member', options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=10000.0, max_value=150000.0, value=60000.0, step=1000.0)

        data = {
            'CreditScore': credit_score,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCreditCard': has_credit_card,
            'IsActiveMember': is_active_member,
            'EstimatedSalary': estimated_salary
        }

        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Preprocess input for prediction
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader("Prediction Result")
    churn_label = {0: "Customer Will Stay", 1: "Customer Will Churn"}
    st.write(churn_label[prediction[0]])

    st.subheader("Prediction Probability")
    st.write(f"Probability of Staying: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Churning: {prediction_proba[0][1]:.2f}")

    st.sidebar.markdown("---")
    st.sidebar.write(f"Model accuracy on test data: **{accuracy * 100:.2f}%**")

    st.markdown("""
    ---
    **Note:** This model is trained on synthetic data for demonstration purposes only.
    """)

if __name__ == "__main__":
    main()
