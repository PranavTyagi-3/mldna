"""A tiny "application" that uses the model mldna fetched. Run from the consumer folder:

    python app.py
"""
import pickle

with open("models/churn/model.pkl", "rb") as f:
    model = pickle.load(f)

customer = [[24, 70.0, 2]]  # tenure_months, monthly_charges, support_calls
risk = model.predict_proba(customer)[0][1]
print(f"Churn risk for a 24-month customer paying 70.00 with 2 support calls: {risk:.0%}")
