"""Project-level configuration constants."""

import pandas as pd

# Reproducibility anchors
SEED  = 42
TODAY = pd.Timestamp("2025-01-01")

# Default column lists
DEMOGRAPHIC = [
    "age","education","employment_status","job",
    "monthly_income","home_district","owns_home"
]

ALTERNATIVE = [
    "monthly_rent",
    "phone_model","phone_purchase_date",
    "owns_car","car_brand","car_purchase_date",
    "owns_credit_card","monthly_subscription_cost",
    "online_shopping_frequency","social_media_active"
]

TARGET = "delinquency_fl"
