import pandas as pd
from sklearn.model_selection import train_test_split
import json

def load_and_clean_data(data_path, target_column, test_size, random_state):
    # Step 1: Load the raw CSV into a DataFrame (like an Excel sheet in code)
    df = pd.read_csv(data_path)

    # Step 2: Drop customerID — it's a unique identifier, not a useful signal
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Step 3: TotalCharges is sometimes stored as blank text instead of a number
    # (happens for brand-new customers with 0 tenure). Force it to numeric,
    # turning any bad values into NaN, then fill those with 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Step 4: Convert the target column from "Yes"/"No" text to 1/0 numbers
    df[target_column] = df[target_column].map({"Yes": 1, "No": 0})

    # Step 5: One-hot encode all remaining text (categorical) columns.
    # Example: a "Contract" column with values Month-to-month/One year/Two year
    # becomes three separate 0/1 columns. Models need numbers, not text.
    df = pd.get_dummies(df, drop_first=True)

    # Step 6: Split into features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Step 7: Split into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test

def save_feature_columns(X_train, path="../models/feature_columns.json"):
    # Save the exact column order the model was trained on.
    # This lets us line up any future single request to match it exactly.
    with open(path, "w") as f:
        json.dump(list(X_train.columns), f)