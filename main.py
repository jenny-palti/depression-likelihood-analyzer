import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import os
import matplotlib.pyplot as plt
import seaborn as sns



def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def inspect_data(df: pd.DataFrame) -> None:
    print("Shape (rows, columns):", df.shape)
    print("\nColumn types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    for column in df.select_dtypes(exclude="number").columns:
        print(f"\nValues in {column}:")
        print(df[column].value_counts())
    print("\nSummary of the number columns:")
    print(df.describe().T.to_string())


def check_unusual_values(df: pd.DataFrame) -> None:
    print("Students with Current_semester = 0:", (df["Current_semester"] == 0).sum())
    print("Students with Current_semester above 8:", (df["Current_semester"] > 8).sum())
    print("Students with Cost_without_study = 0:", (df["Cost_without_study"] == 0).sum())
    print("\nHow many students in each semester:")
    print(df["Current_semester"].value_counts().sort_index())



def plot_depression_by_group(df: pd.DataFrame, column: str, order=None) -> None:
    data = df.copy()
    data["Depression_yes"] = data["Depression"].map({"No": 0, "Yes": 1})
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=data, x=column, y="Depression_yes", order=order, errorbar=None)
    plt.ylabel("Share of students with Depression = Yes")
    plt.title(f"Depression by {column}")
    plt.tight_layout()
    plt.savefig(f"figures/depression_by_{column}.png")
    plt.close()


def prepare_features_and_target(df: pd.DataFrame):
    y = df["Depression"].map({"No": 0, "Yes": 1})
    X = df.drop(columns=["Home_division", "Depression"])
    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def build_model(X):
    text_columns = X.select_dtypes(exclude="number").columns.tolist()
    encoder = ColumnTransformer(
        transformers=[
            ("text_to_numbers", OneHotEncoder(handle_unknown="ignore"), text_columns)
        ],
        remainder="passthrough",
    )

    return Pipeline(
        steps=[
            ("encode", encoder),
            (
                "tree_model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions, target_names=["No", "Yes"]))
    return predictions


def main():
    df = load_data("data/Student_Depression_Dataset.csv")
    inspect_data(df)
    check_unusual_values(df)
    plot_depression_by_group(df, "Health_condition", order=["Good", "Normal", "Abnormal", "Critical"])
    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = build_model(X)
    trained_model = train_model(model, X_train, y_train)
    evaluate_model(trained_model, X_test, y_test)


if __name__ == "__main__":
    main()