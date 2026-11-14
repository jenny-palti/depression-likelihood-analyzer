import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("data/Student_Depression_Dataset.csv")
df = df.drop(columns=["Home_division"])

y = df["Depression"].map({"No": 0, "Yes": 1})
X = df.drop(columns=["Depression"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

text_columns = X.select_dtypes(exclude="number").columns.tolist()

encoder = ColumnTransformer(
    transformers=[("text_to_numbers", OneHotEncoder(handle_unknown="ignore"), text_columns)],
    remainder="passthrough",
)

model = Pipeline(steps=[
    ("encode", encoder),
    ("tree_model", RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions, target_names=["No", "Yes"]))