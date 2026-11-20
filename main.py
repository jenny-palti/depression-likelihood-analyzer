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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


PHQ_ITEM_COLUMNS = [f"phq{i}" for i in range(1, 10)]

PHQ_NAMES = {
    1: "Little interest or pleasure in doing things",
    2: "Feeling down, depressed or hopeless",
    3: "Trouble with sleep",
    4: "Feeling tired or having little energy",
    5: "Poor appetite or overeating",
    6: "Feeling bad about yourself or like a failure",
    7: "Trouble concentrating",
    8: "Moving or speaking very slowly, or being restless",
    9: "Thoughts of being better off dead or of self-harm",
}

PROTOCOLS = {
    "A": {
        "name": "Immediate safety follow-up",
        "crisis": True,
        "text": """This student answered 2 or 3 on PHQ-9 item 9 (thoughts of being better off dead or of self-harm on more than half the days or nearly every day). No score is shown. A follow-up conversation is recommended.
1. Contact the student: same day for level 1, within 24-48 hours for level 2.
2. A trained counselor carries out a structured suicide risk assessment (for example the C-SSRS) and, where possible, consults a second trained colleague before deciding on the level of care.
3. If the student shows immediate danger (a plan, intent, or access to means): do not leave them alone, remove means if it is safe to do so, and call emergency services [local emergency number].
4. Contact the parent or guardian according to institutional policy, and advise securing medications and other dangerous items.
5. Complete a safety plan with the student: warning signs, coping strategies, supports, crisis line numbers.
6. Refer to community mental health services with 24-hour access if needed, and obtain consent to share information with the providers.
7. Document every contact and decision. Follow up within one week and again at one month.
The counselor decides. This tool does not diagnose.""",
    },
    "B": {
        "name": "Clinician review",
        "crisis": True,
        "text": """This student answered 1 on PHQ-9 item 9 (such thoughts on several days). No score is shown. A follow-up conversation is recommended.
1. A counselor reviews and contacts the student: within 3 days for level 3 (PHQ-9 total 15 or higher), within 4 days for level 4 (total 14 or lower). The order is a sorting aid, not a difference in risk.
2. In the conversation, ask directly about suicidal thoughts and carry out a structured screening (for example the C-SSRS screener).
3. If the screening suggests higher risk, move to Path A at once.
4. If not, agree a safety plan where needed, share resources and counseling options, and involve the parent or guardian according to policy.
5. Document the contact and the plan. Check in again within 2 weeks.
The counselor decides. This tool does not diagnose.""",
    },
    "C": {
        "name": "Routine support",
        "crisis": False,
        "text": """The student answered 0 on PHQ-9 item 9. This is not a crisis path, and a score may be shown.
1. Level 5 (PHQ-9 total 15 or higher): offer a check-in conversation and information about support services.
2. Level 6 (total 14 or lower): routine support. Share wellbeing resources. Students are listed by total, highest first.
3. An answer of 0 does not rule out risk. If a concern arises later from the student or anyone else, use Path A or B at any time.
The counselor decides. This tool does not diagnose.""",
    },
}

LEVEL_INFO = {
    1: {"protocol": "A", "label": "Same day"},
    2: {"protocol": "A", "label": "Within 24-48 hours"},
    3: {"protocol": "B", "label": "Priority review"},
    4: {"protocol": "B", "label": "Scheduled review"},
    5: {"protocol": "C", "label": "Check-in offered"},
    6: {"protocol": "C", "label": "Routine"},
}


def severity_band(total: int) -> str:
    if total <= 4:
        return "Minimal"
    if total <= 9:
        return "Mild"
    if total <= 14:
        return "Moderate"
    if total <= 19:
        return "Moderately severe"
    return "Severe"


def assign_level(item9: int, total: int) -> int:
    if item9 == 3:
        return 1
    if item9 == 2:
        return 2
    if item9 == 1 and total >= 15:
        return 3
    if item9 == 1:
        return 4
    if total >= 15:
        return 5
    return 6


def get_reasons(row) -> list:
    reasons = []
    if row["phq9"] >= 1:
        reasons.append(PHQ_NAMES[9])
    for i in range(1, 9):
        if row[f"phq{i}"] >= 2:
            reasons.append(PHQ_NAMES[i])
    return reasons


def add_phq_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["phq_total"] = data[PHQ_ITEM_COLUMNS].sum(axis=1)
    data["phq_band"] = data["phq_total"].apply(severity_band)
    data["phq10_plus"] = (data["phq_total"] >= 10).astype(int)
    data["level"] = [assign_level(a, t) for a, t in zip(data["phq9"], data["phq_total"])]
    data["reasons"] = data.apply(get_reasons, axis=1)
    return data


FEATURE_COLUMNS = (
    ["life_satisfaction"]
    + [f"{name}_yesterday" for name in ["happy", "laugh", "learn", "enjoy", "worried", "angry", "stress", "lonely"]]
    + [f"pss{i}" for i in range(1, 11)]
    + [f"gad{i}" for i in range(1, 8)]
)
GROUP_COLUMNS = ["gender", "ses", "ethnicity"]
CUTOFF = 0.35


def prepare_features_and_target(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df["phq10_plus"]
    groups = df[GROUP_COLUMNS]
    return X, y, groups


def split_data(X, y, groups):
    return train_test_split(X, y, groups, test_size=0.2, random_state=42, stratify=y)


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def evaluate_model(model, X_test, y_test, cutoff: float = CUTOFF):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= cutoff).astype(int)
    print(classification_report(y_test, predictions, target_names=["under 10", "10 or more"]))
    return predictions

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



def split_data(X, y, groups):
    return train_test_split(X, y, groups, test_size=0.2, random_state=42, stratify=y)


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model



def main():
    pass


if __name__ == "__main__":
    main()