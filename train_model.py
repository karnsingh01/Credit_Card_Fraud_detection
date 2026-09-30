import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. Load Dataset
# ==========================================

print("Loading dataset...")

df = pd.read_csv("credit_card_transactions.csv")

print("Dataset loaded!")
print("Shape:", df.shape)


# ==========================================
# 2. Separate Features and Target
# ==========================================

X = df.drop([
    "is_fraud",
    "Unnamed: 0",
    "trans_date_trans_time",
    "first",
    "last",
    "street",
    "dob",
    "trans_num"
], axis=1)

y = df["is_fraud"]


# ==========================================
# 3. Identify Columns
# ==========================================

categorical_cols = X.select_dtypes(
    include=["object", "string"]
).columns

numerical_cols = X.select_dtypes(
    exclude=["object", "string"]
).columns

print("\nCategorical columns:")
print(list(categorical_cols))

print("\nNumerical columns:")
print(list(numerical_cols))


# ==========================================
# 4. Preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "cat",

            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),

                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),

            categorical_cols
        )
    ],

    remainder=Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])
)


# ==========================================
# 5. Random Forest Model
# ==========================================

classifier = RandomForestClassifier(
    n_estimators=20,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# ==========================================
# 6. Complete Pipeline
# ==========================================

model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        classifier
    )
])


# ==========================================
# 7. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 8. Train Model
# ==========================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Model trained successfully!")


# ==========================================
# 9. Predictions
# ==========================================

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)


# ==========================================
# 10. Accuracy
# ==========================================

train_accuracy = accuracy_score(
    y_train,
    train_pred
)

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

print("\nTraining Accuracy:")
print(train_accuracy)

print("\nTesting Accuracy:")
print(test_accuracy)


# ==========================================
# 11. Classification Report
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_pred
    )
)


# ==========================================
# 12. Save Model
# ==========================================

print("\nSaving model...")

joblib.dump(
    model,
    "model.pkl",
    compress=3
)

print("Model saved successfully as model.pkl")