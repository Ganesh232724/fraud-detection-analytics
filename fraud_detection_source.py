"""
====================================================================
 FRAUD DETECTION ANALYTICS SYSTEM
 Complete Source Code
====================================================================
 Description : End-to-end fraud detection pipeline covering data
               loading, cleaning, EDA, feature engineering, model
               training (supervised + unsupervised), evaluation,
               risk scoring, and real-time-style alert generation.
 Language    : Python 3
 Libraries   : pandas, numpy, scikit-learn, matplotlib, seaborn
====================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for scripts/servers)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# --------------------------------------------------------------
# 1. DATA LOADING
# --------------------------------------------------------------
def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load transaction data from a CSV file.
    If no filepath is provided, a synthetic sample dataset is
    generated so the pipeline can run end-to-end for demo/testing.

    Parameters
    ----------
    filepath : str, optional
        Path to the transactions CSV file.

    Returns
    -------
    pd.DataFrame
        Raw transaction data.
    """
    if filepath:
        df = pd.read_csv(filepath)
        return df

    # ---- Synthetic dataset generator (for demo purposes only) ----
    n = 5000
    fraud_ratio = 0.02
    n_fraud = int(n * fraud_ratio)
    n_normal = n - n_fraud

    def generate_records(count, is_fraud):
        return pd.DataFrame({
            "transaction_id": np.arange(count),
            "amount": np.round(
                np.random.exponential(scale=250 if is_fraud else 80, size=count), 2
            ),
            "hour_of_day": np.random.randint(0, 24, size=count),
            "account_age_days": np.random.randint(1, 3000, size=count),
            "distance_from_home_km": np.round(
                np.random.exponential(scale=180 if is_fraud else 15, size=count), 2
            ),
            "transactions_last_24h": np.random.poisson(
                lam=6 if is_fraud else 1.5, size=count
            ),
            "payment_method": np.random.choice(
                ["credit_card", "debit_card", "net_banking", "wallet"], size=count
            ),
            "device_type": np.random.choice(
                ["mobile", "desktop", "pos_terminal"], size=count
            ),
            "is_fraud": is_fraud
        })

    df = pd.concat([
        generate_records(n_normal, 0),
        generate_records(n_fraud, 1)
    ], ignore_index=True)

    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


# --------------------------------------------------------------
# 2. DATA CLEANING
# --------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw transaction data:
    - Drop duplicate transactions
    - Handle missing values
    - Remove invalid/negative amounts

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction data.

    Returns
    -------
    pd.DataFrame
        Cleaned transaction data.
    """
    df = df.drop_duplicates(subset="transaction_id").copy()

    # Fill missing numeric values with median, categorical with mode
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Remove invalid transactions
    df = df[df["amount"] > 0].reset_index(drop=True)

    return df


# --------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# --------------------------------------------------------------
def run_eda(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Generate summary statistics and visualizations to understand
    transaction patterns and fraud distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction data.
    save_path : str, optional
        If provided, saves the EDA figure to this path instead of
        displaying it (useful for headless/server environments).
    """
    print("Dataset shape:", df.shape)
    print("\nFraud distribution:\n", df["is_fraud"].value_counts(normalize=True))
    print("\nSummary statistics:\n", df.describe())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    sns.countplot(x="is_fraud", data=df, ax=axes[0, 0])
    axes[0, 0].set_title("Fraud vs Legitimate Transaction Count")

    sns.boxplot(x="is_fraud", y="amount", data=df, ax=axes[0, 1])
    axes[0, 1].set_title("Transaction Amount by Fraud Label")

    sns.histplot(data=df, x="hour_of_day", hue="is_fraud", bins=24,
                 multiple="stack", ax=axes[1, 0])
    axes[1, 0].set_title("Transaction Hour Distribution")

    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=axes[1, 1])
    axes[1, 1].set_title("Feature Correlation Heatmap")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# --------------------------------------------------------------
# 4. FEATURE ENGINEERING
# --------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that improve fraud signal strength.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction data.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered features.
    """
    df = df.copy()

    # Flag unusually large transactions (relative to overall mean)
    df["high_amount_flag"] = (df["amount"] > df["amount"].mean() +
                               2 * df["amount"].std()).astype(int)

    # Flag transactions made late at night (common fraud window)
    df["night_transaction_flag"] = df["hour_of_day"].apply(
        lambda h: 1 if (h < 5 or h > 22) else 0
    )

    # Flag high-velocity accounts (many transactions in 24h)
    df["high_velocity_flag"] = (df["transactions_last_24h"] > 5).astype(int)

    # Flag new accounts (higher fraud risk)
    df["new_account_flag"] = (df["account_age_days"] < 30).astype(int)

    # Encode categorical variables
    for col in ["payment_method", "device_type"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])

    return df


# --------------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# --------------------------------------------------------------
def split_data(df: pd.DataFrame, feature_cols: list, target_col: str = "is_fraud"):
    """
    Split the dataset into training and testing sets using
    stratified sampling to preserve the fraud ratio, then scale
    numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataset.
    feature_cols : list
        List of feature column names to use for modeling.
    target_col : str
        Name of the target/label column.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, scaler
    """
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# --------------------------------------------------------------
# 6. MODEL TRAINING (SUPERVISED)
# --------------------------------------------------------------
def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Train a Logistic Regression baseline classifier."""
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------------------
# 7. MODEL TRAINING (UNSUPERVISED ANOMALY DETECTION)
# --------------------------------------------------------------
def train_isolation_forest(X_train, contamination: float = 0.02) -> IsolationForest:
    """
    Train an Isolation Forest to detect anomalous transactions
    without relying on fraud labels.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    contamination : float
        Expected proportion of anomalies in the data.

    Returns
    -------
    IsolationForest
        Trained anomaly detection model.
    """
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE
    )
    model.fit(X_train)
    return model


# --------------------------------------------------------------
# 8. MODEL EVALUATION
# --------------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Evaluate a trained classifier using standard fraud-detection
    metrics (precision, recall, F1, ROC-AUC are prioritized over
    plain accuracy because of class imbalance).

    Parameters
    ----------
    model : sklearn estimator
        Trained classification model.
    X_test, y_test : array-like
        Test features and true labels.
    model_name : str
        Label used when printing the report.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    print(f"\n--- {model_name} Evaluation ---")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    return metrics


# --------------------------------------------------------------
# 9. RISK SCORING (ENSEMBLE)
# --------------------------------------------------------------
def compute_risk_score(rf_model, iso_model, X) -> np.ndarray:
    """
    Combine the Random Forest fraud probability and the Isolation
    Forest anomaly score into a single 0-100 risk score per
    transaction.

    Parameters
    ----------
    rf_model : RandomForestClassifier
        Trained supervised model.
    iso_model : IsolationForest
        Trained anomaly detection model.
    X : array-like
        Feature matrix to score.

    Returns
    -------
    np.ndarray
        Array of risk scores (0-100) for each transaction.
    """
    rf_prob = rf_model.predict_proba(X)[:, 1]

    # Isolation Forest: more negative score_samples => more anomalous.
    # Normalize to a 0-1 "anomaly probability".
    raw_scores = iso_model.score_samples(X)
    anomaly_prob = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min())

    # Weighted ensemble: 70% supervised model, 30% anomaly detector
    combined = 0.7 * rf_prob + 0.3 * anomaly_prob
    risk_score = np.round(combined * 100, 2)
    return risk_score


# --------------------------------------------------------------
# 10. ALERT GENERATION
# --------------------------------------------------------------
def generate_alerts(df: pd.DataFrame, risk_scores: np.ndarray,
                     high_threshold: float = 75, review_threshold: float = 40) -> pd.DataFrame:
    """
    Classify transactions into decision buckets based on risk score
    and attach the results to the original transaction records.

    Parameters
    ----------
    df : pd.DataFrame
        Original transaction records (aligned with risk_scores).
    risk_scores : np.ndarray
        Computed risk scores for each transaction.
    high_threshold : float
        Score at or above which a transaction is blocked/flagged.
    review_threshold : float
        Score at or above which a transaction requires manual review.

    Returns
    -------
    pd.DataFrame
        Transactions with 'risk_score' and 'decision' columns.
    """
    result = df.copy()
    result["risk_score"] = risk_scores

    def decide(score):
        if score >= high_threshold:
            return "BLOCK / INVESTIGATE"
        elif score >= review_threshold:
            return "MANUAL REVIEW"
        else:
            return "APPROVE"

    result["decision"] = result["risk_score"].apply(decide)
    return result


# --------------------------------------------------------------
# 11. MAIN PIPELINE
# --------------------------------------------------------------
def main():
    """Run the complete fraud detection pipeline end-to-end."""

    # Step 1: Load data
    df_raw = load_data()

    # Step 2: Clean data
    df_clean = clean_data(df_raw)

    # Step 3: EDA (saved to file instead of shown interactively)
    run_eda(df_clean, save_path="eda_report.png")

    # Step 4: Feature engineering
    df_features = engineer_features(df_clean)

    feature_cols = [
        "amount", "hour_of_day", "account_age_days",
        "distance_from_home_km", "transactions_last_24h",
        "high_amount_flag", "night_transaction_flag",
        "high_velocity_flag", "new_account_flag",
        "payment_method_enc", "device_type_enc"
    ]

    # Step 5: Train/test split
    X_train, X_test, y_train, y_test, scaler = split_data(df_features, feature_cols)

    # Step 6: Train supervised models
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Step 7: Train unsupervised anomaly model
    iso_model = train_isolation_forest(X_train)

    # Step 8: Evaluate models
    evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # Step 9: Compute risk scores on the test set
    risk_scores = compute_risk_score(rf_model, iso_model, X_test)

    # Step 10: Generate alerts / decisions
    test_records = df_features.iloc[y_test.index].reset_index(drop=True)
    alert_report = generate_alerts(test_records, risk_scores)

    print("\nSample Alert Report:\n", alert_report[
        ["transaction_id", "amount", "risk_score", "decision"]
    ].head(10))

    alert_report.to_csv("fraud_alert_report.csv", index=False)
    print("\nPipeline complete. Alert report saved to 'fraud_alert_report.csv'.")


if __name__ == "__main__":
    main()
