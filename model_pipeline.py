import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ── XGBoost (graceful fallback) ──────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠  xgboost not installed – skipping XGBoost model.")

# ─── Utility Tracking ────────────────────────────────────────────────────────
def find_best_threshold(y_true, y_prob):
    best_f1 = 0
    best_thresh = 0.5
    for t in np.arange(0.4, 0.7, 0.02):
        y_pred = (y_prob > t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh

# ─── Data loading & feature engineering ──────────────────────────────────────
def load_and_preprocess_data(data):
    """
    Loads and preprocesses the Patient No-Show dataset.
    'data' can be a file path (str) or a pandas DataFrame.
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    # Rename columns if they come from Supabase with different casing (optional but safe)
    # Standard Kaggle columns: PatientId, AppointmentID, Gender, ScheduledDay, AppointmentDay, Age, Neighbourhood, Scholarship, Hipertension, Diabetes, Alcoholism, Handcap, SMS_received, No-show
    
    # Remove invalid age values
    df = df[df['Age'] >= 0]

    # Convert date columns
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'], errors='coerce')
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], errors='coerce')

    # Feature engineering
    df['WaitingTime'] = (df['AppointmentDay'].dt.normalize() - df['ScheduledDay'].dt.normalize()).dt.days
    df['WaitingTime'] = df['WaitingTime'].apply(lambda x: max(x, 0))
    df['AppointmentDayOfWeek'] = df['AppointmentDay'].dt.dayofweek
    df['AppointmentMonth'] = df['AppointmentDay'].dt.month
    df['IsWeekend'] = df['AppointmentDayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    # Target: 'No' -> 0 (Show), 'Yes' -> 1 (No-Show)
    # Handle cases where Supabase might already have 0/1 or 'No'/'Yes'
    if 'No-show' in df.columns:
        if df['No-show'].dtype == object:
            df['Target'] = df['No-show'].map({'No': 0, 'Yes': 1})
        else:
            df['Target'] = df['No-show']
        df = df.drop(columns=['No-show'])
    elif 'Target' not in df.columns:
        # Fallback if target column name is different
        pass

    # Drop non-predictive columns
    cols_to_drop = ['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return df


# ─── Main training routine ───────────────────────────────────────────────────
def main(df=None):
    print("=" * 60)
    print("  Patient No-Show — Automated Training Pipeline")
    print("=" * 60)

    if df is None:
        # Default fallback to local CSV
        df = load_and_preprocess_data('./Dataset/noshowappointments-kagglev2-may-2016.csv')
    
    print(f"\nDataset shape after preprocessing: {df.shape}")
    
    class_counts = df['Target'].value_counts()
    print(f"Class distribution:\n{class_counts.to_string()}")
    
    # Calculate scale_pos_weight for imbalance handling: count(negative) / count(positive)
    # Class 0: Show, Class 1: No-Show
    neg_count = class_counts.get(0, 1)
    pos_count = class_counts.get(1, 1)
    spw = neg_count / pos_count
    print(f"Calculated scale_pos_weight: {spw:.2f}")

    X = df.drop(columns=['Target'])
    y = df['Target']

    categorical_features = ['Gender', 'Neighbourhood']
    numeric_features = [
        'Age', 'Scholarship', 'Hipertension', 'Diabetes',
        'Alcoholism', 'Handcap', 'SMS_received',
        'WaitingTime', 'AppointmentDayOfWeek', 'AppointmentMonth', 'IsWeekend'
    ]

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    ])

    # ── Correlation Filtering (Global) ──────────────────────────────────
    X_enc = preprocessor.fit_transform(X)
    cat_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    encoded_names = numeric_features + cat_names
    
    corr_df = pd.DataFrame(X_enc, columns=encoded_names)
    corr_df['Target'] = y.values
    corr_matrix = corr_df.corr()['Target'].abs()
    
    # Keep features with > 0.012 correlation
    highly_correlated = corr_matrix[corr_matrix > 0.012].index.tolist()
    if 'Target' in highly_correlated: highly_correlated.remove('Target')
    
    print(f"Features filtered by correlation > 0.012: {len(highly_correlated)} remaining.")
    
    X_filtered = corr_df[highly_correlated].values

    # ── SMOTE on training  Dataset ─────────────────────────────────────────────
    print("Applying SMOTE to training  dataset...")
    X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_filtered, y)

    # ── Train/Test Split on Augmented Data ──────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    # ── Define candidate models (Hyperparameter Maps) ──────────
    models = {
        'Random Forest': (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                'classifier__n_estimators': [200, 300],
                'classifier__max_depth': [10, 20],
                'classifier__min_samples_leaf': [1, 2],
                'classifier__class_weight': ['balanced', None]
            }
        )
    }

    if HAS_XGB:
        models['XGBoost'] = (
            XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1, use_label_encoder=False),
            {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [4, 6, 8],
                'classifier__scale_pos_weight': [spw, spw*0.8, 1.0],
                'classifier__learning_rate': [0.05, 0.1]
            }
        )

    # ── Train & evaluate each model ─────────────────────────
    all_results = {}
    best_f1 = -1
    best_name = None
    # ── Train & evaluate each model ─────────────────────────
    all_results = {}
    best_f1 = -1
    best_name = None
    best_pipeline = None

    for name, (clf, param_grid) in models.items():
        print(f"Training and Tuning {name} …")

        clf_pipeline = Pipeline(steps=[
            ('classifier', clf),
        ])

        from sklearn.model_selection import RandomizedSearchCV
        search = RandomizedSearchCV(
            clf_pipeline, param_distributions=param_grid, 
            n_iter=6, scoring='f1', cv=3, random_state=42, n_jobs=1
        )

        search.fit(X_train, y_train)
        best_clf = search.best_estimator_
        print(f"  Best params: {search.best_params_}")

        if hasattr(best_clf.named_steps['classifier'], "predict_proba"):
            y_prob = best_clf.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_prob)
            best_thresh = find_best_threshold(y_test, y_prob)
            y_pred = (y_prob > best_thresh).astype(int)
        else:
            y_pred = best_clf.predict(X_test)
            roc = 0.0

        y_train_pred = best_clf.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        all_results[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(roc, 4),
            'train_accuracy': round(train_acc, 4),
        }

        print(f"  ✓ Test Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_pipeline = best_clf

    print(f"\n★  Best model: {best_name} — F1-Score {best_f1:.4f}")

    # ── Extract Feature Importances (Dynamic) ──────────────────────────
    try:
        if hasattr(best_pipeline.named_steps['classifier'], 'feature_importances_'):
            importances = best_pipeline.named_steps['classifier'].feature_importances_
        elif hasattr(best_pipeline.named_steps['classifier'], 'coef_'):
            # For linear models, use absolute coefficients
            importances = np.abs(best_pipeline.named_steps['classifier'].coef_[0])
        else:
            importances = np.zeros(len(highly_correlated))
    except Exception as e:
        print(f"⚠  Could not extract feature importances: {e}")
        importances = np.zeros(len(highly_correlated))

    feat_df = (
        pd.DataFrame({'Feature': highly_correlated, 'Importance': importances})
        .sort_values('Importance', ascending=False)
        .head(15)
    )

    # ── Assemble final metrics payload ───────────────────────────────────
    metrics = {
        'best_model': best_name,
        'models': all_results,
        'feature_importances': feat_df.to_dict('records'),
    }

    os.makedirs('models', exist_ok=True)
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    joblib.dump({
        'model': best_pipeline,
        'preprocessor': preprocessor,
        'features': highly_correlated,
        'categorical_features': categorical_features,
        'numeric_features': numeric_features
    }, 'models/best_model_pipeline.joblib')
    print(f"\n✅  Saved metrics → models/metrics.json")
    print(f"✅  Saved pipeline → models/best_model_pipeline.joblib")


if __name__ == '__main__':
    main()
