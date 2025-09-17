# models_src/train_models.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from utils.feature_extraction import get_basic_url_features, url_tokenize_for_vector

# Paths
BASE = os.path.dirname(os.path.dirname(__file__))  # project root
DATA_PATH = os.path.join(BASE, "data", "sample_training_data.csv")
VECT_PATH = os.path.join(BASE, "models", "vectorizer.joblib")
SVM_PATH = os.path.join(BASE, "models", "svm_model.joblib")
RF_PATH  = os.path.join(BASE, "models", "rf_model.joblib")
XGB_PATH = os.path.join(BASE, "models", "xgb_model.joblib")

def load_data(path):
    df = pd.read_csv(path)
    # Expect columns: url,label (1 phishing, 0 legit)
    df = df.dropna(subset=["url", "label"])
    df['label'] = df['label'].astype(int)
    return df

def build_features(df):
    urls = df['url'].astype(str).tolist()
    num_feats = get_basic_url_features(urls)
    text_tokens = [url_tokenize_for_vector(u) for u in urls]
    return num_feats, text_tokens

def join_features(num_feats, tfidf_matrix):
    # tfidf is sparse; convert to array or hstack
    import numpy as np
    if hasattr(tfidf_matrix, "toarray"):
        tfidf_arr = tfidf_matrix.toarray()
    else:
        tfidf_arr = tfidf_matrix
    return np.hstack([num_feats, tfidf_arr])

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    num_feats, text_tokens = build_features(df)

    print("Fitting vectorizer (TF-IDF on URL tokens)...")
    vect = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    tfidf = vect.fit_transform(text_tokens)
    joblib.dump(vect, VECT_PATH)
    print("Saved vectorizer to", VECT_PATH)

    import numpy as np
    X_num = np.array(num_feats)
    X = join_features(X_num, tfidf)
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training SVM (with probabilities)...")
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    svm.fit(X_train, y_train)
    joblib.dump(svm, SVM_PATH)
    print("Saved SVM model to", SVM_PATH)

    print("Training RandomForest...")
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, RF_PATH)
    print("Saved RF model to", RF_PATH)

    print("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, XGB_PATH)
    print("Saved XGBoost model to", XGB_PATH)

    print("Evaluating ensemble on test set...")
    preds_svm = svm.predict(X_test)
    preds_rf = rf.predict(X_test)
    preds_xgb = xgb.predict(X_test)
    # majority voting
    import numpy as np
    combined = np.vstack([preds_svm, preds_rf, preds_xgb]).T
    final = [1 if (row.sum() >= 2) else 0 for row in combined]

    print(classification_report(y_test, final))

if __name__ == "__main__":
    main()
