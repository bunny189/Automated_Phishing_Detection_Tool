# models_src/predict.py
import os
import joblib
import numpy as np
from utils.feature_extraction import get_basic_url_features, url_tokenize_for_vector
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = os.path.dirname(os.path.dirname(__file__))
VECT_PATH = os.path.join(BASE, "models", "vectorizer.joblib")
SVM_PATH = os.path.join(BASE, "models", "svm_model.joblib")
RF_PATH  = os.path.join(BASE, "models", "rf_model.joblib")
XGB_PATH = os.path.join(BASE, "models", "xgb_model.joblib")

class EnsemblePredictor:
    def __init__(self):
        self.vect = joblib.load(VECT_PATH)
        self.svm = joblib.load(SVM_PATH)
        self.rf  = joblib.load(RF_PATH)
        self.xgb = joblib.load(XGB_PATH)

    def _make_features(self, urls):
        num = get_basic_url_features(urls)
        tokens = [url_tokenize_for_vector(u) for u in urls]
        tfidf = self.vect.transform(tokens)
        # join numeric and tfidf
        if hasattr(tfidf, "toarray"):
            tfarr = tfidf.toarray()
        else:
            tfarr = tfidf
        import numpy as np
        X = np.hstack([np.array(num), tfarr])
        return X

    def predict_single(self, url):
        X = self._make_features([url])
        p_svm = int(self.svm.predict(X)[0])
        p_rf  = int(self.rf.predict(X)[0])
        p_xgb = int(self.xgb.predict(X)[0])

        # Attempt to get prediction probabilities (if available)
        def safe_proba(m, X):
            try:
                p = m.predict_proba(X)[0][1]  # probability of phishing (label=1)
                return float(p)
            except Exception:
                return None

        prob_svm = safe_proba(self.svm, X)
        prob_rf  = safe_proba(self.rf, X)
        prob_xgb = safe_proba(self.xgb, X)

        votes = [p_svm, p_rf, p_xgb]
        phishing_votes = sum(votes)
        final = 1 if phishing_votes >= 2 else 0

        # confidence: average available prob or proportion of votes
        probs = [p for p in [prob_svm, prob_rf, prob_xgb] if p is not None]
        if probs:
            confidence = float(sum(probs) / len(probs))
        else:
            confidence = phishing_votes / 3.0

        return {
            "predictions": {"svm": p_svm, "rf": p_rf, "xgb": p_xgb},
            "probs": {"svm": prob_svm, "rf": prob_rf, "xgb": prob_xgb},
            "final": final,
            "confidence": confidence,
            "votes": phishing_votes
        }

# convenience function
_predictor = None
def ensemble_predict(url):
    global _predictor
    if _predictor is None:
        _predictor = EnsemblePredictor()
    return _predictor.predict_single(url)
