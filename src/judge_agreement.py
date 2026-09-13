from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

class HumanJudgeAgreementAnalyzer:
    """
    Evaluates statistical agreement between LLM-as-Judge ratings and Human Gold Annotations.
    Computes:
    - Cohen's Kappa Score (Quadratic Weighted & Unweighted)
    - Exact % Agreement
    - Adjacent % Agreement (within +/- 1 point)
    - Pearson Correlation
    - Confusion Matrix
    """
    @staticmethod
    def evaluate_agreement(human_scores: List[int], judge_scores: List[int]) -> Dict[str, Any]:
        if not human_scores or len(human_scores) != len(judge_scores):
            raise ValueError("Human and Judge score lists must be non-empty and equal length.")

        y_human = np.array(human_scores, dtype=int)
        y_judge = np.array(judge_scores, dtype=int)

        # Exact Agreement
        exact_acc = float(accuracy_score(y_human, y_judge))

        # Adjacent Agreement (+/- 1 score point)
        diff = np.abs(y_human - y_judge)
        adj_acc = float(np.mean(diff <= 1))

        # Cohen's Kappa
        kappa_unweighted = float(cohen_kappa_score(y_human, y_judge, labels=[1, 2, 3, 4, 5]))
        try:
            kappa_weighted = float(cohen_kappa_score(y_human, y_judge, labels=[1, 2, 3, 4, 5], weights='quadratic'))
        except Exception:
            kappa_weighted = kappa_unweighted

        # Pearson Correlation
        if np.std(y_human) > 0 and np.std(y_judge) > 0:
            pearson_corr = float(np.corrcoef(y_human, y_judge)[0, 1])
        else:
            pearson_corr = 1.0

        # Confusion Matrix (labels 1 to 5)
        cm = confusion_matrix(y_human, y_judge, labels=[1, 2, 3, 4, 5]).tolist()

        return {
            "num_samples": len(y_human),
            "exact_agreement": round(exact_acc, 4),
            "adjacent_agreement_pm1": round(adj_acc, 4),
            "cohen_kappa_unweighted": round(kappa_unweighted, 4),
            "cohen_kappa_quadratic": round(kappa_weighted, 4),
            "pearson_correlation": round(pearson_corr, 4),
            "mean_human_score": round(float(np.mean(y_human)), 2),
            "mean_judge_score": round(float(np.mean(y_judge)), 2),
            "confusion_matrix": cm
        }
