from typing import Dict, List, Union

import numpy as np
from absl import logging
from evaluate import load

metric_params = {
    "bleu": {"max_order": 2, "smooth": False},
    "bertscore": {"lang": "en"},
    "meteor": {},
    "rouge": {},
}


def generic_evaluate_function(
    metric_names: List[str], predictions: List[str], references: List[str]
) -> Dict[str, float]:
    def evaluate_metric(
        metric_name: str, predictions: List[str], references: List[str]
    ) -> float:
        metric = load(metric_name)
        try:
            results = metric.compute(
                predictions=predictions,
                references=references,
                **(metric_params.get(metric_name, {})),
            )
        except Exception as e:
            logging.error(f"Error while computing {metric_name}: {e}")
            return 0.0
        if metric_name in ["bleu"]:
            assert isinstance(results, dict)
            return results["bleu"]
        elif metric_name == "mauve":
            return getattr(results, "mauve")
        elif metric_name == "bertscore":
            assert isinstance(results, dict)
            return np.mean(results["f1"])
        elif metric_name == "rouge":
            assert isinstance(results, dict)
            return np.mean(results["rougeL"])
        elif metric_name == "meteor":
            assert isinstance(results, dict)
            return results["meteor"]
        assert isinstance(results, float)
        return results

    return {
        metric_name: evaluate_metric(metric_name, predictions, references)
        for metric_name in metric_names
    }
