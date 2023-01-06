from typing import Dict, List, Union

import numpy as np
from evaluate import load


def generic_evaluate_function(
    metric_names: List[str], predictions: List[str], references: List[str]
) -> Dict[str, Union[Dict[str, float], float]]:
    def evaluate_metric(
        metric_name: str, predictions: List[str], references: List[str]
    ) -> Union[Dict[str, float], float]:
        metric = load(metric_name)
        results = metric.compute(
            predictions=predictions,
            references=references,
            **({} if metric_name != "bertscore" else {"lang": "en"})
        )
        if metric_name in ["bleu"]:
            assert isinstance(results, dict)
            return results
        if metric_name == "mauve":
            return getattr(results, "mauve")
        if isinstance(results, dict):
            for key in results:
                if isinstance(results[key], list):
                    results[key] = np.mean(results[key])
            return results
        elif isinstance(results, list):
            return np.mean(results)
        else:
            assert isinstance(results, float)
            return results

    return {
        metric_name: evaluate_metric(metric_name, predictions, references)
        for metric_name in metric_names
    }
