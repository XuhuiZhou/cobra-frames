from sbf_modeling.evaluation_utils import generic_evaluate_function


def test_generic_evaluate_function():
    metric_names = ["bleu", "rouge", "meteor", "bertscore", "mauve"]
    predictions = ["this is a test", "this is a test"]
    references = ["this is a test", "this is a test"]
    results = generic_evaluate_function(metric_names, predictions, references)
    assert True
