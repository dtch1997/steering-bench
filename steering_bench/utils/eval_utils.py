import pandas as pd
from steering_bench.evaluate import EvalPrediction, EvalResult


# Functions to make pandas dataframes
def eval_prediction_as_dict(
    prediction: EvalPrediction,
):
    dict = {}
    dict.update(prediction.metrics)

    if prediction.positive_output_prob is not None:
        dict["test_example.positive.text"] = prediction.positive_output_prob.text
    else:
        dict["test_example.positive.text"] = None
    if prediction.negative_output_prob is not None:
        dict["test_example.negative.text"] = prediction.negative_output_prob.text
    else:
        dict["test_example.negative.text"] = None
    return dict


def eval_result_as_df(
    eval_result: EvalResult,
) -> pd.DataFrame:
    # predictions
    rows = []
    for idx, pred in enumerate(eval_result.predictions):
        dict = eval_prediction_as_dict(pred)
        dict["test_example.idx"] = idx
        rows.append(dict)
    # TODO: metrics?
    return pd.DataFrame(rows)


def eval_result_sweep_as_df(
    eval_results: dict[float, EvalResult],
) -> pd.DataFrame:
    dfs = []
    for multiplier, result in eval_results.items():
        df = eval_result_as_df(result)
        df["multiplier"] = multiplier
        dfs.append(df)
    return pd.concat(dfs)
