import os
import json
from typing import Dict, Any

from nemoguardrails.evaluate.evaluate_factcheck import FactCheckEvaluation


def run_fact_check_evaluation(
    config_path: str = "config",
    dataset_path: str = "data/factchecking/sample.json",
    num_samples: int = 50,
    create_negatives: bool = True,
    output_dir: str = "eval_outputs/factchecking",
    write_outputs: bool = False,
) -> Dict[str, Any]:
    """
    Programmatic wrapper around the FactCheckEvaluation class to return metrics
    instead of just printing them.

    Returns:
        {
          "num_samples": int,
          "positive": {
             "correct": int, "total": int, "accuracy": float, "avg_time_ms": float
          },
          "negative": { ... }  (only if create_negatives=True),
          "overall_accuracy": float,
          "outputs_written": bool,
          "output_dir": str
        }
    """
    fact_eval = FactCheckEvaluation(
        config=config_path,
        dataset_path=dataset_path,
        num_samples=num_samples,
        create_negatives=create_negatives,
        output_dir=output_dir,
        write_outputs=write_outputs,
    )

    # Run positive split
    pos_predictions, pos_correct, pos_time = fact_eval.check_facts(split="positive")
    pos_total = len(pos_predictions)
    pos_accuracy = (pos_correct / pos_total) if pos_total else 0.0
    pos_avg_time_ms = (pos_time * 1000 / pos_total) if pos_total else 0.0

    neg_predictions = []
    neg_correct = 0
    neg_total = 0
    neg_accuracy = 0.0
    neg_avg_time_ms = 0.0

    if create_negatives:
        # create negatives inside the class instance
        fact_eval.create_negative_samples(fact_eval.dataset)
        neg_predictions, neg_correct, neg_time = fact_eval.check_facts(split="negative")
        neg_total = len(neg_predictions)
        neg_accuracy = (neg_correct / neg_total) if neg_total else 0.0
        neg_avg_time_ms = (neg_time * 1000 / neg_total) if neg_total else 0.0

    overall_total = pos_total + neg_total
    overall_correct = pos_correct + neg_correct
    overall_accuracy = (overall_correct / overall_total) if overall_total else 0.0

    # Optionally write outputs (mirroring FactCheckEvaluation.run() writing style)
    outputs_written = False
    if write_outputs:
        os.makedirs(output_dir, exist_ok=True)
        dataset_name = os.path.basename(dataset_path).split(".")[0]
        positive_path = f"{output_dir}/{dataset_name}_positive_fact_check_predictions.json"
        with open(positive_path, "w") as f:
            json.dump([p.__dict__ for p in pos_predictions], f, indent=2)
        if create_negatives:
            negative_path = f"{output_dir}/{dataset_name}_negative_fact_check_predictions.json"
            with open(negative_path, "w") as f:
                json.dump([p.__dict__ for p in neg_predictions], f, indent=2)
        outputs_written = True

    return {
        "num_samples_requested": num_samples,
        "positive": {
            "correct": pos_correct,
            "total": pos_total,
            "accuracy": round(pos_accuracy, 4),
            "avg_time_ms": round(pos_avg_time_ms, 2),
        },
        "negative": {
            "correct": neg_correct,
            "total": neg_total,
            "accuracy": round(neg_accuracy, 4),
            "avg_time_ms": round(neg_avg_time_ms, 2),
        } if create_negatives else None,
        "overall_accuracy": round(overall_accuracy, 4),
        "outputs_written": outputs_written,
        "output_dir": output_dir if outputs_written else None,
    }