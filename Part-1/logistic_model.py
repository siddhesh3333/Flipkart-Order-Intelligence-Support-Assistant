# Lowering the decision threshold from 0.50 to 0.44 increases the model's ability to identify returned orders. Recall improves from 57.88% to 75.82%, an increase of 17.95 percentage points, meaning the model successfully detects many more actual returns. This improvement comes at the cost of a small reduction in precision from 29.64% to 28.01%, meaning more orders are incorrectly flagged as potential returns. In a return-risk prediction system, this trade-off is often acceptable because missing a genuine high-risk return can be more costly than investigating a few additional false positives. Therefore, choosing the threshold of 0.44 prioritizes identifying more risky orders while accepting a modest increase in false alarms.
# Lowering the decision threshold from 0.50 to 0.44 increases the model's ability to identify returned orders. Recall improves from 57.88% to 75.82%, an increase of 17.95 percentage points, meaning the model successfully detects many more actual returns. This improvement comes at the cost of a small reduction in precision from 29.64% to 28.01%, meaning more orders are incorrectly flagged as potential returns. In a return-risk prediction system, this trade-off is often acceptable because missing a genuine high-risk return can be more costly than investigating a few additional false positives. Therefore, choosing the threshold of 0.44 prioritizes identifying more risky orders while accepting a modest increase in false alarms.

print("\nBest Threshold")


print(f"Threshold : {best_metrics['threshold']:.2f}")

print(f"Precision : {best_metrics['precision']:.4f}")

print(f"Recall    : {best_metrics['recall']:.4f}")

print(f"F1 Score  : {best_metrics['f1']:.4f}")