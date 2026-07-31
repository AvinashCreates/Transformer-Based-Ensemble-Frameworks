Transformer-Based Ensemble Framework with Topographic and Atmospheric Feature Fusion for Sub-Seasonal Prediction of Extreme Rainfall Events Over the Indian Subcontinent

Objective 1 — Build a Transformer-based feature fusion framework
Why?

Atmospheric variables and terrain influence rainfall differently.

Instead of concatenating features, use attention to learn interactions.

Have you achieved it?

YES.

You built:

CMAG
Cross-modal attention
Frozen shared representation

I consider this objective completed

Objective 2 — Integrate atmospheric and terrain information
Why?

Rainfall depends on

pressure
humidity
wind
elevation
slope
terrain barriers

The project isn't just predicting rainfall.

It is showing that both information sources can be fused.

Status

Implementation:

Completed

Scientific validation:

Depends on confirming the data source.

Objective 3 — Build a heterogeneous ensemble

This is actually your biggest novelty.

Not

Transformer only.

Not

LSTM only.

Instead

Transformer

BiLSTM

TCN

XGBoost

Stacking.

Status

Completed.

All branches exist.

All execute.

Outputs produced.

Objective 4 — Predict sub-seasonal rainfall

21

35

49 day lead times.

This directly satisfies the title.

Status:

Completed.

Objective 5 — Predict extreme rainfall

Regression alone isn't enough.

Need

Rainfall amount

Extreme event classification.

You implemented both heads.

Status:

Implementation complete.

Scientific evaluation still pending.

Objective 6 — Demonstrate that the architecture works

This is important.

Your project is not

"I invented rainfall."

It is

"I proposed a framework."

Meaning:

The framework should

execute
converge
generate outputs
support comparison

Status:

Completed.

Objective 7 — Compare the ensemble against individual branches

Notice:

This is NOT

"ensemble must win."

It is

"perform comparative analysis."

Your guide basically hinted this.

Status

Not finished.

Because comparison needs trustworthy experiments.

