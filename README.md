# Feature Selection in the Presence of Monotone Batch Effect
Merging datasets from disparate environments comes with challenges, as datasets from the same environment may be subject to similar biases. For example, differences in genome sequencing machines, hybridization protocols, and transformation methods may lead to batch effects (i.e. systematic non-biological differences between batches of samples) in gene experiments. Batch effects can harm the performance of statistical inference algorithms (in particular, feature selection for detecting useful biomarkers) by imposing bias on the predictions, increasing the false discovery rate, and reducing prediction accuracy. 

Batch effect can be formalized as the changes in the distribution of datasets occuring by non-biological (non-causal) factors. Removing batch effect is an integral part of statistical analysis in biological tasks. The ideal goal is to recover the original distribution of datasets (batch-effect free) by matching the distributions of data batches.


# Methodology
