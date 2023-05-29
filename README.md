## Feature Selection in the Presence of Monotone Batch Effect
Merging datasets from disparate environments comes with challenges, as datasets from the same environment may be subject to similar biases. For example, differences in genome sequencing machines, hybridization protocols, and transformation methods may lead to batch effects (i.e. systematic non-biological differences between batches of samples) in gene experiments. Batch effects can harm the performance of statistical inference algorithms (in particular, feature selection for detecting useful biomarkers) by imposing bias on the predictions, increasing the false discovery rate, and reducing prediction accuracy. 

Batch effect can be formalized as the changes in the distribution of datasets occuring by non-biological (non-causal) factors. Removing batch effect is an integral part of statistical analysis in biological tasks. The ideal goal is to recover the original distribution of datasets (batch-effect free) by matching the distributions of data batches.

![alt text](https://github.com/DesPeradoGoden/Feature-Selection-in-the-Presence-of-Monotone-Batch-Effects/blob/main/Batch%20Effect.png?raw=true)

## Methodology
We assume the predictor x<sup>*</sup> without batch effects follows a multi-variate Gaussian distribution N(μ, Σ) with the zero mean and an unknown covariance matrix. Therefore, we can formulate the batch effect removal problem as finding the optimal transformation and the covariance matrix Σ such that the total difference of transformed distributions and the underlying normal distribution is minimized. Each  transformation is modeled by a two-layers neural network. We add a negativity constraint on the weights of the neural networks to make sure the corresponding transformations are monotone. As mentioned earlier, such a formulation does not consider the bijective constraint on the transformations. To avoid spurious solutions, we unify the feature selection (Lasso regression) and the batch effect removal task into the following optimization problem:
![alt text](https://github.com/DesPeradoGoden/Feature-Selection-in-the-Presence-of-Monotone-Batch-Effects/blob/main/MMD.png?raw=true)

## Result on the Synthetic Data
F1 score measrure the harmonic mean of recall and precision. We can observe that minimizing the above objective function leads to higher F1 score.
![alt text](https://github.com/DesPeradoGoden/Feature-Selection-in-the-Presence-of-Monotone-Batch-Effects/blob/main/Figure_5.png?raw=true)

Further, as we see in the heatmaps, the MMD distance between batches are reduced to less than 0.1 after the transformation, while before that it is around between 1 and 2.
![alt text](https://github.com/DesPeradoGoden/Feature-Selection-in-the-Presence-of-Monotone-Batch-Effects/blob/main/Figure_2.png?raw=true)
![alt text](https://github.com/DesPeradoGoden/Feature-Selection-in-the-Presence-of-Monotone-Batch-Effects/blob/main/Figure_3.png?raw=true)
