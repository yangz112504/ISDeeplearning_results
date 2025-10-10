I tried a_values = [0, 0.5, 1, 2, 5, 100]
when a = 0.5, it was the most accurate and did not significantly overfit unlike other a values of 1,2,5,100

This is evident in the train loss graph where a values greater than 0.5 had significantly decreased loss but that
meant that the model was just overfitting - the model was TOO trained on the training dataset and not applicable enough
to the test data set as shown by the increase of the test loss graph lines where a > 0.5.

Meanwhile, a = 0 didn't really do anything, as there was no signficant loss in the training or test set

The same occurrences in the a values can be seen in SILU graphs

We want to try smaller A values this time: a = {0, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 2, 5}
