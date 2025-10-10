I tried a_values = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 2, 5]


GeLU:
There was a lot less overfitting this time around with the smaller a values.
a = 1 had the highest accuracy while a = 2 had the lowest test loss

SiLU:
Again, a lot less overfitting with the smaller a values between 0 and 5
a = 2 had the highest accuracy and the lowest test loss

This is indicative that the SiLU activation function is better suited for this task than GeLU

Relu is 100
and gelu is 1.0
these results show that gelu performs better than relu over time and doesn't overfit as much


