this week I tried different A values of [0, 0.25, 0.5, 1, 2, 100, 1000]
We plotted 1000 because it would be a placeholder for the actual ReLU function so we can compare side by side how other param values perform in relation to ReLU

We also added a new activation function, Zailu: x * (2 * (1/4 + 1/(2 * torch.pi) * torch.arctan(s * x)))
and we compared this performance to gelu_a and silu_a

During the train loss for both Gelu, Silu, and Zailu, the curves showed a steady decrease of loss
0.5 had the best accuracy and the least overfitting in the test loss graph for all 3 graphs

0.25 had the lowest test loss

Zailu and Gelu were similar in that most param lines started overfitting around 30 epochs while Silu started overfitting a lot sooner at around 20 epochs

for zailu the values we tried were sigma values of s_values = [0, 0.5, 1, 2, 5, 100]
Similarly, 0.5 worked the best and did the least amount of overfitting