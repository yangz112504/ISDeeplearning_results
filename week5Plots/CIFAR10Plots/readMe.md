This week, we kept the code the same except for the architecture, which we used VGG11 for instead of a simple MLP.

With Vgg11, the average accuracy rose up about 20%, from 50% to 70%.

Param 5.0 seemed to perform well across gelu, silu, and zailu

GELU:
A lot of overfitting except from param 0 in the test loss, but in the train loss the loss's decreased steadily

SILU:
A lot of overfitting except from param 0,0.25,0.5 in the test loss, but in the train loss the loss's decreased steadily

Zailu:
A lot of overfitting except from param 0,0.25,0.5 in the test loss, but in the train loss the loss's decreased steadily

