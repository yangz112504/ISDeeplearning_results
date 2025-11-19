Compared to Cifar10, using VGG11 on Cifar100 wasn't as accurate. Accurracies were about 30-40% lower.

Param 5.0 was significantly more accurate than other params for gelu

Relu actually performed very well for Silu and Zailu, better than the other params

When there are more classes, Relu performed well (less data per class, you have same amount of data but in 100 classes). In Cifar 10 we had 5000 points per class / 10 classes, but in Cifar 100 we have 500 points in 100 classes

Also used Vgg11 for the first time