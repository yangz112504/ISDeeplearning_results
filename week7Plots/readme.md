Run Vgg16 without zailu approximiation this time and also time both of them to see how long each epoch takes:

Vgg16 without zailu approx: Total training time: 53402.76 seconds
Vgg16 WITH zailu approx: Total training time: 57103.58 seconds
Very strange...because zailu approximation uses arctanx which should take computationally longer but I guess it didn't...

Even though zailu approx took longer, there was less test loss and more accuracy overall

Benchmarked a wide range of PyTorch activation functions — both built-in and custom — across available devices (CPU, CUDA GPU, and Apple MPS).
It measures execution speed over thousands of iterations and outputs both raw timing data and a formatted research-style table for easy comparison.

We test each activation over a fixed input tensor ([-10, 10] range).
Running each function repeatedly (trials = 10,000) to average out runtime noise.