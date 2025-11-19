I ran noise experiments on deeper ResNet-18 architecture and evaluated multiple activation functions (ReLU, SiLU, GELU, ZaiLU, and ZaiLU-Approx)

I train each activation function cleanly, then test robustness by injecting Gaussian noise (σ = 0–8) at inference time

No training-time noise, no multiple trials, and no repeated seeds were used—the noise robustness curves are produced from one trained model per activation function

3 Trials and 100 epochs

The accuracy all started at 100% for 0 noise and then as noise was increased to 1, decreased to around 40%, and then hovered around 10% for the remaining noise levels

Loss steadily increasing, with gelu having the lowest loss capped at 14 at 8 noise sigma and silu having the highest at 40 at 8 noise sigma