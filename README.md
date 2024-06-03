# FastGL: An Integrative <u>F</u>ramework for <u>A</u>ccelerating <u>S</u>ampling-Based <u>T</u>raining of <u>G</u>NN at <u>L</u>arge-Scale on GPUs


Dear readers, we have provided the source code to reproduce the results of our paper. The framework of FastGL is developed upon Pytorch. To implement the **Memory-Aware** and **Fused-Map**
techniques, we build the dedicated CUDA operators in the folder of `cuda_operators/`. Then we 
construct the Fused-Map sampler in `FusedMapSampler.py`. We implement our **Match-Reorder** 
strategy in `MatchReorderServer.py`. `run_fig*/` provides the training scripts of our experiments.

More information is detailed in **AE appendix**.

