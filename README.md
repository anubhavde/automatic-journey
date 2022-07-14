# automatic-journey

## Road Sign Classification
As research in the field of Self-Driving cars continues, one of the most important challenges faced by scientists and engineers is Computer Vision. Computer Vision allows these cars to develop an understanding of their environment by analyzing digital images. Therefore what we are going to be dealing with is the ability to recognize and distinguish between differetn road signs like for example the STOP signal, SPEED LIMIT signs, yield signs, etc.

So, in this project, I have used Tensorflow to build a neural network to classify road signs based on images of those signs. For this we have used a labeled dataset: a collection of images that have already been categorized by the road signs represented by them.
The labeled dataset that we have used in this project is the German Traffic Sign Recognition Benchmark (GTSRB). The GTSRB is a dataset of over 21,000 images of traffic signs, each with a unique label. The benchmark was created by the German Traffic Sign Recognition Society (VOC) and the German Traffic Sign Recognition Benchmark (GTSRB) is a subset of the VOC dataset.

Moving towards the end of the project, you can see that we have achieved an efficiency of around <b>97.6%</b> which might vary each time you train your model.

You can acces the dataset from https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip, **NOTE:** Dataset is will be downloaded in a folder called gtsrb.zip. You have to unzip the folder and move the folder "gtsrb" to the directory where you have traffic.py file.

#### **OUTPUT**
Your output should be similar to that as shown below except a few things like GPU Information, etc.
```console
(rapids-22.02) ┌──(arch㉿hpenvylaptop15ep0xxx)-[~/Workspace/road_sign_classification]
└─$ <git:(master*)> python traffic.py gtsrb
2022-03-19 19:51:15.651803: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
2022-03-19 19:51:18.662275: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcuda.so.1
2022-03-19 19:51:20.298455: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-03-19 19:51:20.298911: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties:
pciBusID: 0000:01:00.0 name: NVIDIA GeForce GTX 1660 Ti with Max-Q Design computeCapability: 7.5
coreClock: 1.335GHz coreCount: 24 deviceMemorySize: 5.80GiB deviceMemoryBandwidth: 268.26GiB/s
2022-03-19 19:51:20.298975: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
2022-03-19 19:51:20.315256: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcublas.so.11
2022-03-19 19:51:20.315368: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcublasLt.so.11
2022-03-19 19:51:20.355320: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcufft.so.10
2022-03-19 19:51:20.357977: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcurand.so.10
2022-03-19 19:51:20.362083: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcusolver.so.11
2022-03-19 19:51:20.377558: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcusparse.so.11
2022-03-19 19:51:20.378195: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudnn.so.8
2022-03-19 19:51:20.378333: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-03-19 19:51:20.378562: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-03-19 19:51:20.378998: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1871] Adding visible gpu devices: 0
2022-03-19 19:51:20.379485: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-03-19 19:51:20.380084: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-03-19 19:51:20.380240: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties:
pciBusID: 0000:01:00.0 name: NVIDIA GeForce GTX 1660 Ti with Max-Q Design computeCapability: 7.5
coreClock: 1.335GHz coreCount: 24 deviceMemorySize: 5.80GiB deviceMemoryBandwidth: 268.26GiB/s
2022-03-19 19:51:20.380318: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-03-19 19:51:20.380493: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-03-19 19:51:20.380621: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1871] Adding visible gpu devices: 0
2022-03-19 19:51:20.380976: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
2022-03-19 19:51:21.125564: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1258] Device interconnect StreamExecutor with strength 1 edge matrix:
2022-03-19 19:51:21.125599: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1264]      0 
2022-03-19 19:51:21.125607: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1277] 0:   N 
2022-03-19 19:51:21.125780: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-03-19 19:51:21.126071: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-03-19 19:51:21.126337: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:937] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2022-03-19 19:51:21.126545: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1418] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 4665 MB memory) -> physical GPU (device: 0, name: NVIDIA GeForce GTX 1660 Ti with Max-Q Design, pci bus id: 0000:01:00.0, compute capability: 7.5)
2022-03-19 19:51:21.654618: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
2022-03-19 19:51:21.669736: I tensorflow/core/platform/profile_utils/cpu_utils.cc:114] CPU Frequency: 2601325000 Hz
Epoch 1/10
2022-03-19 19:51:22.248594: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudnn.so.8
2022-03-19 19:51:23.457654: I tensorflow/stream_executor/cuda/cuda_dnn.cc:359] Loaded cuDNN version 8302
2022-03-19 19:51:24.569827: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcublas.so.11
2022-03-19 19:51:24.570248: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcublasLt.so.11
500/500 [==============================] - 6s 4ms/step - loss: 4.0957 - accuracy: 0.0795
Epoch 2/10
500/500 [==============================] - 2s 4ms/step - loss: 3.0270 - accuracy: 0.2057
Epoch 3/10
500/500 [==============================] - 2s 4ms/step - loss: 2.1295 - accuracy: 0.3787
Epoch 4/10
500/500 [==============================] - 2s 4ms/step - loss: 1.5573 - accuracy: 0.5120
Epoch 5/10
500/500 [==============================] - 2s 4ms/step - loss: 1.1172 - accuracy: 0.6474
Epoch 6/10
500/500 [==============================] - 2s 4ms/step - loss: 0.7844 - accuracy: 0.7513
Epoch 7/10
500/500 [==============================] - 2s 4ms/step - loss: 0.5978 - accuracy: 0.8154
Epoch 8/10
<p>500/500 [==============================] - 2s 4ms/step - loss: 0.4436 - accuracy: 0.8623
Epoch 9/10
500/500 [==============================] - 2s 4ms/step - loss: 0.3667 - accuracy: 0.8874
Epoch 10/10
500/500 [==============================] - 2s 4ms/step - loss: 0.3200 - accuracy: 0.9046
333/333 - 1s - loss: 0.0963 - accuracy: 0.9768
```
