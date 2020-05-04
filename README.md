# Histogram Equalization on CUDA

## Prerequisites

* `CMake >= 3.8.0`
* `CUDA Compute Capability >= 2.0.0`

## Compilation

Compilation is done through `CMake`.

You may want to change the line `-gencode arch=compute_60,code=sm_60` in
`CMakeLists.txt` with the value corresponding to you GPU.

```bash
mkdir build
cd build
cmake ..
make -j
```

Compilation was tested with :

* `nvcc >= 10.2.89`
* `g++ >= 7.5.0`
* `clang >= 7.0.1`

### Usage

```
Usage:
          equalizer [OPTIONS] INPUT_IMAGE OUTPUT_IMAGE

Adjust image's contrast by via the histogram equalization method.

Mandatory arguments: 

         INPUT_IMAGE
                 Path to the image the histogram equalization will be applied on.

         OUTPUT_IMAGE
                 Path to the output equalized imag.e

Optional arguments (must be before mandatory arguments): 

         -g
                 Execute the program on the GPU (default) use with '-c' to execute on both the GPU and the CPU.
         -c
                 Execute the program on the CPU use with '-g' to execute on both the GPU and the CPU.
         -v
                 Displays information and the time taken by each step of the execution.
         -h
                 Displays this help
```

E.G.: `./equalization -g -c -v ../images/monument_6016x4000.png output.png`

## Result 

Here is the time taken on a CPU and a GPU with different sizes of image.

|                                        |  1500x844  |  2550x1917  |  6016x4000  |
|----------------------------------------|------------|-------------|-------------|
|GTX 1070                                |131.63 ms   |143.176 ms   |248.033 ms   |
|Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz|69.163 ms   |251.922 ms   |1311.17 ms   |
