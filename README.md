# Super-resolution Convolutional Neural Network on Upmem PIM DIMMs

Please find the report under `ACA_Conv_Project.pdf`.

Please note that the Upmem Simulator is extremely slow which is why you need to use extremely small images like `ressources/zebra_small.png` and even then the simulator will need couple of minutes to process the image. 

## Usage

Since for our development and all our testing we needed to use images consisting of only a few pixels there is a large area at the bottom of the image with is not computed due to kernel sizes. This wouldn't be noticable in larger image. Also, we could only really compare our implementation for correctness against the results of `srcnn_without_upmem.c` which is the CPU-only variant. The problem is, for small images it also produces garbage outputs, because it was trained on much larger images, but on these much larger images it works as intended.  
We ensured the correctness of our implementation by comparing the DPU outputs with the CPU output which can be found under `ressources/zebra_upscaled_reference.png`.

Here is how to setup the project:

```sh
# assuming you are in the base directory and `ls` shows src/
cd src
mkdir out 
mkdir out/intermediary

# compile host
gcc -g -o out/srcnn_host srcnn_host.c  -lm `dpu-pkg-config --cflags --libs dpu`
# compile DPU kernels (please adjust number tasklets if necessary)
dpu-upmem-dpurte-clang -DNR_TASKLETS=12 -o out/srcnn srcnn.c 
# run upscaling with DPU
./out/srcnn_host

# compile and run upscaling with CPU
gcc -o out/srcnn_cpu srcnn_without_upmem.c -lm
./out/srcnn_cpu
```

If you want to change the number of DPUs used, which is required to process larger images, you have to change the `NB_DPUS` macro in the `common.h` file and recompile both host and dpu kernel. Only use powers of 2 for the NB_DPUs.

If you want to test other pictures, you need to enter the path in the source files (`srcnn_host.c` or `srcnn_without_upmem.c`) and if you want to run it on DPU, you also need to enter the *upscaled* image size (image width/height x 2) in `common.h` under `INPUT_WIDTH_SRCNN` and `INPUT_HEIGHT_SRCNN`. Then recompile host and dpu kernel code. 

## Disclaimer 

Please note that our image *preprocessing* code exhibited very strange behavior on the testing machine with real Upmem hardware we were given access to. When you execute the software as described above and have a look into the `src/out/intermediary` directory, you may (or may not if you are lucky with your machine) see that the images get preprocessed very strangely (except the bicubic upscaling which works as intended). This applies to both the CPU-only (`srcnn_without_upmem.c`) and host + dpu code. We only noticed this issue very late into our 10 hour time frame on the test machine and were not able to locate the bug with the remaining time. We didn't want to ask for more time since we felt this would be unfair to the other groups and would not be granted. On all of our testing machines (Ubuntu and the compilers given in Pouria's ACA Upmem repository) the preprocessing code worked alright. Perhaps the installed compiler on the Debian testing machine works differently or it is due to the debian distro itself (although unlikely since unbuntu is debian-based). 

Since the final output depends on the stages generated during preprocessing, as the SRCNN output is recombined with the chroma channels, the final output also looks strange. It is still recognizable that it worked but not optimal. We hope this small bug which has nothing to do with the DPU code does not lead to large point deductions. 