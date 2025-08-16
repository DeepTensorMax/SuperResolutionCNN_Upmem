#include "common.h"
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>

__mram_noinit int32_t params[NB_PARAMETERS];
__mram_noinit int input[ROUND_UP_TO_MULTIPLE_OF_8(INPUT_MAX / NB_DPUS)];
__mram_noinit int weights[ROUND_UP_TO_MULTIPLE_OF_8(WEIGHTS_MAX)];
__mram_noinit float bias[ROUND_UP_TO_MULTIPLE_OF_8(BIASES_MAX)];
__mram float output[ROUND_UP_TO_MULTIPLE_OF_8(OUTPUT_MAX / NB_DPUS)];

BARRIER_INIT(common_wram_init_barrier,NR_TASKLETS);

__dma_aligned float bias_wram[BIASES_MAX]; // max 64 * 4 = 256 bytes
__dma_aligned int weight_kernel_row_wram[NR_TASKLETS * ROUND_UP_TO_MULTIPLE_OF_8(KERNEL_SIZE_MAX)]; // max NR_TASKLETS (16) * 9 * 4 = 576 bytes
__dma_aligned int image_kernel_row_wram[NR_TASKLETS * ROUND_UP_TO_MULTIPLE_OF_8(KERNEL_SIZE_MAX)];  // max NR_TASKLETS (16) * 9 * 4 = 576 bytes

// uint32_t checksums[NR_TASKLETS] = {0};
// #define CACHE_SIZE 32
// __dma_aligned uint32_t cache[NR_TASKLETS][CACHE_SIZE];
// __host uint32_t checksum;


int main() {
    // improvement: floating point
    // parallelism
    // working memory
    // reduce number of multiplications in inner-loops 
    // integer multiplication
    // noinit

    int32_t input_width = params[0];
    int32_t input_height = params[1];
    int32_t in_channels = params[2];
    int32_t out_channels = params[3];
    int32_t kernel_size = params[4];
    int32_t perform_relu = params[5];
    if (!me()) {
        // print all parameters
        printf("input_width: %d\n", input_width);
        printf("input_height: %d\n", input_height);
        printf("in_channels: %d\n", in_channels);
        printf("out_channels: %d\n", out_channels);
        printf("kernel_size: %d\n", kernel_size);
        printf("perform_relu: %d\n", perform_relu);
        printf("nr_tasklets: %d\n", NR_TASKLETS);

        // initialize common wram cache
        mram_read(bias,bias_wram,sizeof(float)*BIASES_MAX);
    }

    barrier_wait(&common_wram_init_barrier); // Wait until the main thread has initialized common wram cache

    int output_width = input_width - kernel_size + 1;
    int output_height = input_height - kernel_size + 1;
    int input_size_2D = input_width * input_height;
    int kernel_size_2D = kernel_size * kernel_size;
    int tasklet_offset = me() * ROUND_UP_TO_MULTIPLE_OF_8(KERNEL_SIZE_MAX);
    int mram_row_copy_length = sizeof(int)*ROUND_UP_TO_MULTIPLE_OF_8(kernel_size);

    for (int kernel = 0; kernel < out_channels; kernel++) {
        if (kernel % NR_TASKLETS != me()) {
            continue;
        }
        int kernel_offset_weight = kernel * in_channels * kernel_size_2D;
        int kernel_offset_output = kernel * output_width * output_height;
        for (int y = 0; y < output_height; y++) {
            int row_offset_output = y * output_width;
            for (int x = 0; x < output_width; x++) {
                int sum = (int)(bias_wram[kernel] * SCALING_FACTOR);
                for (int channel = 0; channel < in_channels; channel++) {
                    int channel_offset_image = channel * input_size_2D;
                    int channel_offset_weight = channel * kernel_size_2D;
                    for (int ky = 0; ky < kernel_size; ky++) {
                        // read input and weight rows into cache
                        int input_row_start_idx = channel_offset_image + (y+ky) * input_width + x;
                        int input_alignment_uneven_offset = input_row_start_idx % 2; // reqired to correct alignment to 8 bytes from mram copy in case of uneven idx
                        mram_read(&input[input_row_start_idx] - input_alignment_uneven_offset,&image_kernel_row_wram[tasklet_offset]- input_alignment_uneven_offset,mram_row_copy_length);
                        int weight_row_start_idx = kernel_offset_weight + channel_offset_weight + ky * kernel_size; // ((kernel * in_channels + channel) * kernel_size + ky) * kernel_size;
                        int weight_alignment_uneven_offset = weight_row_start_idx % 2; 
                        mram_read(&weights[weight_row_start_idx] - weight_alignment_uneven_offset,&weight_kernel_row_wram[tasklet_offset]- weight_alignment_uneven_offset,mram_row_copy_length);

                        // compute scalar product on cached rows
                        for (int kx = 0; kx < kernel_size; kx++) {
                            int input_pixel = image_kernel_row_wram[tasklet_offset + kx + input_alignment_uneven_offset];
                            int weight_pixel = weight_kernel_row_wram[tasklet_offset + kx + weight_alignment_uneven_offset];
                            sum +=  input_pixel * weight_pixel;
                        }
                    }
                }
                if (perform_relu) {  // perform relu if desired (for all but last layer)
                    sum = sum > 0 ? sum : 0;
                }
                output[kernel_offset_output + row_offset_output + x] = ((float)sum) / SCALING_FACTOR / SCALING_FACTOR;
            }
        }
    }
    return 0;
}