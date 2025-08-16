#include "common.h"
#include "image_manipulation.h"
#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#ifndef DPU_BINARY
#define DPU_BINARY "./out/srcnn"
#endif
#define NB_ELEMENTS 1024
#define NB_ELEMENTS_PER_DPU (NB_ELEMENTS/NB_DPUS)




// compile with gcc -o cnnexe srcnn.c -lm -O3


//------------------------------------ Model Functions --------------------------------------------

// Function to initialize a convolutional layer
void init_conv_layer(ConvLayer* layer, int in_channels, int out_channels, int kernel_size, int padding) {
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size = kernel_size;
    layer->padding = padding;
    
    int weights_size = out_channels * in_channels * kernel_size * kernel_size;
    layer->weights = (float*)malloc(weights_size * sizeof(float));
    
    layer->bias = (float*)malloc(out_channels * sizeof(float));
}

// Function to initialize the SRCNN model
SRCNN* init_srcnn() {
    SRCNN* model = (SRCNN*)malloc(sizeof(SRCNN));
    
    init_conv_layer(&model->conv1, CONV1_IN_CHANNELS, CONV1_OUT_CHANNELS, CONV1_KERNEL_SIZE, 9 / 2);
    init_conv_layer(&model->conv2, CONV2_IN_CHANNELS, CONV2_OUT_CHANNELS, CONV2_KERNEL_SIZE, 5 / 2);
    init_conv_layer(&model->conv3, CONV3_IN_CHANNELS, CONV3_OUT_CHANNELS, CONV3_KERNEL_SIZE, 5 / 2);
    
    return model;
}

// Function to free the memory allocated for the SRCNN model
void free_srcnn(SRCNN* model) {
    free(model->conv1.weights);
    free(model->conv1.bias);
    free(model->conv2.weights);
    free(model->conv2.bias);
    free(model->conv3.weights);
    free(model->conv3.bias);
    free(model);
}

// Function to load binary data
void load_binary(const char* filename, float* buffer, int size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    
    size_t read_size = fread(buffer, sizeof(float), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading file: %s. Expected %d elements, read %zu\n", filename, size, read_size);
        exit(1);
    }
    
    fclose(file);
}

// Function to load weights for the SRCNN model
void load_srcnn_weights(SRCNN* model) {
    load_binary("../ressources/bin/layer1_weights_x2.bin", model->conv1.weights, model->conv1.in_channels * model->conv1.out_channels * model->conv1.kernel_size * model->conv1.kernel_size);
    load_binary("../ressources/bin/layer1_biases_x2.bin", model->conv1.bias, model->conv1.out_channels);
    
    load_binary("../ressources/bin/layer2_weights_x2.bin", model->conv2.weights, model->conv2.in_channels * model->conv2.out_channels * model->conv2.kernel_size * model->conv2.kernel_size);
    load_binary("../ressources/bin/layer2_biases_x2.bin", model->conv2.bias, model->conv2.out_channels);
    
    load_binary("../ressources/bin/layer3_weights_x2.bin", model->conv3.weights, model->conv3.in_channels * model->conv3.out_channels * model->conv3.kernel_size * model->conv3.kernel_size);
    load_binary("../ressources/bin/layer3_biases_x2.bin", model->conv3.bias, model->conv3.out_channels);
}

void copyPartial(void* dst, void* src, int idx, int tile_size_x, int tile_size_y, int width, int channel, bool smallToBig) {
    int tile_start_x = 0;// idx * tile_size_x;
    int tile_start_y = idx * tile_size_y;

    // Copy partial content of src to dst according to idx
    for (int y = 0; y < tile_size_y; y++) {
        for (int x = 0; x < tile_size_x; x++) {
            int large_idx = ((y + tile_start_y) * width + (x + tile_start_x)) * channel;
            int small_idx = (y * tile_size_x + x) * channel;
            for (int c = 0; c < channel; c++) {
                if (smallToBig) {
                    ((float*) dst)[large_idx + c] = ((float*) src)[small_idx + c];
                } else {
                    ((int*) dst)[small_idx + c] = ((int*) src)[large_idx + c];
                }
            }
        }
    }
}

void transferDataAndLaunchDPU (struct dpu_set_t set, ConvLayer *layer, int *weights, int *input, int input_size, int width, int height, 
        int weights_size, int bias_size, int output_size, int perform_relu) {
    // Transfer Data to DPU
    DPU_ASSERT (dpu_copy_to (set, "weights", 0, weights,ROUND_UP_TO_MULTIPLE_OF_8 (weights_size * sizeof (int))));
    DPU_ASSERT (dpu_copy_to (set, "bias", 0, layer->bias,ROUND_UP_TO_MULTIPLE_OF_8 (bias_size * sizeof (float))));
    DPU_ASSERT (dpu_copy_to (set, "input", 0, input, ROUND_UP_TO_MULTIPLE_OF_8(input_size * sizeof (int))));
    int32_t params[NB_PARAMETERS] = {width,
                                    height,
                                    layer->in_channels,
                                    layer->out_channels,
                                    layer->kernel_size,
                                    perform_relu};
    DPU_ASSERT (dpu_copy_to (set, "params", 0, params, NB_PARAMETERS * sizeof (int32_t)));

    // Run DPU
    DPU_ASSERT(dpu_launch(set, DPU_ASYNCHRONOUS)); // convolution with or without relu depending on `perform_relu`
} 

float* runLayerAsTilesOnDPU (struct dpu_set_t set, struct dpu_set_t dpu, int num_dpus, ConvLayer *layer, float *inputFloat, int input_size, int width, int height, int channel,
        int weights_size, int bias_size, int output_size, int out_width, int out_height, int perform_relu) {
    // convert to floating point
    int* input = (int*)malloc(input_size * sizeof(int));
    layer->weights;
    for(int i = 0; i < input_size; i+=1) {
        input[i] = (int)(inputFloat[i] * SCALING_FACTOR);
    }
    int* weights = (int*)malloc(weights_size * sizeof(int));
    for(int i = 0; i < weights_size; i+=1) {
        weights[i] = (int)(layer->weights[i] * SCALING_FACTOR);
    }
    
    int tile_size_x = width;
    int tile_size_y = height / num_dpus;
    int tile_overlap_x = layer->kernel_size;
    int tile_overlap_y = layer->kernel_size;

    // Launch all DPUs asynchronously on tiles of input data
    int dpu_idx = 0;
    DPU_FOREACH(set, dpu) {
        int tile_size = tile_size_x * tile_size_y * channel;
        int* input_tile = (int*)malloc(tile_size * sizeof(int));

        // Copy the tile data to the DPU input buffer
        copyPartial(input_tile, input, dpu_idx, tile_size_x, tile_size_y, width, channel, false);

        transferDataAndLaunchDPU(dpu, layer, weights, input_tile, tile_size, tile_size_x, tile_size_y, 
            weights_size, bias_size, 0, perform_relu);
        free(input_tile);
        dpu_idx++;
    }

    printf("Waiting for all DPUs of layer to finish\n");
    DPU_ASSERT(dpu_sync(set));

    // Collect output tiles from all DPUs and merge them into a single output buffer
    float* output = (float*)malloc(output_size * sizeof(float));
    tile_size_x = out_width;
    tile_size_y = out_height / num_dpus;
    dpu_idx = 0;
    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_log_read(dpu, stdout));

        int tile_size = output_size/num_dpus;
        float* output_tile = (float*)malloc( tile_size * sizeof(float));
        DPU_ASSERT(dpu_copy_from(dpu, "output", 0, output_tile, tile_size * sizeof(float)));

        // Copy the tiled output to the final joint output buffer
        copyPartial(output, output_tile, dpu_idx, tile_size_x, tile_size_y, out_width, bias_size, true);

        free(output_tile);
        dpu_idx++;
    } 

    return output;
} 

int setupDPUs(struct dpu_set_t *set) {
    DPU_ASSERT(dpu_alloc(NB_DPUS, NULL, set));
    uint32_t num_ranks,num_dpus;
    dpu_get_nr_ranks(*set,&num_ranks);
    dpu_get_nr_dpus(*set,&num_dpus);
    printf("number of ranks for set: %d\n",num_ranks);
    const int num_dpus_to_use = NB_DPUS;
    if (num_dpus < num_dpus_to_use) {
        fprintf(stderr, "Error: Not enough dpus available, need %d, have %d\n", num_dpus_to_use, num_dpus);
        exit(EXIT_FAILURE);
    }
    printf("number of dpus for set available: %d, specified: %d, using: %d\n",num_dpus, NB_DPUS, num_dpus_to_use);
    DPU_ASSERT(dpu_load(*set, DPU_BINARY, NULL));
    return num_dpus_to_use;
}

// Forward pass of SRCNN
float* srcnn_forward(SRCNN* model, float* input, int width, int height) {
    struct dpu_set_t set, dpu;

    // Setup DPU
    const int num_dpus_to_use = setupDPUs(&set);

    // Conv1
    float* conv1_output = runLayerAsTilesOnDPU(set, dpu, num_dpus_to_use, &model->conv1, input, INPUT_SIZE1, INPUT_WIDTH_SRCNN, INPUT_HEIGHT_SRCNN, CONV1_IN_CHANNELS, 
        CONV1_WEIGHTS_SIZE, CONV1_OUT_CHANNELS, CONV1_OUTPUT_SIZE, CONV1_OUT_WIDTH, CONV1_OUT_HEIGHT, PERFORM_RELU);
    
    // Conv2 
    float* conv2_output = runLayerAsTilesOnDPU(set, dpu, num_dpus_to_use, &model->conv2, conv1_output, INPUT_SIZE2, CONV1_OUT_WIDTH, CONV1_OUT_HEIGHT, CONV1_OUT_CHANNELS, 
        CONV2_WEIGHTS_SIZE, CONV2_OUT_CHANNELS, CONV2_OUTPUT_SIZE, CONV2_OUT_WIDTH, CONV2_OUT_HEIGHT, PERFORM_RELU);
    free(conv1_output);

    // Conv3
    float* srcnn_output = runLayerAsTilesOnDPU(set, dpu, num_dpus_to_use, &model->conv3, conv2_output, INPUT_SIZE3, CONV2_OUT_WIDTH, CONV2_OUT_HEIGHT, CONV2_OUT_CHANNELS, 
        CONV3_WEIGHTS_SIZE, CONV3_OUT_CHANNELS, CONV3_OUTPUT_SIZE, CONV3_OUT_WIDTH, CONV3_OUT_HEIGHT, DONT_PERFORM_RELU);
    free(conv2_output);

    dpu_free(set);

    return srcnn_output;
}




//----------------------------------main------------------------------------------

int main() {
    const int nb_imges = 1;
    char* inputImagePaths[] = {
        "zebra",
    };

    for (int i = 0; i < nb_imges; i++) {
        char inputImagePath[256] = "../ressources/";
        strcat(inputImagePath, inputImagePaths[i]);
        strcat(inputImagePath, ".png");
        char outputReconstructedRGBPath[256] = "./out/";
        strcat(outputReconstructedRGBPath, inputImagePaths[i]);
        strcat(outputReconstructedRGBPath, "_upscaled.png");

        int inputWidth, inputHeight, inputChannels, outputWidth, outputHeight;
        unsigned char* ycbcrImageData;
        
        float* srcnn_input = loadAndPreprocessImage(inputImagePath, &inputWidth, &inputHeight, &inputChannels, &outputWidth, &outputHeight, &ycbcrImageData);
        if (outputHeight != INPUT_HEIGHT_SRCNN || outputWidth != INPUT_WIDTH_SRCNN) {
            fprintf(stderr, "Input image dimensions do not match the expected SRCNN input dimensions: %dx%d, got %dx%d \n", INPUT_WIDTH_SRCNN, INPUT_HEIGHT_SRCNN, outputWidth, outputHeight);
            free(ycbcrImageData);
            free(srcnn_input);
            return 1;
        }

        // Initialize the SRCNN model with 1 input channel (for Y channel)
        SRCNN* model = init_srcnn();
        // Load weights for the SRCNN model
        load_srcnn_weights(model);

        // Perform SRCNN forward pass
        printf("Starting forward pass \n");
        clock_t start_time = clock();
        float* srcnn_output = srcnn_forward(model, srcnn_input, outputWidth, outputHeight);
        printf("Ending forward pass done \n");
        double elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        printf("Elapsed time for forward pass: %.2f seconds\n", elapsed_time);
        
        // Reconstruct and save the enhanced RGB image
        int srcnn_output_width = outputWidth - model->conv1.kernel_size - model->conv2.kernel_size - model->conv3.kernel_size + 3;
        int srcnn_output_height = outputHeight - model->conv1.kernel_size - model->conv2.kernel_size - model->conv3.kernel_size + 3;
        reconstructAndSaveImage(inputWidth, inputHeight, inputChannels, ycbcrImageData, srcnn_output, srcnn_output_width, srcnn_output_height, outputWidth, outputHeight, outputReconstructedRGBPath);

        // Free allocated memory
        free(ycbcrImageData);
        free(srcnn_input);
        free(srcnn_output);
        free_srcnn(model);
    }

    return 0;
}