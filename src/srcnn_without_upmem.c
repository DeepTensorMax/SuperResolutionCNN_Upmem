#include "common.h"
#include "image_manipulation.h"
#include <time.h>


// compile with gcc -o out/srcnn_cpu srcnn_without_upmem.c -lm -O3


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
SRCNN* init_srcnn(int num_channels) {
    SRCNN* model = (SRCNN*)malloc(sizeof(SRCNN));
    
    init_conv_layer(&model->conv1, num_channels, 64, 9, 9 / 2);
    init_conv_layer(&model->conv2, 64, 32, 5, 5 / 2);
    init_conv_layer(&model->conv3, 32, num_channels, 5, 5 / 2);
    
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

// Function to print first few elements of a buffer
void print_debug(const char* name, float* buffer, int size) {
    printf("Debug %s: ", name);
    for (int i = 0; i < (size < 5 ? size : 5); i++) {
        printf("%f ", buffer[i]);
    }
    printf("...\n");
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

float relu(float x) {
    return x > 0 ? x : 0;
}

void relu_layer(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = relu(input[i]);
    }
}

// convolve calculates the value of a single convolution operation where the kernel is centered at (x, y) in the input
float convolve(float* input, ConvLayer* layer, int kernel, int x, int y, int input_width, int input_height) {
    float sum = 0.0;
    for (int channel = 0; channel < layer->in_channels; channel++) {
        for (int ky = 0; ky < layer->kernel_size; ky++) {
            for (int kx = 0; kx < layer->kernel_size; kx++) {
                int input_idx = (y+ky) * input_width + (x+kx) + channel * input_width * input_height;
                int weight_idx = ((kernel * layer->in_channels + channel) * layer->kernel_size + ky) * layer->kernel_size + kx;
                sum += input[input_idx] * layer->weights[weight_idx];
            }
        }
    }
    return sum;
}

void convolution_layer(float* input, float* output, ConvLayer* layer, int input_width, int input_height) {
    int output_width = input_width - layer->kernel_size + 1;
    int output_height = input_height - layer->kernel_size + 1;

    for (int kernel = 0; kernel < layer->out_channels; kernel++) {
        for (int y = 0; y < output_height; y++) {
            for (int x = 0; x < output_width; x++) {
                float sum = layer->bias[kernel];
                sum += convolve(input, layer, kernel, x, y, input_width, input_height);
                output[kernel * output_width * output_height + y * output_width + x] = sum;
            }
        }
    }
}

// Forward pass of SRCNN
float* srcnn_forward(SRCNN* model, float* input, int width, int height) {
    // Conv1
    int conv1_out_width = width - model->conv1.kernel_size + 1;
    int conv1_out_height = height - model->conv1.kernel_size + 1;
    float* conv1_output = (float*)malloc(conv1_out_width * conv1_out_height * model->conv1.out_channels * sizeof(float));
    convolution_layer(input, conv1_output, &model->conv1, width, height);
    // Relu
    relu_layer(conv1_output, conv1_output, conv1_out_width * conv1_out_height * model->conv1.out_channels);

    // Conv2 
    int conv2_out_width = conv1_out_width - model->conv2.kernel_size + 1;
    int conv2_out_height = conv1_out_height - model->conv2.kernel_size + 1;
    float* conv2_output = (float*)malloc(conv2_out_width * conv2_out_height * model->conv2.out_channels * sizeof(float));
    convolution_layer(conv1_output, conv2_output, &model->conv2, conv1_out_width, conv1_out_height);
    // Relu
    relu_layer(conv2_output, conv2_output, conv2_out_width * conv2_out_height * model->conv2.out_channels);
    free(conv1_output);

    // Conv3
    int srcnn_output_width = width - model->conv1.kernel_size - model->conv2.kernel_size - model->conv3.kernel_size + 3;
    int srcnn_output_height = height - model->conv1.kernel_size - model->conv2.kernel_size - model->conv3.kernel_size + 3;
    float* srcnn_output = (float*)malloc(srcnn_output_width * srcnn_output_height * sizeof(float));
    convolution_layer(conv2_output, srcnn_output, &model->conv3, conv2_out_width, conv2_out_height);
    free(conv2_output);

    return srcnn_output;
}

//----------------------------------main -------------------------------------------

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

        // Initialize the SRCNN model with 1 input channel (for Y channel)
        SRCNN* model = init_srcnn(1);
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