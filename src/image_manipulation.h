#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Bicubic interpolation function
float cubicInterpolate(float p[4], float x) {
    return p[1] + 0.5 * x * (p[2] - p[0] + x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));
}

// Bicubic upscaling function
void upscaleBicubic(const unsigned char* srcImg, unsigned char* dstImg, int srcWidth, int srcHeight, int channels) {
    int dstWidth = srcWidth * 2;
    int dstHeight = srcHeight * 2;

    for (int y = 0; y < dstHeight; y++) {
        float dy = (float)y / 2.0f;
        int sy = (int)dy;
        float fy = dy - sy;

        for (int x = 0; x < dstWidth; x++) {
            float dx = (float)x / 2.0f;
            int sx = (int)dx;
            float fx = dx - sx;

            for (int c = 0; c < channels; c++) {
                float p[4][4];
                for (int ky = 0; ky < 4; ky++) {
                    for (int kx = 0; kx < 4; kx++) {
                        int sampleY = sy + ky - 1;
                        int sampleX = sx + kx - 1;
                        sampleY = (sampleY < 0) ? 0 : (sampleY >= srcHeight) ? srcHeight - 1 : sampleY;
                        sampleX = (sampleX < 0) ? 0 : (sampleX >= srcWidth) ? srcWidth - 1 : sampleX;
                        p[ky][kx] = srcImg[(sampleY * srcWidth + sampleX) * channels + c];
                    }
                }

                float arr[4];
                for (int i = 0; i < 4; i++) {
                    arr[i] = cubicInterpolate(p[i], fx);
                }
                float val = cubicInterpolate(arr, fy);
                
                // Clamp the value to [0, 255]
                val = (val < 0) ? 0 : (val > 255) ? 255 : val;
                dstImg[(y * dstWidth + x) * channels + c] = (unsigned char)val;
            }
        }
    }
}

// Function to convert RGB to YCbCr
void rgbToYCbCr(unsigned char r, unsigned char g, unsigned char b, unsigned char* y, unsigned char* cb, unsigned char* cr) {
    *y  = (unsigned char)( 0.299 * r + 0.587 * g + 0.114 * b);
    *cb = (unsigned char)(-0.169 * r - 0.331 * g + 0.500 * b + 128);
    *cr = (unsigned char)( 0.500 * r - 0.419 * g - 0.081 * b + 128);
}

// Function to convert RGB image to YCbCr
void convertRGBToYCbCr(const unsigned char* rgbImage, unsigned char* ycbcrImage, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = (i * width + j) * 3;
            unsigned char r = rgbImage[idx];
            unsigned char g = rgbImage[idx + 1];
            unsigned char b = rgbImage[idx + 2];
            
            rgbToYCbCr(r, g, b, &ycbcrImage[idx], &ycbcrImage[idx + 1], &ycbcrImage[idx + 2]);
        }
    }
}

// Function to extract Y channel from YCbCr image
unsigned char* extractYChannel(const unsigned char* ycbcrImage, int width, int height) {
    unsigned char* yChannel = (unsigned char*)malloc(width * height);
    if (!yChannel) {
        fprintf(stderr, "Memory allocation failed for Y channel!\n");
        return NULL;
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int ycbcrIdx = (i * width + j) * 3;
            yChannel[i * width + j] = ycbcrImage[ycbcrIdx];
        }
    }

    return yChannel;
}

// Function to convert YCbCr to RGB
void ycbcrToRgb(unsigned char y, unsigned char cb, unsigned char cr, unsigned char* r, unsigned char* g, unsigned char* b) {
    int r_temp = y + 1.402 * (cr - 128);
    int g_temp = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128);
    int b_temp = y + 1.772 * (cb - 128);

    *r = (unsigned char)(r_temp < 0 ? 0 : (r_temp > 255 ? 255 : r_temp));
    *g = (unsigned char)(g_temp < 0 ? 0 : (g_temp > 255 ? 255 : g_temp));
    *b = (unsigned char)(b_temp < 0 ? 0 : (b_temp > 255 ? 255 : b_temp));
}

// Function to convert YCbCr image back to RGB using extracted Y channel
void convertYCbCrToRGB(const unsigned char* ycbcrImage, const unsigned char* enhancedYChannel, 
                       unsigned char* rgbImage, int origWidth, int origHeight, int newWidth, int newHeight) {
    // Calculate offsets to center the cropped region
    int xOffset = (origWidth - newWidth) / 2;
    int yOffset = (origHeight - newHeight) / 2;

    for (int y = 0; y < newHeight; y++) {
        for (int x = 0; x < newWidth; x++) {
            int newIdx = y * newWidth + x;
            int origIdx = ((y + yOffset) * origWidth + (x + xOffset)) * 3;

            unsigned char yVal = enhancedYChannel[newIdx];
            unsigned char cbVal = ycbcrImage[origIdx + 1];
            unsigned char crVal = ycbcrImage[origIdx + 2];

            unsigned char r, g, b;
            ycbcrToRgb(yVal, cbVal, crVal, &r, &g, &b);

            rgbImage[newIdx * 3] = r;
            rgbImage[newIdx * 3 + 1] = g;
            rgbImage[newIdx * 3 + 2] = b;
        }
    }
}

void reconstructAndSaveImage(int inputWidth, int inputHeight, int inputChannels, 
                             unsigned char* ycbcrImageData, float* srcnn_output, 
                             int srcnn_output_width, int srcnn_output_height, int outputWidth, int outputHeight,
                             const char* outputReconstructedRGBPath) {
    // Convert SRCNN output back to unsigned char
    unsigned char* enhanced_y_channel = (unsigned char*)malloc(srcnn_output_width * srcnn_output_height);
    for (int i = 0; i < srcnn_output_width * srcnn_output_height; i++) {
        float pixel = srcnn_output[i] * 255.0f;
        enhanced_y_channel[i] = (unsigned char)(pixel < 0 ? 0 : (pixel > 255 ? 255 : pixel));
    }

    // Reconstruct RGB image using enhanced Y channel
    unsigned char* reconstructedRGBImage = (unsigned char*)malloc(srcnn_output_width * srcnn_output_height * inputChannels);
    if (!reconstructedRGBImage) {
        fprintf(stderr, "Memory allocation failed for reconstructed RGB image!\n");
        return;
    }

    // Use the original Cb and Cr channels, but crop them to match the new Y channel size
    convertYCbCrToRGB(ycbcrImageData, enhanced_y_channel, reconstructedRGBImage, 
                    outputWidth, outputHeight, srcnn_output_width, srcnn_output_height);
    // Save the reconstructed RGB image
    if (stbi_write_png(outputReconstructedRGBPath, srcnn_output_width, srcnn_output_height, inputChannels, reconstructedRGBImage, srcnn_output_width * inputChannels) == 0) {
        fprintf(stderr, "Error writing reconstructed RGB image!\n");
    } else {
        printf("SRCNN upscaled RGB image saved to: %s\n", outputReconstructedRGBPath);
    }

    free(reconstructedRGBImage);
    free(enhanced_y_channel);
}

float* loadAndPreprocessImage(const char* inputImagePath,
                            int* inputWidth, int* inputHeight, int* inputChannels,
                            int* outputWidth, int* outputHeight, unsigned char** ycbcrImageDataPtr) {
    const char* outputBicubicPath = "./out/intermediary/zebra_upscaled_bicubic.png";
    const char* outputYCbCrImagePath = "./out/intermediary/zebra_upscaled_bicubic_ycbcr.png";
    const char* outputYChannelPath = "./out/intermediary/zebra_upscaled_bicubic_y_channel.png";
    
    unsigned char* inputImageData = stbi_load(inputImagePath, inputWidth, inputHeight, inputChannels, 0);
    if (!inputImageData) {
        fprintf(stderr, "Error loading image: %s\n", stbi_failure_reason());
    }

    *outputWidth = *inputWidth * 2;
    *outputHeight = *inputHeight * 2;

    unsigned char* outputImageData = (unsigned char*)malloc(*outputWidth * *outputHeight * *inputChannels);
    if (!outputImageData) {
        fprintf(stderr, "Memory allocation failed!\n");
        stbi_image_free(inputImageData);
    }

    upscaleBicubic(inputImageData, outputImageData, *inputWidth, *inputHeight, *inputChannels);

    // Save the upscaled RGB image
    if (stbi_write_png(outputBicubicPath, *outputWidth, *outputHeight, *inputChannels, outputImageData, *outputWidth * *inputChannels) == 0) {
        fprintf(stderr, "Error writing output image!\n");
    } else {
        printf("Bicubically upscaled image saved to: %s\n", outputBicubicPath);
    }

    // Convert to YCbCr
    unsigned char* ycbcrImageData = (unsigned char*)malloc(*outputWidth * *outputHeight * *inputChannels);
    if (!ycbcrImageData) {
        fprintf(stderr, "Memory allocation failed for YCbCr image!\n");
        stbi_image_free(inputImageData);
        free(outputImageData);
    }
    convertRGBToYCbCr(outputImageData, ycbcrImageData, *outputWidth, *outputHeight);

    // Save the YCbCr image
    if (stbi_write_png(outputYCbCrImagePath, *outputWidth, *outputHeight, *inputChannels, ycbcrImageData, *outputWidth * *inputChannels) == 0) {
        fprintf(stderr, "Error writing YCbCr image!\n");
    } else {
        printf("YCbCr image saved to: %s\n", outputYCbCrImagePath);
    }

    // Extract Y channel
    unsigned char* yChannel = extractYChannel(ycbcrImageData, *outputWidth, *outputHeight);
    if (!yChannel) {
        stbi_image_free(inputImageData);
        free(outputImageData);
        free(ycbcrImageData);
    }

    // Save the Y channel image
    if (stbi_write_png(outputYChannelPath, *outputWidth, *outputHeight, 1, yChannel, *outputWidth) == 0) {
        fprintf(stderr, "Error writing Y channel image!\n");
    } else {
        printf("Y channel image saved to: %s\n", outputYChannelPath);
    }

    // Prepare input for SRCNN (normalize Y channel)
    float* srcnn_input = (float*)malloc(*outputWidth * *outputHeight * sizeof(float));
    for (int i = 0; i < *outputWidth * *outputHeight; i++) {
        srcnn_input[i] = yChannel[i] / 255.0f;
    }

    stbi_image_free(inputImageData);
    free(outputImageData);
    free(yChannel);

    *ycbcrImageDataPtr = ycbcrImageData;
    return srcnn_input;
}