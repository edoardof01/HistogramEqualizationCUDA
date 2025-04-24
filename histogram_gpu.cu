
#include "stb_image.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <filesystem>
#include "histogram_gpu.hpp"

// Inclusioni per Thrust per la scansione (prefix sum) su GPU
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>


__constant__ float d_rgbToYUVCoeff[9] = {
    0.299f,    // R coeff Y
    0.587f,    // G coeff Y
    0.114f,    // B coeff Y
    -0.14713f, // R coeff U
    -0.28886f, // G coeff U
    0.436f,    // B coeff U
    0.615f,    // R coeff V
    -0.51499f, // G coeff V
    -0.10001f  // B coeff V
};

__constant__ float d_yuvToRGBCoeff[9] = {
    1.0f,       0.0f,        1.13983f,   // coeff per ricavare R
    1.0f,      -0.39465f,   -0.58060f,   // coeff per ricavare G
    1.0f,       2.03211f,    0.0f        // coeff per ricavare B
};


// Funzioni device per conversione tra RGB e YUV
__device__ void rgbToYUV_device(const unsigned char r, const unsigned char g, const unsigned char b,
                                unsigned char &y, unsigned char &u, unsigned char &v)
{
    const auto rf = static_cast<float>(r);
    const auto gf = static_cast<float>(g);
    const auto bf = static_cast<float>(b);

    float yf = d_rgbToYUVCoeff[0] * rf + d_rgbToYUVCoeff[1] * gf + d_rgbToYUVCoeff[2] * bf;
    float uf = d_rgbToYUVCoeff[3] * rf + d_rgbToYUVCoeff[4] * gf + d_rgbToYUVCoeff[5] * bf + 128.0f;
    float vf = d_rgbToYUVCoeff[6] * rf + d_rgbToYUVCoeff[7] * gf + d_rgbToYUVCoeff[8] * bf + 128.0f;

    yf = fminf(fmaxf(yf, 0.0f), 255.0f);
    uf = fminf(fmaxf(uf, 0.0f), 255.0f);
    vf = fminf(fmaxf(vf, 0.0f), 255.0f);

    y = static_cast<unsigned char>(yf);
    u = static_cast<unsigned char>(uf);
    v = static_cast<unsigned char>(vf);
}

__device__ void yuvToRGB_device(const unsigned char y, const unsigned char u, const unsigned char v,
                                unsigned char &r, unsigned char &g, unsigned char &b)
{
    const auto yf = static_cast<float>(y);
    const float uf = static_cast<float>(u) - 128.0f;
    const float vf = static_cast<float>(v) - 128.0f;

    float rf = d_yuvToRGBCoeff[0] * yf + d_yuvToRGBCoeff[1] * uf + d_yuvToRGBCoeff[2] * vf;
    float gf = d_yuvToRGBCoeff[3] * yf + d_yuvToRGBCoeff[4] * uf + d_yuvToRGBCoeff[5] * vf;
    float bf = d_yuvToRGBCoeff[6] * yf + d_yuvToRGBCoeff[7] * uf + d_yuvToRGBCoeff[8] * vf;

    rf = fminf(fmaxf(rf, 0.0f), 255.0f);
    gf = fminf(fmaxf(gf, 0.0f), 255.0f);
    bf = fminf(fmaxf(bf, 0.0f), 255.0f);

    r = static_cast<unsigned char>(rf);
    g = static_cast<unsigned char>(gf);
    b = static_cast<unsigned char>(bf);
}

// KERNEL 1: Conversione RGB -> YUV
// Due versioni:
// - convertRGBtoYUV (base, non coalesced)
// - convertRGBtoYUV_coalesced (optimization)

__global__ void convertRGBtoYUV(const unsigned char* input,
                                unsigned char* Y,
                                unsigned char* U,
                                unsigned char* V,
                                const int width,
                                const int height)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; x < width && y < height) {
        const unsigned int idx = y * width + x;

        const unsigned int globalIdx = idx * 3;
        const unsigned char r = input[globalIdx + 0];
        const unsigned char g = input[globalIdx + 1];
        const unsigned char b = input[globalIdx + 2];

        unsigned char y_val, u_val, v_val;
        rgbToYUV_device(r, g, b, y_val, u_val, v_val);

        Y[idx] = y_val;
        U[idx] = u_val;
        V[idx] = v_val;
    }
}

__global__ void convertRGBtoYUV_coalesced(const uchar3* input,
                                          unsigned char* Y,
                                          unsigned char* U,
                                          unsigned char* V,
                                          const int width,
                                          const int height)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; x < width && y < height) {
        const unsigned int idx = y * width + x;

        uchar3 pixel = input[idx];
        unsigned char y_val, u_val, v_val;
        rgbToYUV_device(pixel.x,pixel.y,pixel.z,y_val,u_val,v_val);

        Y[idx] = y_val;
        U[idx] = u_val;
        V[idx] = v_val;
    }
}


// KERNEL 2: Calcolo dell'istogramma del canale Y
// Due versioni:
// - computeHistogram (base)
// - computeHistogramWarped (ottimizzata, con warps)

__global__ void computeHistogram(const unsigned char* Y,
                                int* histo,
                                 const int size)
{
    constexpr int NUM_BINS = 256;

    __shared__ unsigned int localHisto[NUM_BINS];
    const unsigned int threadId = threadIdx.x;

    if (threadId < NUM_BINS) {
        localHisto[threadId] = 0;
    }
    __syncthreads();

    unsigned int thread_global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = gridDim.x * blockDim.x;

    //Operazione di stride
    for (unsigned int idx = thread_global_idx; idx < size; idx += total_threads) {
        unsigned char y_val = Y[idx];
        atomicAdd(&localHisto[y_val], 1);
    }
    __syncthreads();

    if (threadId < NUM_BINS) {
        atomicAdd(&histo[threadId], static_cast<int>(localHisto[threadId]));
    }
}

__global__ void computeHistogramWarped(const unsigned char* Y,
                                       int* histo,
                                       const int size)
{
    constexpr int NUM_BINS = 256;
    constexpr int WARP_SIZE = 32;
    constexpr int MAX_WARPS = 2;

    const unsigned int threadId = threadIdx.x;
    const unsigned int warpId = threadId / WARP_SIZE;

    // Shared memory: un istogramma per ogni warp (max MAX_WARPS warps)
    __shared__ unsigned int warpHisto[MAX_WARPS][NUM_BINS];

    // Calcolo del numero effettivo di warps nel blocco, limitato a MAX_WARPS
    unsigned int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    numWarps = min(numWarps, MAX_WARPS);

    // Skip in sicurezza se warpId è fuori dal range (protezione extra)
    if (warpId >= MAX_WARPS) return;

    // Inizializzazione istogrammi per warp
    for (unsigned int bin = threadId; bin < numWarps * NUM_BINS; bin += blockDim.x) {
        const unsigned int w = bin / NUM_BINS;
        const unsigned int b = bin % NUM_BINS;
        warpHisto[w][b] = 0;
    }
    __syncthreads();

    const unsigned int thread_global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total_threads = gridDim.x * blockDim.x;

    // Ogni thread aggiorna il proprio warp-local histogram
    for (unsigned int idx = thread_global_idx; idx < size; idx += total_threads) {
        unsigned char y_val = Y[idx];
        atomicAdd(&warpHisto[warpId][y_val], 1);
    }
    __syncthreads();

    // Aggregazione da tutti i warpHisto → localHisto
    __shared__ unsigned int localHisto[NUM_BINS];
    if (threadId < NUM_BINS) {
        localHisto[threadId] = 0;
    }
    __syncthreads();

    for (unsigned int bin = threadId; bin < NUM_BINS; bin += blockDim.x) {
        for (int w = 0; w < numWarps; ++w) {
            atomicAdd(&localHisto[bin], warpHisto[w][bin]);
        }
    }
    __syncthreads();

    // Scrittura finale in memoria globale
    if (threadId < NUM_BINS) {
        atomicAdd(&histo[threadId], static_cast<int>(localHisto[threadId]));
    }
}

// KERNEL 3: Applicazione della LUT al canale Y
__global__ void applyLUT_global(unsigned char* Y, const unsigned char* d_LUT, const int size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (; idx < size; idx += stride) {
        unsigned char val = Y[idx];
        Y[idx] = d_LUT[val];
    }
}


// KERNEL 4: Conversione da YUV a RGB
// Due versioni:
// - convertYUVtoRGB (base)
// - convertYUVtoRGB_coalesced (ottimizzata)

__global__ void convertYUVtoRGB(const unsigned char* Y,
                                const unsigned char* U,
                                const unsigned char* V,
                                unsigned char* outputRGB,
                                const int width,
                                const int height)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; x < width && y < height) {
        const unsigned int idx = y * width + x;
        const unsigned int rgb_idx = idx * 3;

        const unsigned char y_val = Y[idx];
        const unsigned char u_val = U[idx];
        const unsigned char v_val = V[idx];

        unsigned char r, g, b;
        yuvToRGB_device(y_val, u_val, v_val, r, g, b);

        outputRGB[rgb_idx + 0] = r;
        outputRGB[rgb_idx + 1] = g;
        outputRGB[rgb_idx + 2] = b;
    }
}

__global__ void convertYUVtoRGB_coalesced(const unsigned char* Y,
                                          const unsigned char* U,
                                          const unsigned char* V,
                                          uchar3* outputRGB,
                                          const int width,
                                          const int height)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; x < width && y < height) {
        const unsigned int idx = y * width + x;
        const unsigned char y_val = Y[idx];
        const unsigned char u_val = U[idx];
        const unsigned char v_val = V[idx];

        unsigned char r, g, b;
        yuvToRGB_device(y_val, u_val, v_val, r, g, b);
        outputRGB[idx] = make_uchar3(r, g, b); // Scrittura coalesced
    }
}

// NUOVI KERNEL: Calcolo della LUT su GPU
__global__ void computeLUT_gpu(const int* d_cdf, unsigned char* d_LUT, const int size)
{
    __shared__ int cdf_min;

    if (threadIdx.x == 0) {
        cdf_min = 0;
        for (int i = 0; i < 256; i++) {
            if (d_cdf[i] != 0) {
                cdf_min = d_cdf[i];
                break;
            }
        }
    }

    __syncthreads();

    if (unsigned int i = threadIdx.x; i < 256) {
        float val = static_cast<float>(d_cdf[i] - cdf_min) / static_cast<float>(size - cdf_min) * 255.0f;
        val = fminf(fmaxf(val, 0.0f), 255.0f);
        d_LUT[i] = static_cast<unsigned char>(val);
    }
}



// FUNZIONE HOST: Versione Alternativa
// Utilizza i kernel alternativi (coalesced e warp)
float run_gpu_version(const char* inputFile, bool show_image)
{
    int width, height, channels;
    unsigned char* input = stbi_load(inputFile, &width, &height, &channels, 3);
    if (!input) {
        std::cerr << "Errore nel caricamento dell'immagine: " << inputFile << "\n";
        return -1.0f;
    }
    std::cout << "Immagine caricata: " << width << "x" << height << " - Canali: 3\n";
    const int size = width * height;

    uchar3 *d_inputRGB, *d_outputRGB;
    cudaMalloc(&d_inputRGB, size * sizeof(uchar3));
    cudaMemcpy(d_inputRGB, input,size * sizeof(uchar3),
        cudaMemcpyHostToDevice);

    unsigned char *d_Y, *d_U, *d_V;
    cudaMalloc(&d_Y, size * sizeof(unsigned char));
    cudaMalloc(&d_U, size * sizeof(unsigned char));
    cudaMalloc(&d_V, size * sizeof(unsigned char));

    cudaEvent_t startTotal, stopTotal;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventRecord(startTotal);

    {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        convertRGBtoYUV_coalesced<<<gridSize, blockSize>>>(d_inputRGB, d_Y, d_U, d_V, width, height);
    }
    cudaDeviceSynchronize();

    int *d_histogram;
    cudaMalloc(&d_histogram, 256 * sizeof(int));
    cudaMemset(d_histogram, 0, 256 * sizeof(int));
    {
        int blockSize = 256;
        int gridSize  = (size + blockSize - 1) / blockSize;
        computeHistogramWarped<<<gridSize, blockSize>>>(d_Y, d_histogram, size);
    }
    cudaDeviceSynchronize();

    // Calcolo CDF e LUT sull'host (al posto di Thrust)
    int h_histogram[256];
    cudaMemcpy(h_histogram,d_histogram,256 * sizeof(int),
        cudaMemcpyDeviceToHost);

    int cdf[256];
    cdf[0] = h_histogram[0];
    for (int i = 1; i < 256; ++i)
        cdf[i] = cdf[i - 1] + h_histogram[i];

    int cdf_min = 0;
    for (int i = 0; i < 256; ++i) {
        if (cdf[i] != 0) {
            cdf_min = cdf[i];
            break;
        }
    }

    unsigned char h_LUT[256];
    for (int i = 0; i < 256; ++i) {
        float val = static_cast<float>(cdf[i] - cdf_min) / static_cast<float>(size - cdf_min) * 255.0f;
        h_LUT[i] = static_cast<unsigned char>(std::clamp(val, 0.0f, 255.0f));
    }

    unsigned char *d_LUT;
    cudaMalloc(&d_LUT, 256 * sizeof(unsigned char));
    cudaMemcpy(d_LUT, h_LUT, 256 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    {
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        applyLUT_global<<<gridSize, blockSize>>>(d_Y, d_LUT, size);
    }
    cudaDeviceSynchronize();

    {
        cudaMalloc(&d_outputRGB, size * sizeof(uchar3));
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        convertYUVtoRGB_coalesced<<<gridSize, blockSize>>>(d_Y, d_U, d_V, d_outputRGB, width, height);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(stopTotal);
    cudaEventSynchronize(stopTotal);
    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, startTotal, stopTotal);

    cudaEventDestroy(startTotal);
    cudaEventDestroy(stopTotal);

    std::vector<uchar3> output(size);
    cudaMemcpy(output.data(), d_outputRGB, size * sizeof(uchar3), cudaMemcpyDeviceToHost);

    stbi_image_free(input);
    cudaFree(d_inputRGB);
    cudaFree(d_Y);
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_outputRGB);
    cudaFree(d_histogram);
    cudaFree(d_LUT);

    return elapsedMs;
}


// FUNZIONE HOST: Versione BASIC
// Utilizza i kernel base (non coalesced e computeHistogram)
float run_gpu_version_basic(const char* inputFile, bool show_image)
{
    int width, height, channels;
    unsigned char* input = stbi_load(inputFile, &width, &height, &channels, 3);
    if (!input) {
        std::cerr << "Errore nel caricamento dell'immagine: " << inputFile << "\n";
        return -1.0f;
    }
    std::cout << "Immagine caricata: " << width << "x" << height << " - Canali: 3\n";
    const int size = width * height;

    unsigned char *d_inputRGB, *d_outputRGB;
    cudaMalloc(&d_inputRGB, size * 3 * sizeof(unsigned char));
    cudaMemcpy(d_inputRGB, input, size * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    unsigned char *d_Y, *d_U, *d_V;
    cudaMalloc(&d_Y, size * sizeof(unsigned char));
    cudaMalloc(&d_U, size * sizeof(unsigned char));
    cudaMalloc(&d_V, size * sizeof(unsigned char));

    cudaEvent_t startTotal, stopTotal;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventRecord(startTotal);

    {
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        convertRGBtoYUV<<<gridSize, blockSize>>>(d_inputRGB, d_Y, d_U, d_V, width, height);
    }
    cudaDeviceSynchronize();

    int *d_histogram;
    cudaMalloc(&d_histogram, 256 * sizeof(int));
    cudaMemset(d_histogram, 0, 256 * sizeof(int));
    {
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        computeHistogram<<<gridSize, blockSize>>>(d_Y, d_histogram, size);
    }
    cudaDeviceSynchronize();

    // --- Calcolo CDF e LUT sull’host (senza Thrust) ---
    int h_histogram[256];
    cudaMemcpy(h_histogram, d_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);


    int cdf[256];
    cdf[0] = h_histogram[0];
    for (int i = 1; i < 256; ++i)
        cdf[i] = cdf[i - 1] + h_histogram[i];

    int cdf_min = 0;
    for (int i = 0; i < 256; ++i) {
        if (cdf[i] != 0) {
            cdf_min = cdf[i];
            break;
        }
    }
    unsigned char h_LUT[256];
    for (int i = 0; i < 256; ++i) {
        float val = static_cast<float>(cdf[i] - cdf_min) / static_cast<float>(size - cdf_min) * 255.0f;
        h_LUT[i] = static_cast<unsigned char>(std::clamp(val, 0.0f, 255.0f));
    }

    unsigned char *d_LUT;
    cudaMalloc(&d_LUT, 256 * sizeof(unsigned char));
    cudaMemcpy(d_LUT, h_LUT, 256 * sizeof(unsigned char), cudaMemcpyHostToDevice);


    {
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        applyLUT_global<<<gridSize, blockSize>>>(d_Y, d_LUT, size);
    }
    cudaDeviceSynchronize();

    {
        cudaMalloc(&d_outputRGB, size * 3 * sizeof(unsigned char));
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        convertYUVtoRGB<<<gridSize, blockSize>>>(d_Y, d_U, d_V, d_outputRGB, width, height);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(stopTotal);
    cudaEventSynchronize(stopTotal);
    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, startTotal, stopTotal);

    cudaEventDestroy(startTotal);
    cudaEventDestroy(stopTotal);

    std::vector<unsigned char> output(size * 3);
    cudaMemcpy(output.data(), d_outputRGB, size * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Cleanup
    stbi_image_free(input);
    cudaFree(d_inputRGB);
    cudaFree(d_Y);
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_outputRGB);
    cudaFree(d_histogram);
    cudaFree(d_LUT);

    return elapsedMs;
}



