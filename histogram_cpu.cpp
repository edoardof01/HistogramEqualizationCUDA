#include <chrono>
#include "stb_image.h"
#include "stb_image_write.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <fstream>

void rgbToYUV(const unsigned char r, const unsigned char g, const unsigned char b,
              unsigned char &y, unsigned char &u, unsigned char &v) {
    const auto rf = static_cast<float>(r);
    const auto gf = static_cast<float>(g);
    const auto bf = static_cast<float>(b);

    const float yf = 0.299f * rf + 0.587f * gf + 0.114f * bf;
    const float uf = -0.14713f * rf - 0.28886f * gf + 0.436f * bf + 128.0f;
    const float vf = 0.615f * rf - 0.51499f * gf - 0.10001f * bf + 128.0f;

    y = static_cast<unsigned char>(std::clamp(yf, 0.0f, 255.0f));
    u = static_cast<unsigned char>(std::clamp(uf, 0.0f, 255.0f));
    v = static_cast<unsigned char>(std::clamp(vf, 0.0f, 255.0f));
}

void yuvToRGB(const unsigned char y, const unsigned char u, const unsigned char v,
              unsigned char &r, unsigned char &g, unsigned char &b) {
    const auto yf = static_cast<float>(y);
    const float uf = static_cast<float>(u) - 128.0f;
    const float vf = static_cast<float>(v) - 128.0f;

    const float rf = yf + 1.13983f * vf;
    const float gf = yf - 0.39465f * uf - 0.58060f * vf;
    const float bf = yf + 2.03211f * uf;

    r = static_cast<unsigned char>(std::clamp(rf, 0.0f, 255.0f));
    g = static_cast<unsigned char>(std::clamp(gf, 0.0f, 255.0f));
    b = static_cast<unsigned char>(std::clamp(bf, 0.0f, 255.0f));
}

std::chrono::duration<double, std::milli> run_cpu_version(const char* imagePath, const bool show_image) {
    auto start = std::chrono::high_resolution_clock::now();

    int width, height, channels;
    unsigned char* input = stbi_load(imagePath, &width, &height, &channels, 3);
    if (!input) {
        std::cerr << "Errore nel caricamento dell'immagine: " << imagePath << "\n";
        return std::chrono::duration<double, std::milli>(-1.0f);
    }

    const int size = width * height;
    std::vector<unsigned char> Y(size), U(size), V(size);

    for (int i = 0; i < size; ++i)
        rgbToYUV(input[i*3], input[i*3+1], input[i*3+2], Y[i], U[i], V[i]);


    std::vector histogram(256, 0);
    for (int i = 0; i < size; ++i)
        histogram[Y[i]]++;


    std::ofstream hist_before("histogram_before.txt");
    for (int i = 0; i < 256; ++i)
        hist_before << histogram[i] << "\n";
    hist_before.close();


    std::vector cdf(256, 0);
    cdf[0] = histogram[0];
    for (int i = 1; i < 256; ++i)
        cdf[i] = cdf[i-1] + histogram[i];

    int cdf_min = 0;
    for (int i = 0; i < 256; ++i) {
        if (cdf[i] != 0) {
            cdf_min = cdf[i];
            break;
        }
    }
    std::vector<unsigned char> lut(256);
    for (int i = 0; i < 256; ++i) {
        float val = static_cast<float>(cdf[i] - cdf_min) / static_cast<float>(size - cdf_min) * 255.0f;
        lut[i] = static_cast<unsigned char>(std::clamp(val, 0.0f, 255.0f));
    }

    // Equalizzazione
    for (int i = 0; i < size; ++i)
        Y[i] = lut[Y[i]];

    // Istogramma dopo equalizzazione
    std::vector histogram_after(256, 0);
    for (int i = 0; i < size; ++i)
        histogram_after[Y[i]]++;

    std::ofstream hist_after("histogram_after.txt");
    for (int i = 0; i < 256; ++i)
        hist_after << histogram_after[i] << "\n";
    hist_after.close();

    // Conversione YUV → RGB
    std::vector<unsigned char> output(size * 3);
    for (int i = 0; i < size; ++i)
        yuvToRGB(Y[i], U[i], V[i], output[i*3], output[i*3 + 1], output[i*3 + 2]);

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;

    // Salvataggio immagine
    std::filesystem::path p(imagePath);
    std::string outputPath = "equalized_cpu_" + p.stem().string() + ".jpg";
    stbi_write_jpg(outputPath.c_str(), width, height, 3, output.data(), 100);

    // Liberazione memoria
    stbi_image_free(input);

    // Chiamata a script Python per plotting
    system("../venv/bin/python3 ../plot_histograms.py");



    return duration;
}
