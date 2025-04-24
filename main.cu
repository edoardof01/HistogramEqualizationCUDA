#include <iostream>
#include <vector>
#include <string>
#include <iomanip> // per setw
#include "histogram_cpu.hpp"
#include "histogram_gpu.hpp"

int main() {
    std::vector<std::string> immagini = {
        "../images/eugenia-pankiv-9mDxdrqxtOE-unsplash.jpg",
        "../images/peter-thomas-72l6VkOroxo-unsplash.jpg",
        "../images/peter-thomas-JBuDeWF39iE-unsplash.jpg",
    };

    struct Risultato {
        std::string nome;
        float tempo_cpu;
        float tempo_gpu_A;
        float tempo_gpu_B;
        float speedup_A;
        float speedup_B;
    };

    std::vector<Risultato> risultati;

    for (size_t i = 0; i < immagini.size(); ++i) {
        const auto& img = immagini[i];
        std::string nome_immagine = "immagine" + std::to_string(i + 1);
        std::cout << "Elaborazione di " << nome_immagine << ": " << img << "\n";

        auto cpu_time = run_cpu_version(img.c_str(), false);
        float gpu_time_A = run_gpu_version(img.c_str(), false);
        float gpu_time_B = run_gpu_version_basic(img.c_str(), false);

        if (cpu_time.count() < 0.0f || gpu_time_A < 0.0f || gpu_time_B < 0.0f) {
            std::cerr << "Errore durante l'elaborazione di " << img << "\n";
            continue;
        }

        risultati.push_back({
            nome_immagine,
            static_cast<float>(cpu_time.count()),
            gpu_time_A,
            gpu_time_B,
            static_cast<float>(cpu_time.count()) / gpu_time_A,
            static_cast<float>(cpu_time.count()) / gpu_time_B
        });
    }

    // Stampa tabella riepilogativa
    std::cout << "\n================= RISULTATI RIEPILOGATIVI =================\n";
    std::cout << std::left
              << std::setw(12) << "Immagine"
              << std::setw(15) << "Tempo CPU"
              << std::setw(20) << "Tempo GPU A"
              << std::setw(20) << "Tempo GPU B"
              << std::setw(18) << "Speedup A"
              << std::setw(18) << "Speedup B"
              << "\n";

    std::cout << std::string(103, '-') << "\n";

    for (const auto& r : risultati) {
        std::cout << std::left
                  << std::setw(12) << r.nome
                  << std::setw(15) << r.tempo_cpu
                  << std::setw(20) << r.tempo_gpu_A
                  << std::setw(20) << r.tempo_gpu_B
                  << std::setw(18) << r.speedup_A
                  << std::setw(18) << r.speedup_B
                  << "\n";
    }

    return 0;
}
