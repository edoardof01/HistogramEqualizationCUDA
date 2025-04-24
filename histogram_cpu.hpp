#pragma once
#include <chrono>

std::chrono::duration<double, std::milli>  run_cpu_version(const char* inputFile, bool show_image = true);
