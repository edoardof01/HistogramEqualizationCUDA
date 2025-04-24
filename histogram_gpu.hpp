#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    float run_gpu_version(const char* inputFile, bool show_image = true);
    float run_gpu_version_basic(const char* inputFile, bool show_image = true);

#ifdef __cplusplus
}
#endif
