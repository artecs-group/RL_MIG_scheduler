
#include "GPU_info.h"
#include "logging.h"
#include "MIG_manager.h"
#include <nvml.h>
#include <iostream>
#include <sys/time.h>

using namespace std;

// Information for A30
const GPUInfo A30_config = {
    "A30",
    {1, 2, 4},   // Valid sizes for the instances

    {{1, NVML_GPU_INSTANCE_PROFILE_1_SLICE},
     {2, NVML_GPU_INSTANCE_PROFILE_2_SLICE},
     {4, NVML_GPU_INSTANCE_PROFILE_4_SLICE}}, // Valid profiles for the GPU instances

    {{1, NVML_COMPUTE_INSTANCE_PROFILE_1_SLICE},
     {2, NVML_COMPUTE_INSTANCE_PROFILE_2_SLICE},
     {4, NVML_COMPUTE_INSTANCE_PROFILE_4_SLICE}}, // Valid profiles for the compute instances

    4            // Total number of slices
    
};

// Information for A100/H100
const GPUInfo A100_H100_config = {
    "A100/H100",
    {1, 2, 3, 4, 7},  // Valid sizes for the instances

    {{1, NVML_GPU_INSTANCE_PROFILE_1_SLICE},
     {2, NVML_GPU_INSTANCE_PROFILE_2_SLICE},
     {3, NVML_GPU_INSTANCE_PROFILE_3_SLICE},
     {4, NVML_GPU_INSTANCE_PROFILE_4_SLICE},
     {7, NVML_GPU_INSTANCE_PROFILE_7_SLICE},}, // Valid profiles for the GPU instances

    {{1, NVML_COMPUTE_INSTANCE_PROFILE_1_SLICE},
     {2, NVML_COMPUTE_INSTANCE_PROFILE_2_SLICE},
     {3, NVML_COMPUTE_INSTANCE_PROFILE_3_SLICE},
     {4, NVML_COMPUTE_INSTANCE_PROFILE_4_SLICE},
     {7, NVML_COMPUTE_INSTANCE_PROFILE_7_SLICE}}, // Valid profiles for the compute instances

    7                 // Total number of slices
};

// Global GPU info
const GPUInfo* global_GPU_info = nullptr;


void initialize_GPU_info(const string& gpu_name){
    if (gpu_name == "NVIDIA A30") {
        global_GPU_info = &A30_config;
    } else if (gpu_name == "NVIDIA A100" || gpu_name == "NVIDIA H100") {
        global_GPU_info = &A100_H100_config;
    } else {
        cerr << "GPU model unknown: " + gpu_name << endl;
        exit(1);
    }
}

void profile_reconfig_times(nvmlDevice_t device){
    cout << "\nProfiling instance creation and destruction times:" << endl;
    struct timeval init_time, end_time;
    for (unsigned int size: global_GPU_info->valid_instance_sizes){
        // Measure creation time
        gettimeofday(&init_time, NULL);
        Instance instance = create_instance(device, 0, size);
        gettimeofday(&end_time, NULL);
        double creation_time = (end_time.tv_sec - init_time.tv_sec) + (end_time.tv_usec - init_time.tv_usec) / 1000000.0;
        global_GPU_info->times_create[size] = creation_time;

        // Measure destruction time
        gettimeofday(&init_time, NULL);
        destroy_instance(instance);
        gettimeofday(&end_time, NULL);
        double destroy_time = (end_time.tv_sec - init_time.tv_sec) + (end_time.tv_usec - init_time.tv_usec) / 1000000.0;
        global_GPU_info->times_destroy[size] = destroy_time;

        LOG_INFO("Size " + to_string(size) + " Creation time: " + to_string(creation_time) + "s" + " Destruction time: " + to_string(destroy_time) + "s");
    }

}