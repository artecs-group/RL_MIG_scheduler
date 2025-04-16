#ifndef CONFIG_H
#define CONFIG_H

#include <vector>
#include <string>
#include <unordered_map>
#include <nvml.h>

using namespace std;


// Struct for the GPU scheduling information
struct GPUInfo {
    string name;
    vector<unsigned int> valid_instance_sizes; // Valid sizes for the instances
    unordered_map<unsigned int, unsigned int> valid_gi_profiles;   // Valid profiles for the GPU instances
    unordered_map<unsigned int, unsigned int> valid_ci_profiles;   // Valid profiles for the compute instances
    int num_slices;                  // Total number of slices
    mutable unordered_map<unsigned int, double> times_create; // Time to create an instance of each size
    mutable unordered_map<unsigned int, double> times_destroy; // Time to destroy an instance of each size
};

// Declaration of the global GPU info
extern const GPUInfo* global_GPU_info;

// Procedure to initialize the GPU info
void initialize_GPU_info(const string& gpu_name);

// Procedure to profile the instance creation and destruction times
void profile_reconfig_times(nvmlDevice_t device);

#endif // CONFIG_H
