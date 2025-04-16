#include "MIG_manager.h"
#include "GPU_info.h"
#include "logging.h"
#include <iostream>
#include <cuda_runtime.h>
#include <nvml.h>

using namespace std;

Instance::Instance(size_t start, size_t size, nvmlGpuInstance_t gpuInstance, nvmlComputeInstance_t computeInstance, string uuid) : start(start), size(size), gpuInstance(gpuInstance), computeInstance(computeInstance), uuid(uuid) {}

bool Instance::operator==(const Instance& other) const{
    return start == other.start && size == other.size;
}

ostream& operator<<(ostream& os, const Instance& instance){
    os << "Instance(start=" << instance.start << ", size=" << instance.size << ")";
    return os;
}

string get_gpu_name(nvmlDevice_t device){
   // Obtener el nombre de la GPU
   char name[NVML_DEVICE_NAME_BUFFER_SIZE];
   nvmlReturn_t result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
    if (result == NVML_SUCCESS) {
        LOG_INFO("GPU model: " + string(name) + " detected");
    } else {
        LOG_ERROR("Error obteniendo el nombre de la GPU: " + string(nvmlErrorString(result)));
    }
    return string(name);
}

void init_nvml(){
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        LOG_ERROR("Failed to initialize NVML: " + string(nvmlErrorString(result)));
        exit(1);
    }
    LOG_INFO("NVML has been initialized");
}

nvmlDevice_t bind_device(int gpu_number){
    nvmlDevice_t device;
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex(gpu_number, &device);
    if (result != NVML_SUCCESS) {
        LOG_ERROR("Failed to bind device: " + string(nvmlErrorString(result)));
        exit(1);
    }
    LOG_INFO("Device has been binded");
    return device;
}

void MIG_enable(nvmlDevice_t device, int gpu_number){
    nvmlReturn_t status;
    nvmlReturn_t result = nvmlDeviceSetMigMode(device, NVML_DEVICE_MIG_ENABLE, &status);
    if (result != NVML_SUCCESS) {
        LOG_ERROR("Failed to activate MIG: " + string(nvmlErrorString(result)));
    }
    LOG_INFO("MIG has been activated");
}

void MIG_disable(nvmlDevice_t device, int gpu_number){
    nvmlReturn_t status;
    nvmlReturn_t result = nvmlDeviceSetMigMode(device, NVML_DEVICE_MIG_DISABLE, &status);
    if (result != NVML_SUCCESS) {
        LOG_ERROR("Failed to deactivate MIG: " + string(nvmlErrorString(result)));
    }
    LOG_INFO("MIG has been deactivated");
}

static void destroy_all_compute_instances(nvmlGpuInstance_t gpu_instance, unsigned int ci_profile){
    nvmlComputeInstanceProfileInfo_t ci_info;
    nvmlReturn_t result = nvmlGpuInstanceGetComputeInstanceProfileInfo (gpu_instance, ci_profile, NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED, &ci_info);
    if(result != NVML_SUCCESS){
        LOG_ERROR("Failed to get compute instance profile info: " + string(nvmlErrorString(result)));
        exit(1);
    }

    nvmlComputeInstance_t * compute_instances = (nvmlComputeInstance_t *) malloc(sizeof(nvmlComputeInstance_t) * global_GPU_info->num_slices);
    unsigned int compute_count;
    result = nvmlGpuInstanceGetComputeInstances(gpu_instance, ci_info.id, compute_instances, &compute_count);
    if(result != NVML_SUCCESS){
        LOG_ERROR("Failed to get compute instances: " + string(nvmlErrorString(result)));
        exit(1);
    }
    for (int j = 0; j < compute_count; j++){
        result = nvmlComputeInstanceDestroy(compute_instances[j]);
        if(result != NVML_SUCCESS){
            LOG_ERROR("Failed to destroy compute instance: " + string(nvmlErrorString(result)));
            exit(1);
        }
    }

    free(compute_instances);
}

void destroy_all_instances(nvmlDevice_t device){
    int destroyed_count = 0;
    nvmlGpuInstance_t * gpu_instances = (nvmlGpuInstance_t *) malloc(sizeof(nvmlGpuInstance_t) * global_GPU_info->num_slices);
    
    for (auto const& gpu_instance_profile: global_GPU_info->valid_gi_profiles){
        int instance_size = gpu_instance_profile.first;
        unsigned int profile = gpu_instance_profile.second;
        nvmlGpuInstanceProfileInfo_t info;
        nvmlReturn_t result = nvmlDeviceGetGpuInstanceProfileInfo(device, profile, &info);
        if(result != NVML_SUCCESS){
            LOG_ERROR("Failed to get GPU instance profile info: " + string(nvmlErrorString(result)));
            exit(1);
        }
        unsigned int count;
        result = nvmlDeviceGetGpuInstances(device, info.id, gpu_instances, &count);
        if(result != NVML_SUCCESS){
            LOG_ERROR("Failed to get GPU instances: " + string(nvmlErrorString(result)));
            exit(1);
        }
        for (int i = 0; i < count; i++){
            unsigned int ci_profile = global_GPU_info->valid_ci_profiles.at(instance_size);
            destroy_all_compute_instances(gpu_instances[i], ci_profile);

            result = nvmlGpuInstanceDestroy(gpu_instances[i]);
            if(result != NVML_SUCCESS){
                LOG_ERROR("Failed to destroy GPU instance: " + string(nvmlErrorString(result)));
                exit(1);
            }
        }
        destroyed_count += count;
    }

    LOG_INFO("All GPU instances have been destroyed: " + to_string(destroyed_count) << " instances");

    free(gpu_instances);

}

static string get_instance_uuid(nvmlDevice_t device, nvmlGpuInstance_t gpuInstance){
    // Get de id of this gpu instance
    nvmlGpuInstanceInfo_t gi_info;
    nvmlReturn_t result = nvmlGpuInstanceGetInfo(gpuInstance, &gi_info);
    if(result != NVML_SUCCESS){
        LOG_ERROR("Failed to get GPU instance info: " + string(nvmlErrorString(result)));
        exit(1);
    }
    unsigned int target_id = gi_info.id;
    unsigned int max_instances;
    result = nvmlDeviceGetMaxMigDeviceCount(device, &max_instances);
    if(result != NVML_SUCCESS){
        LOG_ERROR("Failed to get max MIG device count: " + string(nvmlErrorString(result)));
        exit(1);
    }
    // Search for its device handle
    nvmlDevice_t mig_device;
    for (int i = 0; i < max_instances; i++){     
        result = nvmlDeviceGetMigDeviceHandleByIndex (device, i, &mig_device);
        if(result != NVML_SUCCESS){
            LOG_ERROR("Failed to get MIG device handle: " + string(nvmlErrorString(result)));
            exit(1);
        }
        unsigned int id;
        result = nvmlDeviceGetGpuInstanceId(mig_device, &id);
        if(result != NVML_SUCCESS){
            LOG_ERROR("Failed to get GPU instance ID: " + string(nvmlErrorString(result)));
            exit(1);
        }
        if(id == target_id){           
            break;
        }       
    }

    // Get the UUID with the handle
    char uuid[NVML_DEVICE_UUID_V2_BUFFER_SIZE];
    result = nvmlDeviceGetUUID(mig_device, uuid, NVML_DEVICE_UUID_V2_BUFFER_SIZE);
    if(result != NVML_SUCCESS){
        LOG_ERROR("Failed to get instance UUID: " + string(nvmlErrorString(result)));
        exit(1);
    }
    //else LOG_INFO("Instance UUID: " + string(uuid));

    return string(uuid); 
}

Instance create_instance(nvmlDevice_t device, size_t start, size_t size){
    nvmlReturn_t result;
    
    // Create GPU instance
    nvmlGpuInstance_t gpuInstance;
    nvmlGpuInstancePlacement_t placement;
    placement.start = start;
    placement.size = size;

    // Get GPU instance profile
    nvmlGpuInstanceProfileInfo_t gi_info;
    result = nvmlDeviceGetGpuInstanceProfileInfo (device, global_GPU_info->valid_gi_profiles.at(placement.size), &gi_info);
    if(NVML_SUCCESS != result){
            LOG_ERROR("Failed getting gpu instance ID: " + string(nvmlErrorString(result)));
            exit(1);
    }

    // Create GPU instance
    result = nvmlDeviceCreateGpuInstanceWithPlacement(device, gi_info.id, &placement, &gpuInstance );
    if(NVML_SUCCESS != result){
        LOG_ERROR("Failed creating gpu instance: " + string(nvmlErrorString(result)));
        exit(1);
    }

    // Get compute instance profile
    nvmlComputeInstanceProfileInfo_t ci_info;
    
    result = nvmlGpuInstanceGetComputeInstanceProfileInfo (gpuInstance, global_GPU_info->valid_ci_profiles.at(placement.size), NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED, &ci_info );
    if(NVML_SUCCESS != result){
            LOG_ERROR("Failed getting compute instance ID: " + string(nvmlErrorString(result)));
            exit(1);
    }

    // Create compute instance
    nvmlComputeInstance_t computeInstance;
    result =  nvmlGpuInstanceCreateComputeInstance (gpuInstance, ci_info.id, &computeInstance );
    if(NVML_SUCCESS != result){
            LOG_ERROR("Failed creating compute instance: " + string(nvmlErrorString(result)));
            exit(1);
    }

    string uuid = get_instance_uuid(device, gpuInstance);

    // Construct the instance object
    Instance instance(start, size, gpuInstance, computeInstance, uuid);

    cout << "INFO: " << instance << " has been created\n";

    return instance;
}


void destroy_instance(Instance const& instance){
    nvmlReturn_t result = nvmlComputeInstanceDestroy(instance.computeInstance);
    if(result != NVML_SUCCESS){
        LOG_ERROR("Failed to destroy compute instance: " + string(nvmlErrorString(result)));
        exit(1);
    }
    result = nvmlGpuInstanceDestroy(instance.gpuInstance);
    // Sometimes the gpu instance destroy fails during a little time, so we try again. It will work if the compute instance has been destroyed
    while(result != NVML_SUCCESS){
        result = nvmlGpuInstanceDestroy(instance.gpuInstance);
    }
}