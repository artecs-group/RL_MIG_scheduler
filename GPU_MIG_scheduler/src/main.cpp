#include "GPU_info.h"
#include "MIG_manager.h"
#include "utils.h"
#include "logging.h"
#include "FAR_scheduler.h"
using namespace std;

int main(int argc, char* argv[]){
     if (argc != 3){
          cerr << "Usage: " << argv[0] << " <gpu_number> <path to kernels filelist>" << endl;
          return 1;
     }
     int gpu_number = atoi(argv[1]);
     string kernels_filename = argv[2];

     DEBUG_PAUSE("Start scheduler initialization."); // Pause in debug mode with info

     // Init the compiler, bind with the device and init GPU config. for the scheduler.
     init_nvml();
     nvmlDevice_t device = bind_device(gpu_number);
     string gpu_name = get_gpu_name(device);
     initialize_GPU_info(gpu_name);

     DEBUG_PAUSE("Start getting tasks for scheduling."); // Pause in debug mode with info

     // Validate the scripts for scheduling
     vector<Task> tasks = get_tasks(kernels_filename);
     if (tasks.empty()){
          LOG_ERROR("No valid tasks for scheduling. Problem parsing the file " + kernels_filename);
          return 1;
     }

     // Enable MIG
     MIG_enable(device, gpu_number);

     // Destroy all instances to start from scratch
     destroy_all_instances(device);

     DEBUG_PAUSE("Start reconfig times profiling."); // Pause in debug mode with info
     
     // Profile instance creation and destruction times
     profile_reconfig_times(device);

     DEBUG_PAUSE("Start task profiling."); // Pause in debug mode with info

     // Profile tasks to get their execution times for each instance size
     profile_tasks(tasks, device);

     DEBUG_PAUSE("Start FAR algorithm scheduling."); // Pause in debug mode with info

     // Perform the scheduling of the tasks
     perform_scheduling(tasks, device);

     //Disable MIG
     MIG_disable(device, gpu_number);

     return 0;
}