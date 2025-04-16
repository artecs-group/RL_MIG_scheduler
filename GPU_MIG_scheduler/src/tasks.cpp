#include "tasks.h"
#include "GPU_info.h"
#include "logging.h"
#include "utils.h"
#include <stdio.h>
#include <sys/time.h>
#include <stdexcept>
#include <cfloat>
#include <unistd.h>

using namespace std;


Task::Task(string const& name, string const& parent_path, string const& script_name) : name(name), parent_path(parent_path), script_name(script_name) {
    exec_times = {};
}

bool Task::operator==(const Task& other) const{
    return this->name == other.name;
}

bool Task::has_error() const{
    // Check if the task has an error (infinite time with all instances sizes)
    for (auto const& exec_time: this->exec_times){
        if (exec_time.second != DBL_MAX) return false;
    }
    return true;
}

static string exec_command(Task const& task, Instance const& instance){
    // Move to task directory
    string command = "cd " + task.parent_path;
    // Concatenate the command to execute the script
    command += " && CUDA_VISIBLE_DEVICES=" + instance.uuid + " sh " + task.script_name;

    // Get current directory
    char currentDirectory[1024];
    if (getcwd(currentDirectory, sizeof(currentDirectory)) == nullptr) {
        LOG_ERROR("Get current directory");
        exit(1);
    }

    // Return to the original directory
    command += " && cd " + string(currentDirectory);

    return command;
}

void Task::execute(Instance const& instance) const{
    // Redirect the stdout and stderr to a log file
    int stdout_backup, stderr_backup;
    redirect_output(*this, stdout_backup, stderr_backup);

    // Execute the task in the given instance
    string command = exec_command(*this, instance);
    int status = system(command.c_str());

    // Restore the stdout and stderr
    restore_output(stdout_backup, stderr_backup);
    if (status != 0){
        cout << "ERROR: Task " << this->name << " failed with " << instance << endl;
        // If there was an error executing the task, throw an exception to set infinite time for it
        throw runtime_error("Task execution failed");
    }
}

void Task::profile_times(nvmlDevice_t device){
    destroy_all_instances(device);
    struct timeval init_time, end_time;
    // For each possible instance size, execute every task
    for (int instance_size: global_GPU_info->valid_instance_sizes){
        // Create the instance
        Instance instance = create_instance(device, 0, instance_size);
        try{
            cout << "\nProfiling task " << this->name << " with size " << instance_size << endl;
            // Measure and save the execution time
            gettimeofday(&init_time, NULL);
            this->execute(instance);
            gettimeofday(&end_time, NULL);
        } catch (runtime_error const& e){
            // If the task failed, set infinite time for it
            this->exec_times[instance_size] = DBL_MAX;
            LOG_ERROR("Task " + this->name + " failed with size " + to_string(instance_size) + " and has infinite time");
            destroy_instance(instance);
            continue;
        }
        double task_time = (end_time.tv_sec - init_time.tv_sec) + (end_time.tv_usec - init_time.tv_usec) / 1000000.0;
        this->exec_times[instance_size] = task_time;
        LOG_INFO("Task " + this->name + " profiled with " + to_string(task_time) + " seconds in size " + to_string(instance_size));
        destroy_instance(instance);
    }
}