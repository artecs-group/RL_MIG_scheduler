#include "RL_scheduler.h"
#include "GPU_info.h"
#include "logging.h"
#include "MIG_manager.h"
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <variant>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <sys/time.h>

// 2 possible kinds of actions
struct Reconfig{
    vector<unsigned int> instance_sizes; // Description of new configuration
    vector<unsigned int> slices_needed; // Slices needed for reconfiguration
};
struct Execute{
    shared_ptr<Task> task; // Name of the task to execute
    unsigned int num_instance; // Number of the instance to execute the task
    unsigned int first_slice; // Fisrt slice of the instance
    unsigned int instance_size; // Instance size
    vector<unsigned int> slices_needed; // Slices needed for the task
};
using Action = variant<Reconfig, Execute>;

// Global variables
static mutex mtx;
static condition_variable cv;
static vector<bool> slices_free;
static vector<unsigned int> current_config;
static unordered_map<int, shared_ptr<Instance>> instances; // Para cada slice, su instancia correspondiente


vector<Action> schedule_tasks(string const& model_path, vector<Task> const& tasks); // Call the RL model to return the scheduling actions


static void write_taskfile(string const& taskfile_path, vector<Task> const& tasks);
static vector<Action> parse_outputfile(string const& output_file, vector<Task> const& tasks);
static void execute_inference(string const& model_path, string const& taskfile_path);
static void show_actions(vector<Action> const& actions);
static void execute_actions(vector<Action> const& actions, nvmlDevice_t device);
static void compute_resources(Action & action);
static void acquire_resources(Action const& action);
static void perform_action(Action const& action, nvmlDevice_t device);
static void release_resources(Action const& action);



static void write_taskfile(string const& taskfile_path, vector<Task> const& tasks){
    // Open the taskfile for writing
    ofstream taskfile(taskfile_path, ios::out | ios::trunc);
    chmod(taskfile_path.c_str(), 0777);
    if (!taskfile.is_open()) {
        perror("Error opening task file");
        exit(1);
    }
    // Write the task info
    for (auto const& task : tasks) {
        taskfile << "Task " << task.name << '\n';
        for (int size : global_GPU_info->valid_instance_sizes) {
            taskfile << size << ' ' << task.exec_times.at(size) << '\n';
            if(size == 2 ){
                taskfile << "3 " << task.exec_times.at(size) << '\n';
            }
            else if(size == 4){
                taskfile << "7 " << task.exec_times.at(size) << '\n';
            }
        }
    }
    taskfile.close();
}

static vector<unsigned int> get_instance_sizes(ifstream & output){
    vector<unsigned int> instance_sizes;
    for (int i = 0; i < global_GPU_info->num_slices; i++){
        unsigned int size;
        output >> size;
        instance_sizes.push_back(size);
    }
    return instance_sizes;
}

static vector<Action> parse_outputfile(string const& output_file, vector<Task> const& tasks){
    // Open the output file for reading
    ifstream output(output_file);
    if (!output.is_open()) {
        perror("Error opening output file");
        exit(1);
    }
    vector<Action> actions;
    string type_action;
    output >> type_action;
    while (output) {
        if (type_action == "reconfig") {
            vector<unsigned int> instance_sizes = get_instance_sizes(output); // Read new configuration
            actions.push_back(Reconfig{instance_sizes});
        } else if (type_action == "assign") {
            string task_name;
            unsigned int num_instance;
            output >> task_name >> num_instance;
            // Find the task in the tasks vector
            auto it = find_if(tasks.begin(), tasks.end(), [&task_name](Task const& task) {
                return task.name == task_name;
            });
            actions.push_back(Execute{make_shared<Task>(*it), num_instance});
        }
        output >> type_action;
    }
    output.close();
    return actions;
}


static void execute_inference(string const& model_path, string const& taskfile_path){
    // Activate python environment
    const char* venv = getenv("VIRTUAL_ENV");
    if (venv == nullptr) {
        cerr << "Error: VIRTUAL_ENV environment variable not set." << endl;
        exit(1);
    }
    // Executing RL model inference
    string cmd = string(venv) + "/bin/python ";
    cmd += "../src/RL_scheduler/inferences.py";
    cmd += " --model_path " + model_path;
    cmd += " --task_path " + taskfile_path;
    cmd += " --output_path ../src/RL_scheduler/tmp/schedule.txt";
    int ret = system(cmd.c_str());
    if (ret == -1) {
        perror("Error executing command");
        exit(1);
    }
}

static string reconfigure_to_string(Reconfig const& reconfig){
    string reconfig_str = "Reconfigure to: ";
    for (auto size: reconfig.instance_sizes){
        reconfig_str += to_string(size) + ' ';
    }
    return reconfig_str;
}


static void show_actions(vector<Action> const& actions){

    cout << "\n\nScheduling plan generated:\n";
    for (Action const& action: actions){
        if (holds_alternative<Reconfig>(action)){
            auto reconfig = get<Reconfig>(action);
            LOG_INFO(reconfigure_to_string(reconfig));
        } else if (holds_alternative<Execute>(action)){
            auto execute = get<Execute>(action);
            LOG_INFO("Execute task " + execute.task->name + " on instance " + to_string(execute.num_instance));
        }
    }
    cout << "\n\n";
}

vector<Action> schedule_tasks(string const& model_path, vector<Task> const& tasks){
    const string taskfile_path = "../src/RL_scheduler/tmp/task_info.txt";
    // Write task info in a file
    //write_taskfile(taskfile_path, tasks);
    LOG_INFO("Tasks info written in " + taskfile_path);
    LOG_INFO("Executing inference of the RL model " + model_path);
    // Execute inferences of RL model to get the scheduling actions
    // execute_inference(model_path, taskfile_path);
    // Output file
    string output_file = "../src/RL_scheduler/tmp/schedule.txt";
    vector<Action> actions = parse_outputfile(output_file, tasks);
    // Show the actions
    show_actions(actions);
    return actions;
}

static void compute_resources(Action & action){
    {
        unique_lock<mutex> lk(mtx);
        if (holds_alternative<Reconfig>(action)){
            auto& reconfig = get<Reconfig>(action);
            // Need free to reconfigure the GPU
            for (int i = 0; i < global_GPU_info->num_slices; i++){
                if (current_config[i] != reconfig.instance_sizes[i]){
                    reconfig.slices_needed.push_back(i);
                }
            }
            current_config = reconfig.instance_sizes;
        }
        else if (holds_alternative<Execute>(action)){
            auto& execute = get<Execute>(action);
            int first_slice = 0;
            for (int i = 0; i < execute.num_instance; i++){
                first_slice += current_config[i];
            }
            execute.first_slice = first_slice;
            execute.instance_size = current_config[execute.num_instance];
            // Need free to execute the task
            for (int i = 0; i < execute.instance_size; i++){
                execute.slices_needed.push_back(first_slice + i);
            }
        }
    }
}


static void acquire_resources(Action const& action){
    auto slices = std::visit([](auto const& a){
        return a.slices_needed;   // Suppose that both structs have that field
    }, action);
    {
        unique_lock<mutex> lk(mtx);
        
        // Wait until the slices are free
        cv.wait(lk, [&]{
            for (int slice_need: slices){
                // There is a slice to be changed not free
                if (!slices_free[slice_need]){
                    return false;
                }
            }
            return true;
        });

        // Mark the slices as not free
        for (int slice_need: slices){
            slices_free[slice_need] = false;
        }
    }
}

static void perform_action(Action const& action, nvmlDevice_t device){
    // Initialize the time only once at the first call
    static struct timeval init_time = {0, 0};
    if (init_time.tv_sec == 0 && init_time.tv_usec == 0){
        gettimeofday(&init_time, NULL);
    }

    if (holds_alternative<Reconfig>(action)){
        auto reconfig = get<Reconfig>(action);
        LOG_INFO(reconfigure_to_string(reconfig));
        // Destroy previous instances in the modified slices
        unordered_set<shared_ptr<Instance>> instances_to_destroy;
        for (int slice: reconfig.slices_needed){
            instances_to_destroy.insert(instances[slice]);
            instances[slice] = nullptr;
        }
        for (auto instance: instances_to_destroy){
            // Destroy the instance
            destroy_instance(*instance);
            LOG_INFO("Destroyed instance " + to_string(instance->start) + " with size " + to_string(instance->size));
        }
        // Create new instances in the modified slices
        for (int slice = 0; slice < global_GPU_info->num_slices; slice++){
            if (instances[slice] == nullptr){
                // Create the instance with the new size
                Instance new_instance = create_instance(device, slice, reconfig.instance_sizes[slice]);
                for (int i = 0; i < reconfig.instance_sizes[slice]; i++){
                    instances[slice + i] = make_shared<Instance>(new_instance);
                }
            }
        }
    }
    else if (holds_alternative<Execute>(action)){
        auto execute = get<Execute>(action);
        struct timeval curr_time;
        shared_ptr<Task> task = execute.task;
        // Get the instance to execute the task
        auto instance = instances[execute.first_slice];
        gettimeofday(&curr_time, NULL);
        double curr_time_s = (curr_time.tv_sec - init_time.tv_sec) + (curr_time.tv_usec - init_time.tv_usec) / 1000000.0;
        LOG_INFO("Start task " + task->name + " at " + to_string(curr_time_s) + "s");
        task->execute(*instance);
        gettimeofday(&curr_time, NULL);
        curr_time_s = (curr_time.tv_sec - init_time.tv_sec) + (curr_time.tv_usec - init_time.tv_usec) / 1000000.0;
        LOG_INFO("End task " + task->name + " at " + to_string(curr_time_s) + "s");
    }
}

static void release_resources(Action const& action){
    auto slices = std::visit([](auto const& a){
        return a.slices_needed;   // Suppose that both structs have that field
    }, action);
    {
        lock_guard<mutex> lk(mtx);
        for (int slice_need: slices){
            slices_free[slice_need] = true;
        }
    }
    cv.notify_one();
}


static void execute_actions(vector<Action> & actions, nvmlDevice_t device){
    // Init to config. max and all GPU free
    Instance instance = create_instance(device, 0, global_GPU_info->num_slices);
    auto instance_ptr = make_shared<Instance>(instance);
    for (int i = 0; i < global_GPU_info->num_slices; i++){
        instances[i] = instance_ptr;
    }
    // Set all slices as free
    slices_free = vector<bool>(global_GPU_info->num_slices, true);
    current_config = vector<unsigned int>(global_GPU_info->num_slices, global_GPU_info->num_slices); // In A100, size 7 times 7

    vector<thread> threads;
    for (Action & action: actions){
        compute_resources(action);
        acquire_resources(action);
        // Execute the action in a different thread
        threads.emplace_back([action = move(action), device](){
            // Perform the action
            perform_action(action, device);           
            // Release the resources
            release_resources(action);
        });
    }

    // Wait for all threads to finish
    for (auto & thread: threads){
        thread.join();
    }
    LOG_INFO("All actions executed\n");
}

void perform_RL_schedule(string const& model_path, vector<Task> const& tasks, nvmlDevice_t device){
    // Compute scheduling with the RL model
    vector<Action> actions = schedule_tasks(model_path, tasks);

    // Execute the actions in the GPU
    execute_actions(actions, device);
}