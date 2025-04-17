#include "RL_scheduler.h"
#include "GPU_info.h"
#include "logging.h"
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <fstream>

static void write_taskfile(string const& taskfile_path, vector<Task> const& tasks);
static vector<Action> parse_outputfile(string const& output_file);
static void execute_inference(string const& model_path, string const& taskfile_path);
static void show_actions(vector<Action> const& actions);

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

static vector<Action> parse_outputfile(string const& output_file){
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
            vector<unsigned int> instance_sizes = {1,1,1,1,1,1,1};
            unsigned int size;
            output >> size;
            actions.push_back(Reconfig{instance_sizes});
        } else if (type_action == "assign") {
            string task_name;
            unsigned int num_instance;
            output >> task_name >> num_instance;
            actions.push_back(Execute{task_name, num_instance});
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

static void show_actions(vector<Action> const& actions){

    LOG_INFO ("\n\nScheduling plan generated:");
    for (Action const& action: actions){
        if (holds_alternative<Reconfig>(action)){
            auto reconfig = get<Reconfig>(action);
            cout << "Reconfigure to: ";
            for (auto size: reconfig.instance_sizes){
                cout << size << ' ';
            }
            cout << endl;
        } else if (holds_alternative<Execute>(action)){
            auto execute = get<Execute>(action);
            cout << "Execute task " << execute.task_name << " on instance " << execute.num_instance << endl;
        }
    }
    cout << "\n\n";
}

vector<Action> schedule_tasks(string const& model_path, vector<Task> const& tasks){
    const string taskfile_path = "../src/RL_scheduler/tmp/task_info.txt";
    // Write task info in a file
    write_taskfile(taskfile_path, tasks);
    LOG_INFO("Tasks info written in " + taskfile_path);
    LOG_INFO("Executing inference of the RL model " + model_path);
    // Execute inferences of RL model to get the scheduling actions
    execute_inference(model_path, taskfile_path);
    // Output file
    string output_file = "../src/RL_scheduler/tmp/schedule.txt";
    vector<Action> actions = parse_outputfile(output_file);
    // Show the actions
    show_actions(actions);
    return actions;
}

void perform_RL_schedule(string const& model_path, vector<Task> const& tasks){
    schedule_tasks(model_path, tasks);
}