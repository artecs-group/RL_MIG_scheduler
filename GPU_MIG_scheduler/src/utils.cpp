#include "utils.h"
#include "logging.h"
#include <fstream>
#include <cstdlib>
#include <sys/stat.h>
#include <algorithm>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

using namespace std;

void profile_tasks(vector<Task> & tasks, nvmlDevice_t device){
    vector<Task*> tasks_failed;
    for (auto & task: tasks){
        task.profile_times(device);
        if (task.has_error()){
            tasks_failed.push_back(&task);
        }
    }
    // Remove the tasks that failed from the list
    for (auto task: tasks_failed){
        tasks.erase(remove(tasks.begin(), tasks.end(), *task), tasks.end());
    }
    // Show the profiled times
    cout << "======================================" << endl;
    cout << "Profiled times for the tasks:" << endl;
    for (auto const& task: tasks){
        cout << "Task " << task.name << ":" << endl;
        for (auto const& exec_time: task.exec_times){
            cout << "Size " << exec_time.first << ": " << exec_time.second << " seconds" << endl;
        }
    }
    cout << "======================================" << endl;
}

static bool valid_exec_path(const string & path){
    struct stat fileInfo;
    // Verify if the path is valid and executable
    bool isValid = (stat(path.c_str(), &fileInfo) == 0) && (fileInfo.st_mode & S_IXUSR);
    if (!isValid){
        LOG_ERROR("Path " + path + " doesn't exist or is not executable");
    }
    return isValid;
}

vector<Task> get_tasks(const string & kernels_filename){    
    vector<Task> tasks;

    ifstream file(kernels_filename);
    if (!file.is_open()){
        LOG_ERROR("Could not open file " + kernels_filename);
        exit(1);
    }
    string name, parent_path, script_name; // Task data
    while (file >> name >> parent_path >> script_name){
        // Validate path as executable file
        string path = parent_path + "/" + script_name;
        bool isValid = valid_exec_path(path);
        if (isValid) tasks.push_back(Task(name, parent_path, script_name));
    }
    file.close();

    LOG_INFO("Tasks loaded from " + kernels_filename  + ":");
    for (auto const& task: tasks){
        cout << task.name << endl;
    }
    cout << endl;
    return tasks;
}

void redirect_output(Task const& task, int & stdout_backup, int & stderr_backup){
    // Get current directory
    char currentDirectory[1024];
    if (getcwd(currentDirectory, sizeof(currentDirectory)) == nullptr) {
        LOG_ERROR("Get current directory");
        exit(1);
    }
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    string dir_logs = string(currentDirectory) + "/logs-" + to_string(t->tm_year + 1900) + "-" + to_string(t->tm_mon + 1) + "-" + to_string(t->tm_mday) + "/";
    struct stat st;
    if (stat(dir_logs.c_str(), &st) == -1) {
        if (errno == ENOENT) {
            // The directory does not exist, create it
            if (mkdir(dir_logs.c_str(), 0755) == -1) {
                perror("Error creating /logs directory");
                exit(1);
            }
        } else {
            perror("Error checking /logs directory");
            exit(1);
        }
    }
    string file_log = dir_logs + task.name + ".log";
    int log_fd = open(file_log.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (log_fd == -1){
        LOG_ERROR("Error opening log file for task " + task.name);
        perror("open");
        exit(1);
    }
    stdout_backup = dup(STDOUT_FILENO);
    stderr_backup = dup(STDERR_FILENO);
    if (stdout_backup == -1 || stderr_backup == -1){
        LOG_ERROR("Error duplicating stdout and stderr");
        perror("dup");
        exit(1);
    }
    if (dup2(log_fd, STDOUT_FILENO) == -1 || dup2(log_fd, STDERR_FILENO) == -1){
        LOG_ERROR("Error redirecting stdout and stderr to log file");
        perror("dup2");
        exit(1);
    }
    close(log_fd);
}

void restore_output(int stdout_backup, int stderr_backup){
   if (dup2(stdout_backup, STDOUT_FILENO) == -1 || dup2(stderr_backup, STDERR_FILENO) == -1){
        LOG_ERROR("Error restoring stdout and stderr");
        perror("dup2");
        exit(1);
    }
    close(stdout_backup);
    close(stderr_backup);
}