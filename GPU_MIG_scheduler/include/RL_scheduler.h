#ifndef RL_SCHEDULER_H
#define RL_SCHEDULER_H

#include "tasks.h"
#include <variant>
using namespace std;

// 2 possible kinds of actions
struct Reconfig{
    vector<unsigned int> instance_sizes; // Description of new configuration
};
struct Execute{
    string task_name; // Name of the task to execute
    unsigned int num_instance; // Number of the instance to execute the task
};
using Action = variant<Reconfig, Execute>;


vector<Action> schedule_tasks(string const& model_path, vector<Task> const& tasks); // Call the RL model to return the scheduling actions

void perform_RL_schedule(string const& model_path, vector<Task> const& tasks); // Perform the RL model decisions in the GPU

#endif // RL_SCHEDULER_H