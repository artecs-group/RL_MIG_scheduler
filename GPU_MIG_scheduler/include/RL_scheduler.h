#ifndef RL_SCHEDULER_H
#define RL_SCHEDULER_H

#include "tasks.h"
using namespace std;

// 3 possible kinds of actions
struct Advance{};
struct Reconfig{
    vector<unsigned int> instance_sizes; // Description of new configuration
}
struct Execute{
    unsigned int num_task; // Number of the task to execute
    unsigned int num_instance; // Number of the instance to execute the task
}
using Action = variant<Advance, Reconfig, Execute>;


vector<Action> schedule_tasks(string const& model_path, vector<Task> const& tasks); // Call the RL model to return the scheduling actions


#endif // RL_SCHEDULER_H