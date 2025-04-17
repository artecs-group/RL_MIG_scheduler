#ifndef RL_SCHEDULER_H
#define RL_SCHEDULER_H

#include "tasks.h"
using namespace std;

void perform_RL_schedule(string const& model_path, vector<Task> const& tasks, nvmlDevice_t device); // Perform the RL model decisions in the GPU

#endif // RL_SCHEDULER_H