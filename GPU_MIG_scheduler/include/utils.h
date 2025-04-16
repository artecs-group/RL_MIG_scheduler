#ifndef UTILS_H
#define UTILS_H
#include "tasks.h"
#include <limits>
#include <ios>
#include <vector>
#include <string>
#include <nvml.h>
using namespace std;

// Define the DEBUG_MODE macro to do pauses in that execution mode
#ifdef DEBUG_MODE
    #define DEBUG_PAUSE(msg) \
        do { \
            cout << endl << (msg) << " Press Enter to continue..." << endl; \
            cin.ignore(numeric_limits<streamsize>::max(), '\n'); \
        } while(0)
#else
    #define DEBUG_PAUSE(msg) // Do nothing in other mode
#endif

vector<Task> get_tasks(const string & kernels_path);
void profile_tasks(vector<Task> & tasks, nvmlDevice_t device);
void redirect_output(Task const& task, int & stdout_backup, int & stderr_backup);
void restore_output(int stdout_backup, int stderr_backup);

#endif // UTILS_H