#ifndef FAR_SCHEDULER_H
#define FAR_SCHEDULER_H

#include "tasks.h"
#include <unordered_map>
#include <unordered_set>
#include <memory>
using namespace std;

typedef unordered_map<unsigned int, unordered_set<Task*>> Allocation;

// GPU slice scheudling info
struct Slice{
    unsigned int number;
    double end_time = 0;
};

// Repartition tree structure
struct TreeNode{
    int start, size; // Slice of start and size of the instance
    vector<shared_ptr<Slice>> slices; // Slices of the instance
    vector<Task*> tasks;// Tasks to execute in this node  
    vector<shared_ptr<TreeNode>> children; // Children of this node
    vector<double> end_times; // End time of the tasks in this node
    double end; // End time of the last task in this node
    weak_ptr<TreeNode> parent; // Parent of this node (needed for the refinement)

    // Constructor
    TreeNode(vector<shared_ptr<Slice>> const& slices, weak_ptr<TreeNode> parent = weak_ptr<TreeNode>());

    void show_tree() const; // Show the complete current tree in detail
    double get_makespan() const; // Get the makespan of the tree
    void execute_tasks(nvmlDevice_t device) const; // Execute the tasks in the GPU following the tree
};

TreeNode FAR_schedule_tasks(vector<Task> & tasks); // FAR's algorithm

void perform_FAR_schedule(vector<Task> & tasks, nvmlDevice_t device); // Perform the scheduling of the tasks

#endif // FAR_SCHEDULER_H