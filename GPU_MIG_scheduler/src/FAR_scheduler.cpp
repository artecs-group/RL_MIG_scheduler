#include "FAR_scheduler.h"
#include "GPU_info.h"
#include "logging.h"
#include "utils.h"
#include <vector>
#include <cfloat>
#include <algorithm>
#include <queue>
#include <functional>
#include <thread>
#include <sys/time.h>

// Create tree structure for the algorithm
static shared_ptr<TreeNode> create_repartition_tree();
// Phase 1 of FAR's algorithm
static vector<Allocation> get_allocations_family(vector<Task> & tasks);
// Phase 2 of FAR's algorithm
static shared_ptr<TreeNode> repartitioning_schedule(Allocation const& allocation);
// Phase 3 of FAR's algorithm: refinement of the tree
static void refinement(shared_ptr<TreeNode> & root);

// Auxiliary private methods for refinement
static void set_slices_end(shared_ptr<TreeNode> &tree);
static void alternative_instance(unsigned int size, shared_ptr<TreeNode>const& node, double& end, shared_ptr<TreeNode>& alt);
static int search_move_critical_task(unsigned int size, double makespan, double alt_end, vector<Task*> const& tasks);
static void update_times_tree(shared_ptr<TreeNode>& root);

// Recursive execution of the task in the tree
static void recursive_tree_execution(shared_ptr<TreeNode> node, nvmlDevice_t device, int init_time);


TreeNode::TreeNode(vector<shared_ptr<Slice>> const& slices, weak_ptr<TreeNode> parent) : parent(parent) {
    if (slices.empty()){
        LOG_ERROR("Empty slices in the tree node");
        exit(1);
    }
    start = slices[0]->number;
    size = slices.size();
    this->slices = slices;
    tasks = {};
    end_times = {};
    end = 0;
}

static int min_area_size(Task const& task, int min_instance_size = -1){
    int min_size = -1;
    double min_area = DBL_MAX;
    for (auto const& [size, time]: task.exec_times){
        if (size <= min_instance_size) continue;
        if (size * time < min_area){
            min_area = size * time;
            min_size = size;
        }
    }
    return min_size;
}

static Allocation copy(Allocation const& alloc){
    Allocation alloc_copy;
    for (auto const& [size, tasks]: alloc){
        alloc_copy[size] = tasks;
    }
    return alloc_copy;
}

static void search_longest_task(Allocation const& allocation, Task* & longest_task, int & instance_size){
    double longest_time = 0;
    for (auto const& [size, tasks]: allocation){
        for (auto const& task: tasks){
            if (task->exec_times.at(size) > longest_time){
                longest_time = task->exec_times.at(size);
                longest_task = task;
                instance_size = size;
            }
        }
    }
}


TreeNode FAR_schedule_tasks(vector<Task> & tasks){
    vector<Allocation> allocations = get_allocations_family(tasks);
    double min_makespan = DBL_MAX;
    shared_ptr<TreeNode> best_tree;
    for (auto const& allocation: allocations){
        shared_ptr<TreeNode> tree = repartitioning_schedule(allocation);
        set_slices_end(tree);
        //(*tree).show_tree();
        double makespan = tree->get_makespan();
        //cout << "Tree makespan: " << makespan << 's' << endl;
        if (makespan < min_makespan){
            min_makespan = makespan;
            best_tree = tree;
        }
    }
    refinement(best_tree);
    return *best_tree;
}

static vector<Allocation> get_allocations_family(vector<Task> & tasks){
    vector<Allocation> allocations;
    
    // Get empty first allocation
    Allocation first_allocation;
    for (int size: global_GPU_info->valid_instance_sizes){
        first_allocation[size] = {};
    }
    // Put each task with its best size in the first allocation
    for (auto & task: tasks){
        int best_size = min_area_size(task);
        first_allocation[best_size].insert(&task);
    }
    allocations.push_back(first_allocation);

    while(true){
        // Copy the last allocation and search the longest task
        Allocation next_allocation = copy(allocations.back());
        Task* longest_task;
        int instance_size;
        search_longest_task(next_allocation, longest_task, instance_size);

        // If instance size for the longest task is all the GPU, the family is complete (break)
        if (instance_size == global_GPU_info->num_slices) break;

        // Remove the longest task from the allocation
        next_allocation[instance_size].erase(longest_task);

        // Put the longest task in the next best size
        int next_best_size = min_area_size(*longest_task, instance_size);
        next_allocation[next_best_size].insert(longest_task);

        // Add the new allocation to the family
        allocations.push_back(next_allocation);
    }

    return allocations;
}


static shared_ptr<TreeNode> repartitioning_schedule(Allocation const& allocation){
    // Create a map with the tasks decreasingly ordered by time
    unordered_map<unsigned int, vector<Task*>> tasks_by_size;
    for (auto const& [size, tasks]: allocation){
        vector<Task*> tasks_vector(tasks.begin(), tasks.end());
        sort(tasks_vector.begin(), tasks_vector.end(), [size](Task* a, Task* b){
            return a->exec_times.at(size) < b->exec_times.at(size);
        });
        tasks_by_size[size] = move(tasks_vector);
    }
    // Get the root of the repartitioning tree
    shared_ptr<TreeNode> root = create_repartition_tree();

    // Min-heap of TreeNode pointers opened
    auto compare_nodes = [](shared_ptr<TreeNode> a, shared_ptr<TreeNode> b){
        return a->end > b->end;
    };
    priority_queue<shared_ptr<TreeNode>, vector<shared_ptr<TreeNode>>, decltype(compare_nodes)> heap(compare_nodes);

    // Put the root in the heap
    heap.push(root);

    // For sequential reconfiguration
    double reconfig_end = 0;

    // While there are nodes opened in the heap
    while(!heap.empty()){
        // Get the node with the smallest end time
        auto node = heap.top();
        heap.pop();

        // If there are unscheduled task assigned to the node instance size
        if(tasks_by_size.count(node->size) && !tasks_by_size[node->size].empty()){

            // Get the task with the longest execution time
            Task* task = tasks_by_size[node->size].back();
            tasks_by_size[node->size].pop_back();
            if(tasks_by_size[node->size].empty()){
                tasks_by_size.erase(node->size);
            }

            // If it's the first task, create the instance
            if(node->tasks.empty()){
                reconfig_end = max(reconfig_end, node->end);
                reconfig_end += global_GPU_info->times_create[node->size];
                node->end = reconfig_end;
            }
            // Assign the task to the node
            node->tasks.push_back(task);
            node->end += task->exec_times.at(node->size);
            // Save the end time of the task for logging purposes
            node->end_times.push_back(node->end);

            // Return the node to the heap
            heap.push(node);
        }
        // If there are unscheduled tasks
        else if (!tasks_by_size.empty()){
            // Give time to destroy
            if (!node->tasks.empty()){
                reconfig_end = max(reconfig_end, node->end);
                reconfig_end += global_GPU_info->times_destroy[node->size];
            }
            // Create the children instances
            for (auto child: node->children){
                child->end = node->end;
                heap.push(child);
            }
        }
    }
    return root;
}



// Definition of the repartition tree for the corresponding GPU
static shared_ptr<TreeNode> create_repartition_tree(){
    // Create the slices
    vector<shared_ptr<Slice>> slices;
    for (unsigned int i = 0; i < global_GPU_info->num_slices; i++){
        slices.push_back(make_shared<Slice>(Slice{i}));
    }
    if (global_GPU_info->name == "A30"){
        

        auto root = make_shared<TreeNode>(slices);

        auto node_0_2 = make_shared<TreeNode>(vector<shared_ptr<Slice>>(slices.begin(), slices.begin()+2), root);
        auto node_2_2 = make_shared<TreeNode>(vector<shared_ptr<Slice>>(slices.begin()+2, slices.begin()+4), root);

        auto node_0_1 = make_shared<TreeNode>(vector<shared_ptr<Slice>>(slices.begin(), slices.begin()+1), node_0_2);
        auto node_1_1 = make_shared<TreeNode>(vector<shared_ptr<Slice>>(slices.begin()+1, slices.begin()+2), node_0_2);

        auto node_2_1 = make_shared<TreeNode>(vector<shared_ptr<Slice>>(slices.begin()+2, slices.begin()+3), node_2_2);
        auto node_3_1 = make_shared<TreeNode>(vector<shared_ptr<Slice>>(slices.begin()+3, slices.begin()+4), node_2_2);

        root->children = {node_0_2, node_2_2};

        node_0_2->children = {node_0_1, node_1_1};
        node_2_2->children = {node_2_1, node_3_1};

        node_0_1->children = {};
        node_1_1->children = {};
        node_2_1->children = {};
        node_3_1->children = {};
        
        return root;
    } else if (global_GPU_info->name == "A100/H100"){
        auto root = make_shared<TreeNode>(slices);

        auto node_0_4 = make_shared<TreeNode>(vector<shared_ptr<Slice>>(slices.begin(), slices.begin()+4), root);
        auto node_4_3 = make_shared<TreeNode>(vector<shared_ptr<Slice>>(slices.begin()+4, slices.begin()+7), root);

        auto node_0_3 = make_shared<TreeNode>(vector<shared_ptr<Slice>>(slices.begin(), slices.begin()+3), node_0_4);

        auto node_0_2 = make_shared<TreeNode>(vector<shared_ptr<Slice>>(slices.begin(), slices.begin()+2), node_0_3);
        auto node_2_2 = make_shared<TreeNode>(vector<shared_ptr<Slice>>(slices.begin()+2, slices.begin()+4), node_0_3);

        auto node_0_1 = make_shared<TreeNode>(vector<shared_ptr<Slice>>(slices.begin(), slices.begin()+1), node_0_2);
        auto node_1_1 = make_shared<TreeNode>(vector<shared_ptr<Slice>>(slices.begin()+1, slices.begin()+2), node_0_2);

        auto node_2_1 = make_shared<TreeNode>(vector<shared_ptr<Slice>>(slices.begin()+2, slices.begin()+3), node_2_2);
        auto node_3_1 = make_shared<TreeNode>(vector<shared_ptr<Slice>>(slices.begin()+3, slices.begin()+4), node_2_2);

        auto node_4_2 = make_shared<TreeNode>(vector<shared_ptr<Slice>>(slices.begin()+4, slices.begin()+6), node_4_3);
        auto node_6_1 = make_shared<TreeNode>(vector<shared_ptr<Slice>>(slices.begin()+6, slices.begin()+7), node_4_3);

        auto node_4_1 = make_shared<TreeNode>(vector<shared_ptr<Slice>>(slices.begin()+4, slices.begin()+5), node_4_2);
        auto node_5_1 = make_shared<TreeNode>(vector<shared_ptr<Slice>>(slices.begin()+5, slices.begin()+6), node_4_2);

        root->children = {node_0_4, node_4_3};

        node_0_4->children = {node_0_3};
        node_4_3->children = {node_4_2, node_6_1};

        node_0_3->children = {node_0_2, node_2_2};
        node_0_2->children = {node_0_1, node_1_1};
        node_2_2->children = {node_2_1, node_3_1};

        node_4_2->children = {node_4_1, node_5_1};
        return root;
    }
    else{
        LOG_ERROR("GPU model unknown");
        exit(1);
    }
}

static void refinement(shared_ptr<TreeNode> & root){
    queue<shared_ptr<TreeNode>> nodes; // Queue of open instances nodes
    bool stop = false; // Stop condition
    while(!stop){
        // Get the makespan of the tree
        double makespan = root->get_makespan();
        // Push the critical leaf nodes (reaching makespan) to the queue
        function<void(shared_ptr<TreeNode> const&)> push_leafs = [&nodes, makespan, &push_leafs](shared_ptr<TreeNode> const& node){
            if (node->children.empty()){
                if (node->end == makespan) nodes.push(node);
            }
            for (auto const& child: node->children){
                push_leafs(child);
            }
        };
        push_leafs(root);
        unordered_set<shared_ptr<TreeNode>> visited; // Set of visited nodes

        // While there are nodes in the queue
        while(!nodes.empty()){
            // Pop the first node and mark as visited
            auto instance = nodes.front();
            nodes.pop();
            visited.insert(instance);
            // If the instance is the root, stop
            if (instance == root){
                stop = true;
                break;
            }
            // Search alternative instance with the same size and minimum end time of slice
            double alt_end = instance->get_makespan();
            shared_ptr<TreeNode> alt = nullptr;
            alternative_instance(instance->size, root, alt_end, alt);

            if(alt == nullptr){// If there is no alternative instance, continue with the parent if possible
                if (visited.find(instance->parent.lock()) == visited.end())  nodes.push(instance->parent.lock());    
                continue;
            }

            // Search a task for moving in the current instance
            int task_index = search_move_critical_task(instance->size, instance->get_makespan(), alt_end, instance->tasks);

            if (task_index != -1){ // If there is a task to move
                // Insert the task sorted by time in the alternative instance
                Task* task = instance->tasks[task_index];
                auto it = lower_bound(alt->tasks.begin(), alt->tasks.end(), task, [instance](Task* a, Task* b){
                    return a->exec_times.at(instance->size) > b->exec_times.at(instance->size);
                });
                // Remove the task from the current instance vector
                instance->tasks.erase(instance->tasks.begin() + task_index);

                // Update the end time of the slices of instance and alternative
                for (auto const& slice: instance->slices){
                    slice->end_time -= task->exec_times.at(instance->size);
                }
                for (auto const& slice: alt->slices){
                    slice->end_time += task->exec_times.at(instance->size);
                }
            }
            else if (visited.find(instance->parent.lock()) == visited.end()){ // If the parent is not visited, push it to the queue
                nodes.push(instance->parent.lock());    
            }
        }
    }
    update_times_tree(root); // Update the end times of the tree
}

void TreeNode::show_tree() const{
    cout << "======================================" << endl;
    cout << "Repartitioning tree:" << endl;

    function<void(const TreeNode &, int)> show_node = [&](const TreeNode & node, int level){
        for (int i = 0; i < level; i++){
            cout << "--";
        }
        cout << "Node(start=" << node.start << ", size=" << node.size << ", tasks=[";
        int num_tasks = (node.tasks).size();
        for (int i = 0; i < num_tasks - 1; i++){
            cout << (node.tasks[i])->name << ", ";
        }
        if (!node.tasks.empty()) cout << (node.tasks.back())->name;
        cout << "], ends=[";
        int num_end_times = (node.end_times).size();
        for (int i = 0; i < num_end_times - 1; i++){
            cout << node.end_times[i] << ", ";
        }
        if (!node.end_times.empty()) cout << node.end_times.back();
        cout << "])" << endl;
        for (auto const& child: node.children){
            show_node(*child, level + 1);
        }
    };

    show_node(*this, 0);
}

double TreeNode::get_makespan() const{
    double makespan = 0;
    for (auto const& slice : this->slices){
        makespan = max(makespan, slice->end_time);
    }
    return makespan;
}


static void recursive_tree_execution(TreeNode const& node, nvmlDevice_t device, timeval const& init_time){
    // Create the instance, execute the tasks and destroy it if there are tasks in this node
    if (!node.tasks.empty()){
        struct timeval curr_time;
        Instance instance = create_instance(device, node.start, node.size);
        for (auto const& task: node.tasks){
            // Measure the init task execution time
            gettimeofday(&curr_time, NULL);
            double curr_time_s = (curr_time.tv_sec - init_time.tv_sec) + (curr_time.tv_usec - init_time.tv_usec) / 1000000.0;
            LOG_INFO("Start task " + task->name + " at " + to_string(curr_time_s) + "s");
            // Execute the task
            task->execute(instance);
            gettimeofday(&curr_time, NULL);
            curr_time_s = (curr_time.tv_sec - init_time.tv_sec) + (curr_time.tv_usec - init_time.tv_usec) / 1000000.0;
            LOG_INFO("End task " + task->name + " at " + to_string(curr_time_s) + "s");
        }
        destroy_instance(instance);
    }
    // Parallel execution of tasks in the children nodes
    vector<thread> children_threads;
    for (auto const& child: node.children){
        children_threads.emplace_back([&child, device, &init_time]{
            recursive_tree_execution(*child, device, init_time);
        });
    }
    // Wait for the children threads to finish
    for (auto & thread: children_threads){
        thread.join();
    }
}

void TreeNode::execute_tasks(nvmlDevice_t device) const{
    struct timeval init_time;
    gettimeofday(&init_time, NULL);
    recursive_tree_execution(*this, device, init_time);
}


static void set_slices_end(shared_ptr<TreeNode> &tree){
    function<void(shared_ptr<TreeNode> &, double)> tree_traverse_slices_end = [&tree_traverse_slices_end](shared_ptr<TreeNode> &node, double end){
       double new_end = max(end, node->end);
        if (node->children.empty()){
            for (auto const& slice: node->slices){
                slice->end_time = new_end;
            }
            return;
        }
        for (auto & child: node->children){
            tree_traverse_slices_end(child, new_end);
        }
    };

    tree_traverse_slices_end(tree, 0.0);
}

static void alternative_instance(unsigned int size, shared_ptr<TreeNode>const& node, double& end, shared_ptr<TreeNode>& alt){
    if (node->size != size){
        for(auto const& child: node->children){
            alternative_instance(size, child, end, alt);
        }
        return;
    }
    // If the sizes are the same, set as alternative if the max end time is less
    double min_time = node->get_makespan();
    if (min_time < end){
        end = min_time;
        alt = node;
    }
}

static int search_move_critical_task(unsigned int size, double makespan, double alt_end, vector<Task*> const& tasks){
    double upper_bound = makespan - alt_end; // Upper bound to guarantee improvement
    double target_time = upper_bound / 2; // Target time to select the best task

    // Binary search to find the task with the closest time to the target time lower than the upper bound
    int task_index = -1;
    if (tasks.empty()) return task_index;

    unsigned int left = 0, right = tasks.size() - 1;
    while(left <= right){
        unsigned int mid = (left + right) / 2;
        if (tasks[mid]->exec_times.at(size) >= target_time){  // The task list is sorted in descending order
            left = mid + 1;
        } else {
            right = mid - 1; 
        }
        bool better_solution = tasks[mid]->exec_times.at(size) < upper_bound - 0.0001;
        bool closet_to_target = task_index == -1 || (abs(tasks[mid]->exec_times.at(size) - target_time) < abs(tasks[task_index]->exec_times.at(size) - target_time));
        if (better_solution && closet_to_target){
            task_index = mid;
        }
    }
    return task_index;
}

static void update_times_tree(shared_ptr<TreeNode>& root){
    double reconfig_end = 0;
    // Min-heap of TreeNode pointers opened
    auto compare_nodes = [](shared_ptr<TreeNode> a, shared_ptr<TreeNode> b){
        return a->end > b->end;
    };
    priority_queue<shared_ptr<TreeNode>, vector<shared_ptr<TreeNode>>, decltype(compare_nodes)> heap(compare_nodes);
    while(!heap.empty()){
        auto node = heap.top();
        heap.pop();
        node->end_times.clear();
        if (!node->tasks.empty()){
            reconfig_end = max(reconfig_end, node->end);
            reconfig_end += global_GPU_info->times_create[node->size];
            node->end = reconfig_end;
            for (auto task: node->tasks){
                node->end += task->exec_times.at(node->size);
                node->end_times.push_back(node->end);
            }
            reconfig_end += global_GPU_info->times_destroy[node->size];
        }
        for (auto const& child: node->children){
            child->end = node->end;
            heap.push(child);
        }
    }
}

void perform_scheduling(vector<Task> & tasks, nvmlDevice_t device){
    // Get the allocations family
    TreeNode tree_schedule = FAR_schedule_tasks(tasks);

    // Show final schedule
    LOG_INFO("Final schedule:");
    tree_schedule.show_tree();
    cout << "Tree makespan: " << tree_schedule.get_makespan() << 's' << endl;

    DEBUG_PAUSE("Start execution of the schedule."); // Pause in debug mode with info
    
    // Execute the tasks in the GPU following the tree schedule
    tree_schedule.execute_tasks(device);
}