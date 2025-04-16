from collections import defaultdict
from pprint import pprint
import heapq
import math
from bisect import bisect_left, insort



def create_allotments_family(times, n_slices):
    # Función para calcular el trabajo de una tarea con ciertos gpcs
    def work(time_task):
        _, slices, time = time_task
        return slices * time
    allotmets_family = [[min(times_task, key=work) for times_task in times]]
    while True:
        allotment_prev = allotmets_family[-1]
        index_max_time = max(enumerate(allotment_prev), key=lambda x: x[1][2])[0]
        _, num_slices, time = allotment_prev[index_max_time]
        if num_slices == n_slices:
            break
        allotment_curr = allotment_prev.copy()
        more_slices_task = [(index, slices, time) for index, slices, time in times[index_max_time] if slices > num_slices]
        allotment_curr[index_max_time] = min(more_slices_task, key=work)
        allotmets_family.append(allotment_curr)
    # print("\n\nAllotments family\n\n")
    # pprint(allotmets_family)
    return allotmets_family

class Task:
    def __init__(self, first_slice, slices, slices_used, start_time, time, index=None):
        self.first_slice = first_slice
        self.slices = slices
        self.slices_used = slices_used
        self.start_time = start_time
        self.time = time
        if index != None:
            self.index = index
        
    def __lt__(self, other):
        return self.start_time + self.time < other.start_time + other.time
    
    def __repr__(self):
        return f'\nTask(first_slice={self.first_slice},\n\tslices={self.slices},\n\tslices_used={self.slices_used},\n\tstart_time={self.start_time},\n\ttime={self.time})'

creation_times = {"A30":{1:0.11, 2:0.12, 4:0.13},
                  "A100":{1:0.16,2:0.17,3:0.20,4:0.21,7:0.24},
                  "H100":{1:0.16,2:0.21,3:0.33,4:0.38,7:0.42}}

destroy_times = {"A30":{1:0.10, 2:0.10, 4:0.10},
                  "A100":{1:0.20,2:0.20,3:0.21,4:0.21,7:0.22},
                  "H100":{1:0.21,2:0.23,3:0.25,4:0.26,7:0.26}}

class TaskTree:
    def __init__(self, index, start, time):
        self.index = index
        self.start = start
        self.time = time
    def __lt__(self, other):
        self.time < other.time
    def __repr__(self):
        return f"Task(i={self.index}, start={self.start}, time={self.time})"

class Slice:
    def __init__(self, num_slice):
        self.num_slice = num_slice
        self.end = 0
        self.start = float("inf")

    def __eq__(self, other):
        self.num_slice == other.num_slice
    
    def __repr__(self):
        return f"(s: {self.num_slice}, end: {self.end}, start: {self.start})"

class InstanceTree:

    def __init__(self, slices, device, parent=None, all_slices=None):
        if not all_slices:
            self.all_slices = [Slice(num_slice) for num_slice in slices]
        else:
            self.all_slices = all_slices
        self.slices = [self.all_slices[num_slice] for num_slice in slices]
        self.size = len(slices)
        self.tasks = []
        self.end = 0
        self.parent = parent
        #print(self.size)
        if self.size == 1:
            self.children = []
        else:
            if (device == "A100" or device == "H100") and self.size == 4:
                self.children = [InstanceTree([0,1,2], device, parent=self, all_slices=self.all_slices)]
            elif (device == "A100" or device == "H100") and self.slices == self.all_slices[:3]:
                self.children = [InstanceTree([0,1], device, parent=self, all_slices=self.all_slices), InstanceTree([2,3], device, parent=self, all_slices=self.all_slices)]
            else:
                half = math.ceil(self.size / 2)
                left = slices[0:half]
                right = slices[half:]
                self.children = [InstanceTree(left, device, parent=self, all_slices=self.all_slices), InstanceTree(right, device, parent=self, all_slices=self.all_slices)]

    def update_slices_end(self):
        if self.tasks != []:
            last_task = self.tasks[-1]
            for my_slice in self.slices:
                my_slice.end = max(my_slice.end, last_task.start + last_task.time)
        if self.children != []:
            for child in self.children:
                child.update_slices_end()

    def __eq__(self, other):
        return self.slices == other.slices
    
    def __lt__(self, other):
        return self.end < other.end
    
    def __repr__(self):
        rep = f"slices {self.slices}: "
        for task in self.tasks:
            rep += str(task)
            rep += ' '
        rep += '\n'
        # for child in self.children:
        #     rep += str(child)
        return rep


def tasks_scheduling_tree(num_slices, allotment, device):
    # Agrupo por número de slices
    allotment_by_slices = defaultdict(list)
    for index, slices, time in allotment:
        allotment_by_slices[slices].append((time, index))
    
    # Ordeno de mayor a menor en cada grupo
    allotment_by_slices = {slices: sorted(task, reverse=True) for slices, task in allotment_by_slices.items()}
    reconfig_end = 0
    
    # Introduzco la raiz
    tree = InstanceTree(list(range(0,num_slices)), device)
    pq = []
    heapq.heappush(pq, tree)
    while pq:
        current_instance = heapq.heappop(pq)
        if current_instance.size in allotment_by_slices:
            if current_instance.tasks == []:
                reconfig_end = max(reconfig_end, current_instance.end)
                reconfig_end += creation_times[device][current_instance.size]
                current_instance.end = reconfig_end
            next_task_time, next_index = allotment_by_slices[current_instance.size].pop(0)
            if allotment_by_slices[current_instance.size] == []:
                allotment_by_slices.pop(current_instance.size)
            task = TaskTree(index = next_index, start=current_instance.end, time=next_task_time)
            current_instance.tasks.append(task)
            current_instance.end += task.time
            heapq.heappush(pq, current_instance)
        elif allotment_by_slices != {}:
            if current_instance.tasks != []:
                reconfig_end = max(reconfig_end, current_instance.end)
                reconfig_end += destroy_times[device][current_instance.size]
            #pprint(allotment_by_slices)
            #print(current_instance)
            for child in current_instance.children:
                child.end = current_instance.end
                heapq.heappush(pq, child)
    tree.update_slices_end()
    return tree

def give_makespan(scheduling):
    return max(task.start_time + task.time for task in scheduling)

def give_makespan_tree(tree):
    maxi = 0
    for task in tree.tasks:
        maxi = max(maxi, task.start+task.time)
    for child in tree.children:
        maxi = max(maxi, give_makespan_tree(child))
    return maxi

def leaf_nodes(tree):
    if tree.children == []:
        return [tree]
    leafs = []
    for child in tree.children:
        leafs += leaf_nodes(child)
    return leafs

def lower_bound_makespan_opt(allotmets_family, n_slices):
    allotment_0 = allotmets_family[0]
    return sum(slices*time for _, slices, time in allotment_0) / n_slices

def moldable_scheduler_tree(n_slices, allotmets_family, device):
    scheduling = None
    best_makespan = float("inf")
    for allotment in allotmets_family:
        tree = tasks_scheduling_tree(n_slices, allotment, device)
        if (give_makespan_tree(tree) < best_makespan):
            best_makespan = give_makespan_tree(tree)
            scheduling = tree
    
    return scheduling

def instances_by_size(tree, map_instance_sizes):
    if tree.size not in map_instance_sizes:
        map_instance_sizes[tree.size] = [tree]
    else:
        map_instance_sizes[tree.size].append(tree)
    if tree.children != []:
        for child in tree.children:
           instances_by_size(child, map_instance_sizes) 

def alternative_instance(instances):
    alt_instance = None
    minimum_end = float("inf")
    for instance in instances:
        last_slice_finish = max([my_slice.end for my_slice in instance.slices])
        if last_slice_finish < minimum_end:
            minimum_end = last_slice_finish
            alt_instance = instance
    return alt_instance, minimum_end


def alternative_instance_concat(instances, tree1):
    alt_instance = None
    maximum_idle = -1
    for instance in instances:
        slice_min_idle = min([my_slice.start-tree1.slices[my_slice.num_slice].end for my_slice in instance.slices])
        if slice_min_idle > maximum_idle:
            maximum_idle = slice_min_idle
            alt_instance = instance
    return alt_instance, maximum_idle

def bin_search_time(tasks, upper_bound, close_time):
    pos = None
    left, right = 0, len(tasks) - 1
    while left <= right:
        mid = (left + right) // 2
        if tasks[mid].time >= close_time:
            left = mid + 1
        else:
            right = mid - 1
        if tasks[mid].time < upper_bound - 0.00001 and (not pos or (tasks[mid].time - close_time) < (abs(tasks[pos].time - close_time))):
            pos = mid
    return pos


def insert_decreasing_ordered(tasks, move_task):
    left, right = 0, len(tasks) - 1
    while left <= right:
        mid = (left + right) // 2
        if tasks[mid].time < move_task.time:
            right = mid - 1
        else:
            left = mid + 1
    
    tasks.insert(left, move_task)

def task_for_swap(i_tasks, alt_tasks, margin):
    min_dist = float("inf")
    i_opt, alt_opt = None, None
    i_index, alt_index = 0, 0
    while i_index < len(i_tasks) and alt_index < len(alt_tasks):
        dif = i_tasks[i_index].time - alt_tasks[alt_index].time
        if dif <= 0:
            alt_index += 1
        elif dif >= margin:
            i_index += 1
        else:
            if abs(dif - margin/2) < min_dist:
                min_dist = abs(dif -  margin/2)
                i_opt, alt_opt = i_index, alt_index
            if dif < margin/2:
                alt_index += 1
            else:
                i_index += 1
    return i_opt, alt_opt

def update_slice_end(leafs, instance, alt_instance, move_task):
    for my_slice in instance.slices:
        my_slice.end -= move_task.time
    if len(instance.slices) == 3 and instance.slices[0].num_slice == 0:
        leafs[3].slices[0].end -= move_task.time
    for my_slice in alt_instance.slices:
        my_slice.end += move_task.time
    if len(alt_instance.slices) == 3 and alt_instance.slices[0].num_slice == 0:
        leafs[3].slices[0].end += move_task.time

def _nodes_tree(tree):
    nodes = [tree]
    for child in tree.children:
        nodes += _nodes_tree(child)
    return nodes


def update_task_times(tree, device) -> None:
    for i, inst in enumerate(_nodes_tree(tree)):
        inst.index = i
    indexes = {inst.index: 0 for inst in _nodes_tree(tree)}
    tree.end = 0
    pq = []
    heapq.heappush(pq, tree)
    reconfig_end = 0
    while pq != []:
        instance = heapq.heappop(pq)
        if instance.tasks == []:
            for child in instance.children:
                child.end = instance.end
                heapq.heappush(pq, child)
            continue
        if indexes[instance.index] == 0:
            reconfig_end = max(reconfig_end, instance.end)
            reconfig_end += creation_times[device][instance.size]
            instance.end = reconfig_end
        if indexes[instance.index] < len(instance.tasks):
            task = instance.tasks[indexes[instance.index]]
            indexes[instance.index] += 1
            task.start = instance.end
            instance.end += task.time
            heapq.heappush(pq, instance)
        else:
            reconfig_end = max(reconfig_end, instance.end)
            reconfig_end += destroy_times[device][instance.size]
            for child in instance.children:
                child.end = instance.end
                heapq.heappush(pq, child)


def refinement(tree, device):
    queue = []
    stop = False
    leafs = leaf_nodes(tree)
    map_instance_sizes = {}
    instances_by_size(tree, map_instance_sizes)
    makespan = give_makespan_tree(tree)
    num_iters, num_moves, num_swaps = 0, 0, 0
    while not stop and num_iters < 100:
        num_iters += 1
        for leaf in leafs:
            if leaf.slices[0].end == makespan:
                queue.append(leaf)
        while queue != []:
            instance = queue.pop(0)
            if not instance.parent:
                stop = True
                break
            # Elección de instancia alternativa
            alt_instance, minimum_end = alternative_instance(map_instance_sizes[instance.size])
            if alt_instance != instance:
                index_task = bin_search_time(instance.tasks, makespan - minimum_end, (makespan - minimum_end)/2)
                if index_task != None:
                    move_task = instance.tasks.pop(index_task)
                    num_moves += 1
                    # Sorted insertion
                    insert_decreasing_ordered(alt_instance.tasks, move_task)
                    update_slice_end(leafs, instance, alt_instance, move_task)
                else:
                    i_index, alt_index = task_for_swap(instance.tasks, alt_instance.tasks, margin = makespan - minimum_end)
                    if i_index != None and alt_index != None:
                        task_i = instance.tasks.pop(i_index)
                        task_alt = alt_instance.tasks.pop(alt_index)
                        insert_decreasing_ordered(alt_instance.tasks, task_i)
                        insert_decreasing_ordered(instance.tasks, task_alt)
                        num_swaps += 1
                        #print("Swapped task")
                        update_slice_end(leafs, instance, alt_instance, task_i)
                        update_slice_end(leafs, alt_instance, instance, task_alt)
                    elif queue == [] or queue[-1] != instance.parent:
                        queue.append(instance.parent)
                    
            elif queue == [] or queue[-1] != instance.parent:
                queue.append(instance.parent)
        # Update makespan
        makespan = 0
        for my_slice in tree.slices:
            makespan = max(makespan, my_slice.end)
    update_task_times(tree, device)
    return num_iters, num_moves, num_swaps
        


def _reverse_schedule(instance, makespan):
    for task in instance.tasks:
        task.start = makespan - (task.start + task.time)

    for child in instance.children:
        _reverse_schedule(child, makespan)

def _min_start_time(tree):
    min_instance_task_start = float("inf")
    for task in tree.tasks:
        min_instance_task_start = min(min_instance_task_start, task.start)

    for child in tree.children:
        min_instance_task_start = min(min_instance_task_start, _min_start_time(child))
    return min_instance_task_start

def _update_end_time(tree):
    for s in tree.slices:
        s.end = 0
    tree.update_slices_end()

def _concat_schedule_in_point(instance, move):

    for task in instance.tasks:
        task.start += move

    for child in instance.children:
        _concat_schedule_in_point(child, move)

def consecutive_concat(tree1, tree2):
    makespan1 = give_makespan_tree(tree1)
    _concat_schedule_in_point(tree2, makespan1)

def _concat_point(tree1, tree2):
    best_move_overlap = None
    max_makespan_overlap = -1
    for i in range(len(tree1.slices)):
        if tree1.slices[i].end + tree2.slices[i].end > max_makespan_overlap:
            max_makespan_overlap = tree1.slices[i].end + tree2.slices[i].end
            best_move_overlap = tree1.slices[i].end - tree2.slices[i].start

    return best_move_overlap

def _clear_start(tree):
    for s in tree.slices:
        s.start = float("inf")

def set_slices_start(instance):

    for task in instance.tasks:
        for s in instance.slices:
            if task.start < s.start:
                s.start = task.start
    
    for child in instance.children:
        set_slices_start(child)

def _draw_concat_trees(n_slices, trees):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    plt.close()
    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots()
    max_makespan = give_makespan_tree(trees[-1])
    for tree in trees:
        cola = [tree]
        plt.xlim((-0.2, n_slices+0.2))
        plt.ylim((0, max_makespan+0.5))
        #line = plt.axhline(y=lb_makespane_opt, color='red', label = "baseline", linewidth=2)
        while cola != []:
            instance = cola.pop(0)
            for task in instance.tasks:
                #print("Draw", task)
                # if instance.slices == tree.all_slices[:3]:
                #     instance.size = 4
                rect = patches.Rectangle((instance.slices[0].num_slice, task.start), instance.size, task.time, facecolor = colors[task.index % len(colors)], alpha = 0.55, linewidth = 1, edgecolor = 'black')
                ax.add_patch(rect)
            for child in instance.children:
                cola.append(child)
    plt.xlabel("Slices", fontsize=20)
    plt.ylabel("Time", labelpad=5, fontsize=20)
    plt.show()


def concat_schedules(tree1, tree2, reverse):
    _update_end_time(tree1)
    _update_end_time(tree2)
    makespan2 = give_makespan_tree(tree2)
    if reverse:
        _reverse_schedule(tree2, makespan2)
    _clear_start(tree2)
    set_slices_start(tree2)
    best_move_overlap = _concat_point(tree1, tree2)
    _concat_schedule_in_point(tree2, best_move_overlap)

def concat_moves_swaps(tree1, tree2, device):
    queue = []
    stop = False
    leafs = leaf_nodes(tree2)
    map_instance_sizes = {}
    instances_by_size(tree2, map_instance_sizes)
    _clear_start(tree2)
    set_slices_start(tree2)
    num_iters, num_moves, num_swaps = 0, 0, 0
    while not stop and num_iters < 20:
        num_iters += 1
        # Add critical leafs
        max_makespan_overlap = 0
        for i in range(len(tree1.slices)):
            if tree1.slices[i].end + tree2.slices[i].end > max_makespan_overlap:
                max_makespan_overlap = tree1.slices[i].end + tree2.slices[i].end
        for i in range(len(tree1.slices)):
            if tree1.slices[i].end + tree2.slices[i].end == max_makespan_overlap:
                queue.append(leafs[i])        
        while queue != []:

            instance = queue.pop(0)
            if not instance.parent:
                stop = True
                break
            # Elección de instancia alternativa
            alt_instance, max_idle = alternative_instance_concat(map_instance_sizes[instance.size], tree1)
            if alt_instance != instance:
                index_task = bin_search_time(instance.tasks, max_idle, max_idle/2)
                if index_task != None:
                    move_task = instance.tasks.pop(index_task)
                    # Sorted insertion
                    insert_decreasing_ordered(alt_instance.tasks, move_task)
                    update_task_times(tree2, device)
                    #_draw_concat_trees(7, [tree2])
                    num_moves += 1
                    concat_schedules(tree1, tree2, reverse=True)
                    _clear_start(tree2)
                    set_slices_start(tree2)
                    #_draw_concat_trees(7, [tree1, tree2])
                else:
                    i_index, alt_index = task_for_swap(instance.tasks, alt_instance.tasks, max_idle)
                    if i_index != None and alt_index != None:
                        task_i = instance.tasks.pop(i_index)
                        task_alt = alt_instance.tasks.pop(alt_index)
                        insert_decreasing_ordered(alt_instance.tasks, task_i)
                        insert_decreasing_ordered(instance.tasks, task_alt)
                        update_task_times(tree2, device)
                        #_draw_concat_trees(7, [tree2])
                        num_swaps += 1
                        concat_schedules(tree1, tree2, reverse=True)
                        _clear_start(tree2)
                        set_slices_start(tree2)
                        #_draw_concat_trees(7, [tree1, tree2])
                    elif queue == [] or queue[-1] != instance.parent:
                        queue.append(instance.parent)
                    
            elif queue == [] or queue[-1] != instance.parent:
                queue.append(instance.parent)
    return num_iters, num_moves, num_swaps

    

from itertools import permutations


def _sum_speed(tasks_times, partition):
    speed = 0
    for task_times, instance_size in zip(tasks_times, partition):
        time_1 = next((time for _, slices, time in task_times if slices == 1), None)
        time_size = next((time for _, slices, time in task_times if slices == instance_size), None)
        speed += time_1 / time_size
    return speed


def _select_partition(times, partitions):
    best_speed = 0
    best_partition = None
    best_next_times = None
    for partition in partitions:
        n_instances = len(partition)
        task_goal = times[:n_instances]
        speed = _sum_speed(task_goal, partition)
        #print(partition, best_speed_partition)
        if speed > best_speed:
            best_speed = speed
            best_partition = partition
            best_next_times = task_goal
    return best_partition, best_next_times

partitions_A30 = [[4], [2,2], [2,1,1], [1,1,2], [1,1,1,1]]
partitions_A100 = [[7], [4,3], [4,2,1], [4,1,1,1],\
                    [3,3], [3,2,1], [3,1,1,1],\
                    [2,2,3],[2,2,2,1], [2,2,1,1,1],\
                    [2,1,1,3],[2,1,1,2,1], [2,1,1,1,1,1],\
                    [1,1,2,3],[1,1,2,2,1], [1,1,2,1,1,1],\
                    [1,1,1,1,3],[1,1,1,1,2,1], [1,1,1,1,1,1,1]]

def no_dynamic_reconfig(device, times):
    scheduling = []
    partitions = partitions_A100 if device == "A100" else partitions_A30
    num_slices = 4 if device == "A30" else 7
    start_times = [0 for slice in range(num_slices)]
    while times != []:
        best_partition, best_order = _select_partition(times, partitions)
        first_slice = 0
        for task_times, instance_size in zip(best_order, best_partition):
            index, time = next(((index, time) for index, slices, time in task_times if slices == instance_size), None)
            slices = 4 if first_slice == 0 and instance_size == 3 else instance_size
            next_start_time = max(start_times[first_slice:first_slice+slices])
            scheduling.append(Task(first_slice=first_slice, slices=slices, slices_used=instance_size,\
                                   start_time=next_start_time, time=time, index=index))
            for slice in range(first_slice, first_slice+slices):
                start_times[slice] = next_start_time + time
            first_slice += slices

        times = times[len(best_partition):]
    return scheduling

def fifo_partition(times, partition):
    pq = []
    scheduling = []
    first_slice = 0
    for instance_size in partition:
            reserved_slices = instance_size
            if instance_size == 3 and partition[-1] != 3:
                reserved_slices = 4
            heapq.heappush(pq, Task(first_slice=first_slice, slices=reserved_slices, slices_used=instance_size,\
                                start_time=0, time=0))
            first_slice += reserved_slices
    for task_times in times:
        task_finish = heapq.heappop(pq)
        if task_finish.time > 0:
            scheduling.append(task_finish)
        next_index, next_time = next(((index, time) for index, slices, time in task_times if slices == task_finish.slices_used), None)
        heapq.heappush(pq, Task(first_slice=task_finish.first_slice, slices=task_finish.slices, slices_used=task_finish.slices_used,\
                            start_time=task_finish.start_time + task_finish.time, time=next_time, index=next_index))
    while pq:
        task_finish = heapq.heappop(pq)
        if task_finish.time > 0:
            scheduling.append(task_finish)
    return scheduling


def fifo_fixed(device, times):
    partitions = partitions_A100 if device == "A100" else partitions_A30
    best_scheduling = None
    best_makespan = float("inf")
    for partition in partitions:
        scheduling_partition = []
        pq = []
        first_slice = 0
        for instance_size in partition:
            reserved_slices = instance_size
            if instance_size == 3 and partition[-1] != 3:
                reserved_slices = 4
            heapq.heappush(pq, Task(first_slice=first_slice, slices=reserved_slices, slices_used=instance_size,\
                                start_time=0, time=0))
            first_slice += reserved_slices
        for task_times in times:
            task_finish = heapq.heappop(pq)
            if task_finish.time > 0:
                scheduling_partition.append(task_finish)
            next_index, next_time = next(((index, time) for index, slices, time in task_times if slices == task_finish.slices_used), None)
            heapq.heappush(pq, Task(first_slice=task_finish.first_slice, slices=task_finish.slices, slices_used=task_finish.slices_used,\
                                start_time=task_finish.start_time + task_finish.time, time=next_time, index=next_index))
        while pq:
            task_finish = heapq.heappop(pq)
            if task_finish.time > 0:
                scheduling_partition.append(task_finish)
        makespan_partition = give_makespan(scheduling_partition)
        if makespan_partition < best_makespan:
            best_makespan = makespan_partition
            best_scheduling = scheduling_partition
    return best_scheduling
        


        
    