from plotting import draw_concat_trees, draw_rects, draw_rects_tree, plot_speedup_inputs
from algorithm import *
from inputs import generate_tasks, get_input_config, read_task_rodinia
import random
import statistics

def inputs_config_task_size_test():
    repetitions = 500
    for n in [10,15,20,25,30,35]:
        n_scale = {}
        l_percs = [[(1,50),(2,50),(3,0),(4,0),(7,0)], [(1,20),(2,20),(3,20),(4,20),(7,20)], [(1,0),(2,0),(3,0),(4,50),(7,50)]]
        for percs in l_percs:
            n_in = 0
            for slices, perc in percs:
                n_scale[slices] = perc*n // 100
                n_in +=n_scale[slices]
            possible_slices = [1,2,3,4,7]
            for _ in range(n-n_in):
                while n_scale[possible_slices[0]] == 0:
                    possible_slices = possible_slices[1:]
                n_scale[possible_slices[0]] += 1
            instance_sizes = [1,2,3,4,7]
            ratios_algorithm = []
            for _ in range(repetitions):
                times, instance_sizes = generate_tasks(instance_sizes, n_scale, device="A100")
                #plot_speedup_inputs(device="A100", times=times)
                
                n_slices = instance_sizes[-1]
                random.shuffle(times)

                allotmets_family = create_allotments_family(times, n_slices)
                lb_makespane_opt = lower_bound_makespan_opt(allotmets_family, n_slices)
                tree = moldable_scheduler_tree(n_slices, allotmets_family, "A100")
                refinement(tree, "A100")
                # scheduling_fifo_fixed = fifo_fixed(device, times)
                # scheduling_no_dynamic = no_dynamic_reconfig(device, times)
                # makespan_fifo_fixed = give_makespan(scheduling_fifo_fixed)
                # makespan_no_dynamic = give_makespan(scheduling_no_dynamic)
                # draw_rects(n_slices, scheduling_algorithm, scheduling_algorithm, scheduling_algorithm, scheduling_algorithm, scheduling_algorithm, lb_makespane_opt)
                makespan_algorithm = give_makespan_tree(tree)
                ratios_algorithm.append(makespan_algorithm/lb_makespane_opt)
            
            print(f"Alg ratio n= {n}, {percs}: {statistics.mean(ratios_algorithm):.3f}+-{statistics.stdev(ratios_algorithm):.3f}")


def compare_size_test():
    repetitions = 100
    for n in [15, 30]:
        n_scale = {}
        l_percs = [[(1,50),(2,50),(3,0),(4,0),(7,0)], [(1,20),(2,20),(3,20),(4,20),(7,20)], [(1,0),(2,0),(3,0),(4,50),(7,50)]]
        for percs in l_percs:
            n_in = 0
            for slices, perc in percs:
                n_scale[slices] = perc*n // 100
                n_in +=n_scale[slices]
            possible_slices = [1,2,3,4,7]
            for _ in range(n-n_in):
                while n_scale[possible_slices[0]] == 0:
                    possible_slices = possible_slices[1:]
                n_scale[possible_slices[0]] += 1
            instance_sizes = [1,2,3,4,7]
            device = "A100"
            for times_range in [[1,100], [90,100]]:
                ratios_fixed, ratios_fixed7s, ratios_fixed1s = [], [], []
                ratios_speed_indep = []
                for i in range(repetitions):
                    times, instance_sizes = generate_tasks(instance_sizes, n_scale, device, 50, times_range)
                    #plot_speedup_inputs(device, times)
                    
                    n_slices = instance_sizes[-1]
                    random.shuffle(times)

                    allotmets_family = create_allotments_family(times, n_slices)
                    lb_makespane_opt = lower_bound_makespan_opt(allotmets_family, n_slices)
                    _, scheduling_algorithm = moldable_scheduler(n_slices, allotmets_family)
                    scheduling_fifo_fixed = fifo_fixed(device, times)
                    scheduling_no_dynamic = no_dynamic_reconfig(device, times)
                    scheduling_7s = fifo_partition(times, [7])
                    scheduling_1s = fifo_partition(times, [1,1,1,1,1,1,1])
                    makespan_fifo_fixed = give_makespan(scheduling_fifo_fixed)
                    makespan_no_dynamic = give_makespan(scheduling_no_dynamic)
                    makespan_algorithm = give_makespan(scheduling_algorithm)
                    makespan1s = give_makespan(scheduling_1s)
                    makespan7s = give_makespan(scheduling_7s)
                    #draw_rects(n_slices, scheduling_no_dynamic, scheduling_1s, scheduling_fifo_fixed, scheduling_7s, scheduling_algorithm, lb_makespane_opt)
                    ratios_fixed7s.append(makespan7s / makespan_algorithm)
                    ratios_fixed1s.append(makespan1s / makespan_algorithm)
                    ratios_fixed.append(makespan_fifo_fixed / makespan_algorithm)
                    ratios_speed_indep.append(makespan_no_dynamic / makespan_algorithm)
                    #print(makespan_fifo_fixed / makespan_algorithm)
                print(f"Fifo fix-best n= {n}, {percs}, {times_range}: {statistics.mean(ratios_fixed):.2f}+-{statistics.stdev(ratios_fixed):.2f}")
                print(f"Fifo fix-1s n= {n}, {percs}, {times_range}: {statistics.mean(ratios_fixed1s):.2f}+-{statistics.stdev(ratios_fixed1s):.2f}")
                print(f"Fifo fix-7s n= {n}, {percs}, {times_range}: {statistics.mean(ratios_fixed7s):.2f}+-{statistics.stdev(ratios_fixed7s):.2f}")
                print(f"Fifo speed-indep n= {n}, {percs}, {times_range}: {statistics.mean(ratios_speed_indep):.2f}+-{statistics.stdev(ratios_speed_indep):.2f}")
                print()

def rodinia_kernels_test():
    times, test_names = read_task_rodinia()
    n_slices = 7
    allotmets_family = create_allotments_family(times, n_slices)
    lb_makespane_opt = lower_bound_makespan_opt(allotmets_family, n_slices)
    tree = moldable_scheduler_tree(n_slices, allotmets_family, "A100")
    draw_rects_tree(n_slices, tree, lb_makespane_opt)
    refinement(tree, "A100")
    draw_rects_tree(n_slices, tree, lb_makespane_opt)
    scheduling_no_dynamic = no_dynamic_reconfig("A100", times)
    scheduling_fifo_fixed = fifo_fixed("A100", times)
    scheduling_7s = fifo_partition(times, [7])
    scheduling_1s = fifo_partition(times, [1,1,1,1,1,1,1])
    makespan_fifo_fixed = give_makespan(scheduling_fifo_fixed)
    makespan_no_dynamic = give_makespan(scheduling_no_dynamic)
    makespan1s = give_makespan(scheduling_1s)
    makespan7s = give_makespan(scheduling_7s)
    makespan_algorithm = give_makespan_tree(tree)
    print(f"Rho Rodinia: {makespan_algorithm/ lb_makespane_opt:.3f}")
    print(f"Ratio 1s: {makespan1s/ makespan_algorithm:.3f}")
    print(f"Ratio 7s: {makespan7s/ makespan_algorithm:.3f}")
    print(f"Ratio Fix-Part-Best: {makespan_fifo_fixed/ makespan_algorithm:.3f}")
    print(f"Ratio Speed-Indep: {makespan_no_dynamic/ makespan_algorithm:.3f}")
    draw_rects(n_slices, scheduling_no_dynamic, scheduling_1s, scheduling_fifo_fixed, scheduling_7s, scheduling_7s, lb_makespane_opt, names = test_names)




def refinement_test():
    repetitions = 1000
    for n in [10,20,30]:
        n_scale = {}
        l_percs = [[(1,50),(2,50),(3,0),(4,0),(7,0)], [(1,20),(2,20),(3,20),(4,20),(7,20)], [(1,0),(2,0),(3,0),(4,50),(7,50)]]
        for percs in l_percs:
            n_in = 0
            for slices, perc in percs:
                n_scale[slices] = perc*n // 100
                n_in +=n_scale[slices]
            possible_slices = [1,2,3,4,7]
            for _ in range(n-n_in):
                while n_scale[possible_slices[0]] == 0:
                    possible_slices = possible_slices[1:]
                n_scale[possible_slices[0]] += 1
            instance_sizes = [1,2,3,4,7]
            ratios_algorithm, iters_algorithm, moves, swaps = [], [], [], []
            for _ in range(repetitions):
                times, instance_sizes = generate_tasks(instance_sizes, n_scale, device="A100", times_range=[1,100])
                #plot_speedup_inputs(device="A100", times=times)
                
                n_slices = instance_sizes[-1]
                random.shuffle(times)

                allotmets_family = create_allotments_family(times, n_slices)
                tree = moldable_scheduler_tree(n_slices, allotmets_family, "A100")
                makespan_no_refinement = give_makespan_tree(tree)
                num_iters, num_moves, num_swaps = refinement(tree, "A100")
                makespan_refinement = give_makespan_tree(tree)
                ratios_algorithm.append(makespan_no_refinement/makespan_refinement)
                iters_algorithm.append(num_iters)
                moves.append(num_moves)
                swaps.append(num_swaps)
                #print(num_iters)
            
            print(f"Alg ratio n= {n}, {percs}: {statistics.mean(ratios_algorithm):.3f}+-{statistics.stdev(ratios_algorithm):.3f}, {statistics.mean(moves):.3f}, {statistics.mean(swaps):.3f}")
            print("\n\n")

import copy


def rodinia_kernels_concat():
    times, test_names = read_task_rodinia()
    n_slices = 7
    allotmets_family = create_allotments_family(times, n_slices)
    lb_makespane_opt = lower_bound_makespan_opt(allotmets_family, n_slices)
    tree1 = moldable_scheduler_tree(n_slices, allotmets_family, "A100")
    #draw_rects_tree(n_slices, tree, lb_makespane_opt)
    refinement(tree1, "A100")
    tree2 = copy.deepcopy(tree1)
    tree_copy_1 = copy.deepcopy(tree1)
    tree_copy_2 = copy.deepcopy(tree2)

    consecutive_concat(tree_copy_1, tree_copy_2)
    makespan_consecutive = give_makespan_tree(tree_copy_2)
    draw_concat_trees(n_slices, [tree_copy_1, tree_copy_2])
    concat_schedules(tree1, tree2, reverse=True)
    makespan_reverse_not_moves_swaps = give_makespan_tree(tree2)
    draw_concat_trees(n_slices, [tree1, tree2])
    concat_moves_swaps(tree1, tree2, "A100")
    makespan_reverse_moves_swaps = give_makespan_tree(tree2)
    draw_concat_trees(n_slices, [tree1, tree2])
    #draw_rects_tree(n_slices, tree, lb_makespane_opt)
   
    print(f"Ratio reverse: {makespan_consecutive / makespan_reverse_not_moves_swaps:.3f}")
    print(f"Ratio moves/swaps: {makespan_consecutive / makespan_reverse_moves_swaps:.3f}")
    #draw_rects(n_slices, scheduling_no_dynamic, scheduling_1s, scheduling_fifo_fixed, scheduling_7s, scheduling_7s, lb_makespane_opt, names = test_names)

def concat_test():
    repetitions = 1000
    for n in [10,20,30]:
        n_scale = {}
        l_percs = [[(1,50),(2,50),(3,0),(4,0),(7,0)], [(1,20),(2,20),(3,20),(4,20),(7,20)], [(1,0),(2,0),(3,0),(4,50),(7,50)]]
        for percs in l_percs:
            n_in = 0
            for slices, perc in percs:
                n_scale[slices] = perc*n // 100
                n_in +=n_scale[slices]
            possible_slices = [1,2,3,4,7]
            for _ in range(n-n_in):
                while n_scale[possible_slices[0]] == 0:
                    possible_slices = possible_slices[1:]
                n_scale[possible_slices[0]] += 1
            instance_sizes = [1,2,3,4,7]
            ratios_reverse, ratios_moves_swaps, moves, swaps = [], [], [], []
            for i in range(repetitions):
                if i %  100 == 0:
                    print(i)
                times, instance_sizes = generate_tasks(instance_sizes, n_scale, device="A100", times_range=[90,100])
                #plot_speedup_inputs(device="A100", times=times)
                
                n_slices = instance_sizes[-1]
                random.shuffle(times)

                allotmets_family = create_allotments_family(times, n_slices)
                tree1 = moldable_scheduler_tree(n_slices, allotmets_family, "A100")
                makespan_no_refinement = give_makespan_tree(tree1)
                _, _, _ = refinement(tree1, "A100")


                allotmets_family = create_allotments_family(times, n_slices)
                tree2 = moldable_scheduler_tree(n_slices, allotmets_family, "A100")
                makespan_no_refinement = give_makespan_tree(tree2)
                _, _, _ = refinement(tree2, "A100")

                tree1_copy = copy.deepcopy(tree1)
                tree2_copy = copy.deepcopy(tree2)

                consecutive_concat(tree1_copy, tree2_copy)
                makespan_consecutive = give_makespan_tree(tree2_copy)

                concat_schedules(tree1, tree2, reverse=True)
                makespan_reverse_not_moves_swaps = give_makespan_tree(tree2)

                ratios_reverse.append(makespan_consecutive/makespan_reverse_not_moves_swaps)

                num_iters, num_moves, num_swaps = concat_moves_swaps(tree1, tree2, "A100")
                moves.append(num_moves)
                swaps.append(num_swaps)
                makespan_reverse_moves_swaps = give_makespan_tree(tree2)
                
                ratios_moves_swaps.append(makespan_consecutive/makespan_reverse_moves_swaps)
            
            print(f"Ratio reverse n= {n}, {percs}: {statistics.mean(ratios_reverse):.3f}+-{statistics.stdev(ratios_reverse):.3f}. Ratio moves/swaps: {statistics.mean(ratios_moves_swaps):.3f}+-{statistics.stdev(ratios_moves_swaps):.3f}")
            print(f"Moves/swaps n= {n}, {percs}: {statistics.mean(moves):.3f}+-{statistics.stdev(moves):.3f}. Swaps: {statistics.mean(swaps):.3f}+-{statistics.stdev(swaps):.3f}")
            print("\n\n")


def multi_batch_test():
    repetitions = 1000
    for n in [10,15, 20,15, 30, 35]:
        n_scale = {}
        l_percs = [[(1,50),(2,50),(3,0),(4,0),(7,0)], [(1,20),(2,20),(3,20),(4,20),(7,20)], [(1,0),(2,0),(3,0),(4,50),(7,50)]]
        for percs in l_percs:
            n_in = 0
            for slices, perc in percs:
                n_scale[slices] = perc*n // 100
                n_in +=n_scale[slices]
            possible_slices = [1,2,3,4,7]
            for _ in range(n-n_in):
                while n_scale[possible_slices[0]] == 0:
                    possible_slices = possible_slices[1:]
                n_scale[possible_slices[0]] += 1
            instance_sizes = [1,2,3,4,7]
            baseline, sum_consecutive_concat, sum_overlaps = 0, 0, 0

            times1, instance_sizes = generate_tasks(instance_sizes, n_scale, device="A100", times_range=[1,100])
            #plot_speedup_inputs(device="A100", times=times)
            
            n_slices = instance_sizes[-1]
            random.shuffle(times1)


            
            for i in range(repetitions):
                if i %  100 == 0:
                    print(i)

                allotmets_family = create_allotments_family(times1, n_slices)
                baseline += lower_bound_makespan_opt(allotmets_family, n_slices)
                tree1 = moldable_scheduler_tree(n_slices, allotmets_family, "A100")
                makespan_no_refinement = give_makespan_tree(tree1)
                _, _, _ = refinement(tree1, "A100")

                times2, instance_sizes = generate_tasks(instance_sizes, n_scale, device="A100", times_range=[1,100])
                    #plot_speedup_inputs(device="A100", times=times)
                    
                n_slices = instance_sizes[-1]
                random.shuffle(times2)


                allotmets_family = create_allotments_family(times2, n_slices)
                tree2 = moldable_scheduler_tree(n_slices, allotmets_family, "A100")
                makespan_no_refinement = give_makespan_tree(tree2)
                _, _, _ = refinement(tree2, "A100")

                tree1_copy = copy.deepcopy(tree1)
                tree2_copy = copy.deepcopy(tree2)

                consecutive_concat(tree1_copy, tree2_copy)
                makespan_consecutive = give_makespan_tree(tree2_copy)
                sum_consecutive_concat += makespan_consecutive


                concat_schedules(tree1, tree2, reverse=True)
                makespan_reverse_not_moves_swaps = give_makespan_tree(tree2)


                num_iters, num_moves, num_swaps = concat_moves_swaps(tree1, tree2, "A100")
                makespan_reverse_moves_swaps = give_makespan_tree(tree2)

                overlap = makespan_consecutive - makespan_reverse_moves_swaps
                sum_overlaps += overlap
                times1 = times2

            baseline += lower_bound_makespan_opt(allotmets_family, n_slices)
            multi_batch_makespan = sum_consecutive_concat-sum_overlaps
            print(f"Ratio multi-batch n= {n}, {percs}: {multi_batch_makespan/baseline}")
            print("\n\n")    


def main():
    times, instance_sizes = generate_tasks([1,2,3,4,7], {1:2,2:2,3:2,4:2,7:2}, "A100", 50)
    allotmets_family = create_allotments_family(times, 7)
    lb_makespane_opt = lower_bound_makespan_opt(allotmets_family, 7)
    tree = moldable_scheduler_tree(7, allotmets_family, "A100")
    refinement(tree, "A100")
    draw_rects_tree(7, tree, lb_makespane_opt)
    #rodinia_kernels_concat()
    #concat_test()
    #multi_batch_test()
    #refinement_test()
    #compare_size_test()
    #inputs_config_task_size_test()
    # rodinia_kernels_test()
    # repetitions, instance_sizes, n_scale, perc_membound, device = get_input_config()
    # for _ in range(repetitions):
    #     times, instance_sizes = generate_tasks(instance_sizes, n_scale, device, perc_membound)
    #     #plot_speedup_inputs(device, times)
        
    #     n_slices = instance_sizes[-1]
    #     random.shuffle(times)

    #     allotmets_family = create_allotments_family(times, n_slices)
    #     lb_makespane_opt = lower_bound_makespan_opt(allotmets_family, n_slices)
    #     tree = moldable_scheduler_tree(n_slices, allotmets_family, device)
    #     draw_rects_tree(n_slices, tree, lb_makespane_opt)
    #     makespan_no_refinement = give_makespan_tree(tree)
    #     print("Makespan no refinement:", makespan_no_refinement)
    #     refinement(tree, device)
    #     makespan_refinement = give_makespan_tree(tree)
    #     print("Makespan refinement:", makespan_refinement)
    #     draw_rects_tree(n_slices, tree, lb_makespane_opt)
    #     input("parar")

if __name__ == "__main__":
    main()