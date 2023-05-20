import numpy as np
import threading

def my_sum(arr):
    sum_of_elems = 0
    for el in arr:
        sum_of_elems += el
    return sum_of_elems

def my_max(arr):
    max_el = np.NINF
    for el in arr:  
        if el > max_el:
            max_el = el
    return max_el

def my_min(arr):
    min_el = np.Inf
    for el in arr:  
        if el < min_el:
            min_el = el
    return min_el

def my_average(arr):
    sum_of_elems = 0
    for el in arr:
        sum_of_elems += el
    return sum_of_elems / len(arr)



def my_sum_parallel(arr, sum_of_elems, i):
    for el in arr:
        sum_of_elems[i] += el


def my_max_parallel(arr, max_el, i):
    for el in arr:  
        if el > max_el[i]:
            max_el[i] = el


def my_min_parallel(arr, min_el, i):
    for el in arr:  
        if el < min_el[i]:
            min_el[i] = el


def my_average_parallel(arr, sum_of_elems, i):
    for el in arr:
        sum_of_elems[i] += el

def threads_cpu(func, array, num_threads_cpu, dim_array, partial_array, threads):
    for i in range(num_threads_cpu):
        t = threading.Thread(target=func, args=(array[i * dim_array // num_threads_cpu: (i+1) * dim_array // num_threads_cpu ], partial_array, i,))
        threads.append(t)
        
    #start threads
    for t in threads:
        t.start()

    for t in threads:
        t.join()