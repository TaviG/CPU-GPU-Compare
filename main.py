import numpy as np
import time
import threading
import sys
import cupy as cp
from numba import cuda, int32
from numba.typed import List
from funcs import *


N = int(sys.argv[1])
num_threads_cpu = int(sys.argv[2])
block_size = int(sys.argv[3])
array = np.zeros(N,dtype=int)

for i in range(len(array)):
    array[i] = 1

time_bef_suma = time.time()
suma = sum(array)
print(suma)
time_aft_suma = time.time()
print('Functie sum: ', time_aft_suma - time_bef_suma)
print(suma)
time_bef_maxim = time.time()
maxim = max(array)
time_aft_maxim = time.time()
print('Functie max: ',time_aft_maxim - time_bef_maxim)

time_bef_minim = time.time()
minim = min(array)
time_aft_minim = time.time()
print('Functie min: ',time_aft_minim - time_bef_minim)

time_bef_average = time.time()
average = np.average(array)
time_aft_average = time.time()
print('Functie avg: ',time_aft_average - time_bef_average)


time_bef_suma = time.time()
my_suma = my_sum(array)
time_aft_suma = time.time()
print('My sum: ', time_aft_suma - time_bef_suma)

time_bef_maxim = time.time()
my_maxim = my_max(array)
time_aft_maxim = time.time()
print('My max: ',time_aft_maxim - time_bef_maxim)

time_bef_minim = time.time()
my_minim = my_min(array)
time_aft_minim = time.time()
print('My min: ',time_aft_minim - time_bef_minim)

time_bef_average = time.time()
my_avg = my_average(array)
time_aft_average = time.time()
print('My avg: ',time_aft_average - time_bef_average)

# MultiThread CPU
threads = []
dim_array = len(array)
partial_sums = np.zeros(num_threads_cpu, dtype=int)
partial_maxs = np.ones(num_threads_cpu, dtype=int) * np.NINF
partial_mins = np.ones(num_threads_cpu, dtype=int) * np.Inf
partial_avgs = np.zeros(num_threads_cpu, dtype=int)

time_bef_suma = time.time()

threads_cpu(my_sum_parallel, array, num_threads_cpu, dim_array, partial_sums, threads)

sum = my_sum(partial_sums)
time_aft_suma = time.time()
print('My CPU parallel sum: ', time_aft_suma - time_bef_suma)
print(sum)


# Maxim

threads = []

time_bef_maxim = time.time()

threads_cpu(my_max_parallel, array, num_threads_cpu, dim_array, partial_maxs, threads)
maxim = my_max(partial_maxs)

time_aft_maxim = time.time()
print('My CPU parallel max: ',time_aft_maxim - time_bef_maxim)

# Minim
threads = []

time_bef_minim = time.time()

threads_cpu(my_min_parallel, array, num_threads_cpu, dim_array, partial_mins, threads)
minim = my_min(partial_mins)

time_aft_minim = time.time()
print('My CPU parallel min: ',time_aft_minim - time_bef_minim)

# Average
threads = []

time_bef_average = time.time()
threads_cpu(my_average_parallel, array, num_threads_cpu, dim_array, partial_avgs, threads)
avg = my_average(partial_avgs)

time_aft_average = time.time()
print('My CPU parallel avg: ',time_aft_average - time_bef_average)



# GPU

array_gpu = cp.array(array)

time_bef_suma = time.time()
suma = cp.sum(array_gpu)
time_aft_suma = time.time()
print('Function GPU parallel sum: ', time_aft_suma - time_bef_suma)

time_bef_maxim = time.time()
maxim = cp.amax(array_gpu)
time_aft_maxim = time.time()
print('Function GPU parallel max: ',time_aft_maxim - time_bef_maxim)

time_bef_minim = time.time()
minim = cp.amin(array_gpu)
time_aft_minim = time.time()
print('Function GPU parallel min: ',time_aft_minim - time_bef_minim)

time_bef_average = time.time()
average = cp.average(array_gpu)
time_aft_average = time.time()
print('Function GPU parallel avg: ',time_aft_average - time_bef_average)


# GPU Numba

@cuda.jit(device=True)
def lock(mutex):
    while cuda.atomic.compare_and_swap(mutex, 0, 1) != 0:
        pass
    cuda.threadfence()


@cuda.jit(device=True)
def unlock(mutex):
    cuda.threadfence()
    cuda.atomic.exch(mutex, 0, 0)

@cuda.jit
def my_sum_parallel_gpu(arr, sum_of_elems, mutex, interval):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    tot_threads =  cuda.gridsize(1)
    sum = cuda.shared.array(shape=(1), dtype=int32)
    if tid == 0 and bid == 0:
        sum[0] = 0
    cuda.syncthreads()  
    psum = 0
    for i in range(interval):
          psum += arr[bid * tot_threads + tid * interval + i]
    lock(mutex)
    sum[0] += psum
    unlock(mutex)
    cuda.syncthreads()
    if tid == bid:
        sum_of_elems[0] += sum[0]


@cuda.jit    
def my_max_parallel_gpu(arr, max_el, mutex, interval):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    tot_threads =  cuda.gridsize(1)
    max = cuda.shared.array(shape=(1), dtype=int32)
    if tid == 0 and bid == 0:
        max[0] = -9999
    cuda.syncthreads()
    pmax = -9999
    for i in range(interval):
        if arr[bid * tot_threads + tid * interval + i] > pmax:
            pmax = arr[bid * tot_threads + tid * interval + i]

    lock(mutex)
    if pmax > max[0]:
        max[0] = pmax
    unlock(mutex)
    cuda.syncthreads()
    if tid == bid:
        if max_el[0] < max[0]:
            max_el[0] = max[0]

@cuda.jit    
def my_min_parallel_gpu(arr, min_el, mutex, interval):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    tot_threads =  cuda.gridsize(1)
    min = cuda.shared.array(shape=(1), dtype=int32)
    if tid == 0 and bid == 0:
        min[0] = 999999999
    cuda.syncthreads()
    pmin = 999999999
    for i in range(interval):
        if arr[bid * tot_threads + tid * interval + i] < pmin:
            pmin = arr[bid * tot_threads + tid * interval + i]
    lock(mutex)
    if pmin < min[0]:
        min[0] = pmin
    unlock(mutex)
    cuda.syncthreads()
    if tid == bid:
        if min[0] < min_el[0]:
            min_el[0] = min[0]

@cuda.jit  
def my_average_parallel_gpu(arr, avg_of_elems, mutex, interval):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    tot_threads =  cuda.gridsize(1)
    sum = cuda.shared.array(shape=(1), dtype=int32)
    if tid == 0 and bid == 0:
        sum[0] = 0
    cuda.syncthreads()  
    psum = 0
    for i in range(interval):
          psum += arr[bid * tot_threads + tid * interval + i]
    lock(mutex)
    sum[0] += psum
    unlock(mutex)
    cuda.syncthreads()
    if tid == bid:
        avg_of_elems[0] += sum[0]


# for i in range(2,N+1):
#     numar = N/i
#     if numar.is_integer():
#         print(i)

# print("terminat")

# suma = np.zeros(1, dtype=int)
# maxim = np.zeros(1, dtype=int)
# minim = np.zeros(1, dtype=int)
# avg = np.zeros(1, dtype=float)




nr_threads = 20
nr_blocks = 20

interval = N // (nr_threads * nr_blocks)

suma = np.zeros(1, dtype=int)
maxim = np.zeros(1, dtype=int)
minim = np.zeros(1, dtype=int)
avg = np.zeros(1, dtype=float)



array_gpu = cuda.to_device(array)

mutex = cuda.to_device(np.zeros((1,), dtype=np.int64))
suma_gpu = cuda.to_device(suma)
maxim_gpu = cuda.to_device(maxim)
minim_gpu = cuda.to_device(minim)
avg_gpu = cuda.to_device(avg)

time_bef_suma = time.time()
my_sum_parallel_gpu[nr_blocks , nr_threads](array_gpu, suma_gpu, mutex, interval)
time_aft_suma = time.time()
print('My GPU parallel sum: ', time_aft_suma - time_bef_suma)


time_bef_maxim = time.time()
my_max_parallel_gpu[nr_blocks , nr_threads](array_gpu, maxim_gpu, mutex, interval)
time_aft_maxim = time.time()
print('My GPU parallel max: ',time_aft_maxim - time_bef_maxim)


time_bef_minim = time.time()
my_min_parallel_gpu[nr_blocks , nr_threads](array_gpu, minim_gpu, mutex, interval)
time_aft_minim = time.time()
print('My GPU parallel min: ',time_aft_minim - time_bef_minim)

time_bef_average = time.time()
my_average_parallel_gpu[nr_blocks , nr_threads](array_gpu, avg_gpu, mutex, interval)
time_aft_average = time.time()
print('My GPU parallel avg: ',time_aft_average - time_bef_average)






@cuda.jit
def sum_kernel(arr, result):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    # Calculate the starting index for this block
    start = bid * bdim
    
    #print(type(a))
    # Calculate the ending index for this block
    end = min(start + bdim, arr.size)
    # Shared memory for partial sums within the block
    s_arr = cuda.shared.array(block_size, dtype=np.int32)

    # Initialize partial sum to 0
    s_arr[tid] = 0
    cuda.syncthreads()

    # Calculate sum within the block
    for i in range(start + tid, end, bdim):
        s_arr[tid] += arr[i]
    cuda.syncthreads()

    # Store the final sum of this block in the global memory
    if tid == 0:
        for i in s_arr:
            result[bid] += i
        
        cuda.syncthreads()


@cuda.jit
def max_kernel(arr, result):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x

    # Calculate the starting index for this block
    start = bid * bdim

    # Calculate the ending index for this block
    end = min(start + bdim, arr.size)
    # Shared memory for partial sums within the block
    s_arr = cuda.shared.array(block_size, dtype=np.int32)

    # Initialize partial sum to 0
    s_arr[tid] = -9999
    cuda.syncthreads()

    # Calculate sum within the block
    for i in range(start + tid, end, bdim):
        if arr[i] > s_arr[tid]:
            s_arr[tid] = arr[i]
    cuda.syncthreads()

    # Store the final sum of this block in the global memory
    if tid == 0:
        for i in s_arr:
            if i > result[bid]:
                result[bid] = i
        
        cuda.syncthreads()




@cuda.jit
def min_kernel(arr, result):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x

    # Calculate the starting index for this block
    start = bid * bdim
    
    #print(type(a))
    # Calculate the ending index for this block
    end = min(start + bdim, arr.size)
    # Shared memory for partial sums within the block
    s_arr = cuda.shared.array(block_size, dtype=np.int32)


    # Initialize partial sum to 0
    s_arr[tid] = 99999
    cuda.syncthreads()
    # Calculate sum within the block
    for i in range(start + tid, end, bdim):
        if arr[i] < s_arr[tid]:
            s_arr[tid] = arr[i]
    cuda.syncthreads()
    


    # Store the final sum of this block in the global memory
    if tid == 0:
        for i in s_arr:
            if i < result[bid]:
                result[bid] = i
        
        cuda.syncthreads()


@cuda.jit
def avg_kernel(arr, result):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x

    # Calculate the starting index for this block
    start = bid * bdim
    
    #print(type(a))
    # Calculate the ending index for this block
    end = min(start + bdim, arr.size)
    # Shared memory for partial sums within the block
    s_arr = cuda.shared.array(block_size, dtype=np.int32)
    #s_arr = cuda.shared.array_like(np.int32, bdim)

    # Initialize partial sum to 0
    s_arr[tid] = 0
    cuda.syncthreads()

    # Calculate sum within the block
    for i in range(start + tid, end, bdim):
        s_arr[tid] += arr[i]
    cuda.syncthreads()


    # Store the final sum of this block in the global memory
    if tid == 0:
        for i in s_arr:
            result[bid] += i
        
        cuda.syncthreads()




def sum_array(arr):
    # Transfer the array to the device
    d_arr = cuda.to_device(arr)

    # Calculate the number of blocks
    blockdim = block_size
    griddim = (arr.size + blockdim - 1) // blockdim

    # Create an array to store block sums
    block_sums = np.zeros(griddim, dtype=np.int32)
    block_sums_gpu = cuda.to_device(block_sums)
    # Invoke the kernel
    sum_kernel[griddim, blockdim](d_arr, block_sums_gpu)

    # Reduce the block sums on the CPU
    final_sum = np.sum(block_sums_gpu)

    return final_sum

def max_array(arr):
    # Transfer the array to the device
    d_arr = cuda.to_device(arr)

    # Calculate the number of blocks
    blockdim = block_size
    griddim = (arr.size + blockdim - 1) // blockdim

    # Create an array to store block sums
    block_maxs = np.zeros(griddim, dtype=np.int32)
    block_maxs_gpu = cuda.to_device(block_maxs)
    # Invoke the kernel
    max_kernel[griddim, blockdim](d_arr, block_maxs_gpu)

    # Reduce the block sums on the CPU
    final_max = np.max(block_maxs_gpu)

    return final_max

def min_array(arr):
    # Transfer the array to the device
    d_arr = cuda.to_device(arr)

    # Calculate the number of blocks
    blockdim = block_size
    griddim = (arr.size + blockdim - 1) // blockdim

    # Create an array to store block sums
    block_mins = np.ones(griddim, dtype=np.int32) * 999999
    block_mins_gpu = cuda.to_device(block_mins)
    # Invoke the kernel
    min_kernel[griddim, blockdim](d_arr, block_mins_gpu)
    # Reduce the block sums on the CPU
    final_min = np.min(block_mins_gpu)

    return final_min

def avg_array(arr):
    # Transfer the array to the device
    d_arr = cuda.to_device(arr)

    # Calculate the number of blocks
    blockdim = block_size
    griddim = (arr.size + blockdim - 1) // blockdim

    # Create an array to store block sums
    block_avgs = np.zeros(griddim, dtype=np.int32)
    block_avgs_gpu = cuda.to_device(block_avgs)
    # Invoke the kernel
    avg_kernel[griddim, blockdim](d_arr, block_avgs_gpu)

    # Reduce the block sums on the CPU
    final_avg = np.sum(block_avgs_gpu) / len(arr)

    return final_avg


time_bef_suma = time.time()
result = sum_array(array)
# print(result)
time_aft_suma = time.time()
print('My GPU parallel sum: ', time_aft_suma - time_bef_suma)
#print(result)

time_bef_maxim = time.time()
result = max_array(array)
time_aft_maxim = time.time()
print('My GPU parallel max: ',time_aft_maxim - time_bef_maxim)
#print(result)

time_bef_minim = time.time()
result = min_array(array)
time_aft_minim = time.time()
print('My GPU parallel min: ',time_aft_minim - time_bef_minim)
#print(result)

time_bef_average = time.time()
result = avg_array(array)
time_aft_average = time.time()
print('My GPU parallel avg: ',time_aft_average - time_bef_average)
