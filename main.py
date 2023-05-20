import numpy as np
import time
import threading
import sys
from funcs import *


N = int(sys.argv[1])
num_threads_cpu = int(sys.argv[2])
array = np.zeros(N,dtype=int)

for i in range(len(array)):
    array[i] = i+1

time_bef_suma = time.time()
suma = sum(array)
time_aft_suma = time.time()
print('Functie sum: ', time_aft_suma - time_bef_suma)

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