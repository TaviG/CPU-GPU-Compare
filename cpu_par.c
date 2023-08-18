#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define ARRAY_SIZE 150000000
#define NUM_THREADS 8

typedef struct {
    int thread_id;
    int* array;
    int chunk_size;
    int partial_sum;
} ThreadData;

typedef struct {
    int thread_id;
    int* array;
    int chunk_size;
    int partial_max;
} ThreadData2;

typedef struct {
    int thread_id;
    int* array;
    int chunk_size;
    int partial_min;
} ThreadData3;

typedef struct {
    int thread_id;
    int* array;
    int chunk_size;
    double partial_sum;
} ThreadData4;


void* calculate_sum(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int start_index = data->thread_id * data->chunk_size;
    int end_index = start_index + data->chunk_size;
    int local_sum = 0;

    for (int i = start_index; i < end_index; i++) {
        local_sum += data->array[i];
    }

    data->partial_sum = local_sum;
    pthread_exit(NULL);
}

void* calculate_max(void* arg) {
    ThreadData2* data = (ThreadData2*)arg;
    int start_index = data->thread_id * data->chunk_size;
    int end_index = start_index + data->chunk_size;
    int local_max = 0;

    for (int i = start_index; i < end_index; i++) {
        if (data->array[i] > local_max) {
            local_max = data->array[i];
        }
    }

    data->partial_max = local_max;
    pthread_exit(NULL);
}

void* calculate_min(void* arg) {
    ThreadData3* data = (ThreadData3*)arg;
    int start_index = data->thread_id * data->chunk_size;
    int end_index = start_index + data->chunk_size;
    int local_min = data->array[start_index];

    for (int i = start_index + 1; i < end_index; i++) {
        if (data->array[i] < local_min) {
            local_min = data->array[i];
        }
    }

    data->partial_min = local_min;
    pthread_exit(NULL);
}


int main() {
    int *array = (int*)malloc(ARRAY_SIZE * sizeof(int));
    int sum = 0;

    // Initialize the array with some values
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = 1;
    }

    

    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];
    int chunk_size = ARRAY_SIZE / NUM_THREADS;


    clock_t t;
    t = clock();

    // Create threads and assign data
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].array = array;
        thread_data[i].chunk_size = chunk_size;

        pthread_create(&threads[i], NULL, calculate_sum, (void*)&thread_data[i]);
    }

    // Wait for threads to complete and accumulate partial sums
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        sum += thread_data[i].partial_sum;
    }


     t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
 
    printf("Sum Took %f seconds to execute \n", time_taken);
    printf("Sum: %d\n", sum);

    // Max
    int max_value = 0;

    pthread_t threads2[NUM_THREADS];
    ThreadData2 thread_data2[NUM_THREADS];
    chunk_size = ARRAY_SIZE / NUM_THREADS;
    
    t = clock();

    // Create threads and assign data
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data2[i].thread_id = i;
        thread_data2[i].array = array;
        thread_data2[i].chunk_size = chunk_size;

        pthread_create(&threads2[i], NULL, calculate_max, (void*)&thread_data2[i]);
    }

    // Wait for threads to complete and find the maximum value
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads2[i], NULL);
        if (thread_data2[i].partial_max > max_value) {
            max_value = thread_data2[i].partial_max;
        }
    }

    t = clock() - t;
    time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
 
    printf("Max Took %f seconds to execute \n", time_taken);
    printf("Max value: %d\n", max_value);

    // Minim

    int min_value = 0;

    pthread_t threads3[NUM_THREADS];
    ThreadData3 thread_data3[NUM_THREADS];
    chunk_size = ARRAY_SIZE / NUM_THREADS;

     t = clock();

    // Create threads and assign data
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data3[i].thread_id = i;
        thread_data3[i].array = array;
        thread_data3[i].chunk_size = chunk_size;

        pthread_create(&threads3[i], NULL, calculate_min, (void*)&thread_data3[i]);
    }

    // Wait for threads to complete and find the minimum value
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads3[i], NULL);
        if (thread_data3[i].partial_min < min_value || i == 0) {
            min_value = thread_data3[i].partial_min;
        }
    }

    t = clock() - t;
    time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

    printf("Min Took %f seconds to execute \n", time_taken);
    printf("Min value: %d\n", min_value);


    // Average 

    int local_sum = 0;
    double average = 0;
    pthread_t threads4[NUM_THREADS];
    ThreadData4 thread_data4[NUM_THREADS];
    chunk_size = ARRAY_SIZE / NUM_THREADS;

    t = clock();
    // Create threads and assign data
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data4[i].thread_id = i;
        thread_data4[i].array = array;
        thread_data4[i].chunk_size = chunk_size;

        pthread_create(&threads4[i], NULL, calculate_sum, (void*)&thread_data4[i]);
    }

    // Wait for threads to complete and accumulate partial sums
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads4[i], NULL);
        sum += thread_data4[i].partial_sum;
    }

    average = sum / ARRAY_SIZE;

    t = clock() - t;
    time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
    printf("Average Took %f seconds to execute \n", time_taken);

    printf("Average: %.2f\n", average);
    return 0;
}