/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#include "coll_test.h"
#define DATATYPE int64_t

int main(int argc, char **argv) {
    int status = 0;
    int mype, npes;
    size_t alloc_size;
    size_t min_elems, max_elems;
    DATATYPE *buffer = NULL;
    DATATYPE *h_buffer = NULL;
    DATATYPE *d_source, *d_dest;
    DATATYPE *h_source, *h_dest;
    int PE_root = 0;
    char size_string[100];

    read_args(argc, argv);

    uint64_t *size_array = (uint64_t *)calloc(max_size_log, sizeof(uint64_t));
    double **latency_array = (double **)malloc(max_size_log * sizeof(double *));
    cudaStream_t stream;

    for (int i = 0; i < max_size_log; i++) {
        latency_array[i] = (double *)calloc(iters, sizeof(double));
    }

    DEBUG_PRINT("symmetric size requested %lu\n", max_size * 2);
    sprintf(size_string, "%lu", max_size * 2);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        fprintf(stderr, "setenv failed \n");
        status = -1;
        goto out;
    }

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    (void)npes;  // Silence unused variable warning
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaHostAlloc(&h_buffer, max_size * 2, cudaHostAllocDefault));
    h_source = (DATATYPE *)h_buffer;
    h_dest = (DATATYPE *)&h_source[max_size / sizeof(DATATYPE)];
    if (use_mmap) {
        buffer = (DATATYPE *)allocate_mmap_buffer(max_size * 2, mem_handle_type, use_egm);
        DEBUG_PRINT("Allocated mmap buffer\n");
    } else {
        buffer = (DATATYPE *)nvshmem_malloc(max_size * 2);
        DEBUG_PRINT("Allocated nvshmem malloc buffer\n");
    }
    if (!buffer) {
        fprintf(stderr, "nvshmem_malloc failed \n");
        status = -1;
        goto out;
    }
    d_source = (DATATYPE *)buffer;
    d_dest = (DATATYPE *)&d_source[max_size / sizeof(DATATYPE)];

#define CALL_RUN_COLL_ON_STREAM(TYPENAME, TYPE)                                              \
    RUN_COLL_ON_STREAM(broadcast, BCAST, TYPENAME, TYPE, (TYPE *)d_source, (TYPE *)h_source, \
                       (TYPE *)d_dest, (TYPE *)h_dest, npes, PE_root, stream, size_array,    \
                       latency_array);

    switch (datatype.type) {
        case NVSHMEM_INT:
            CALL_RUN_COLL_ON_STREAM(int, int);
            break;
        case NVSHMEM_LONG:
            CALL_RUN_COLL_ON_STREAM(long, long);
            break;
        case NVSHMEM_LONGLONG:
            CALL_RUN_COLL_ON_STREAM(longlong, long long);
            break;
        case NVSHMEM_ULONGLONG:
            CALL_RUN_COLL_ON_STREAM(ulonglong, unsigned long long);
            break;
        case NVSHMEM_SIZE:
            CALL_RUN_COLL_ON_STREAM(size, size_t);
            break;
        case NVSHMEM_PTRDIFF:
            CALL_RUN_COLL_ON_STREAM(ptrdiff, ptrdiff_t);
            break;
        case NVSHMEM_FLOAT:
            CALL_RUN_COLL_ON_STREAM(float, float);
            break;
        case NVSHMEM_DOUBLE:
            CALL_RUN_COLL_ON_STREAM(double, double);
            break;
        case NVSHMEM_UINT:
            CALL_RUN_COLL_ON_STREAM(uint, unsigned int);
            break;
        case NVSHMEM_INT32:
            CALL_RUN_COLL_ON_STREAM(int32, int32_t);
            break;
        case NVSHMEM_INT64:
            CALL_RUN_COLL_ON_STREAM(int64, int64_t);
            break;
        case NVSHMEM_UINT32:
            CALL_RUN_COLL_ON_STREAM(uint32, uint32_t);
            break;
        case NVSHMEM_UINT64:
            CALL_RUN_COLL_ON_STREAM(uint64, uint64_t);
            break;
        case NVSHMEM_FP16:
            CALL_RUN_COLL_ON_STREAM(half, half);
            break;
        case NVSHMEM_BF16:
            CALL_RUN_COLL_ON_STREAM(bfloat16, __nv_bfloat16);
            break;
        default:
            printf("Incorrect datatype specified\n");
            exit(1);
            break;
    }
    if (!mype) {
        print_table_v2("broadcast_on_stream", datatype.name.c_str(), "size (bytes)", "latency",
                       "us", '-', size_array, latency_array, max_size_log, iters);
    }

    CUDA_CHECK(cudaFreeHost(h_buffer));
    if (use_mmap) {
        free_mmap_buffer(buffer);
    } else {
        nvshmem_free(buffer);
    }
    nvshmem_barrier_all();
    CUDA_CHECK(cudaStreamDestroy(stream));

    finalize_wrapper();

out:
    return status;
}
