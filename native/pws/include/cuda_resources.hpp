#ifndef CUDA_RESOURCES_HPP
#define CUDA_RESOURCES_HPP

#include <cstddef>

#include <cufft.h>

#include "cuda.util.cuh"

template <typename T>
class CudaDeviceBuffer
{
public:
    CudaDeviceBuffer() : ptr_(NULL) {}
    ~CudaDeviceBuffer() { GpuFree((void **)&ptr_); }

    void allocate(std::size_t count)
    {
        GpuMalloc((void **)&ptr_, count * sizeof(T));
    }

    T *get() const
    {
        return ptr_;
    }

    void free_now()
    {
        GpuFree((void **)&ptr_);
    }

private:
    CudaDeviceBuffer(const CudaDeviceBuffer &);
    CudaDeviceBuffer &operator=(const CudaDeviceBuffer &);

    T *ptr_;
};

class CufftPlanGuard
{
public:
    CufftPlanGuard() : handle_(0), active_(false) {}
    ~CufftPlanGuard() { destroy_now(); }

    void create_many(int rank, int *n, int *inembed,
                     int istride, int idist, int *onembed,
                     int ostride, int odist, cufftType type, int batch)
    {
        CufftPlanAlloc(&handle_, rank, n, inembed, istride, idist,
                       onembed, ostride, odist, type, batch);
        active_ = true;
    }

    cufftHandle get() const
    {
        return handle_;
    }

    void destroy_now()
    {
        if (active_)
        {
            cufftDestroy(handle_);
            active_ = false;
            handle_ = 0;
        }
    }

private:
    CufftPlanGuard(const CufftPlanGuard &);
    CufftPlanGuard &operator=(const CufftPlanGuard &);

    cufftHandle handle_;
    bool active_;
};

#endif
