#ifndef TFPWS_HOST_WORKSPACE_BUDGET_HPP
#define TFPWS_HOST_WORKSPACE_BUDGET_HPP

#include <condition_variable>
#include <cstddef>
#include <mutex>

class HostWorkspaceBudget
{
public:
    explicit HostWorkspaceBudget(std::size_t capacity_bytes)
        : capacity_bytes_(capacity_bytes), used_bytes_(0)
    {
    }

    bool acquire(std::size_t bytes)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (bytes > capacity_bytes_)
            return false;
        cv_.wait(lock, [this, bytes] {
            return used_bytes_ <= capacity_bytes_ - bytes;
        });
        used_bytes_ += bytes;
        return true;
    }

    void release(std::size_t bytes)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (bytes > used_bytes_)
            used_bytes_ = 0;
        else
            used_bytes_ -= bytes;
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::size_t capacity_bytes_;
    std::size_t used_bytes_;
};

class HostWorkspaceLease
{
public:
    HostWorkspaceLease() : budget_(NULL), bytes_(0), acquired_(false)
    {
    }

    ~HostWorkspaceLease()
    {
        release();
    }

    bool acquire(HostWorkspaceBudget *budget, std::size_t bytes)
    {
        budget_ = budget;
        bytes_ = bytes;
        acquired_ = budget_ && budget_->acquire(bytes_);
        return acquired_;
    }

    void release()
    {
        if (acquired_ && budget_)
        {
            budget_->release(bytes_);
            acquired_ = false;
        }
    }

private:
    HostWorkspaceBudget *budget_;
    std::size_t bytes_;
    bool acquired_;
};

#endif
