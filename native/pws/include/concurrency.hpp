#ifndef CONCURRENCY_HPP
#define CONCURRENCY_HPP

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

template <typename T>
class WorkQueue
{
public:
    void push(T &&item)
    {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            queue_.push_back(std::move(item));
        }
        cv_.notify_one();
    }

    bool pop(T &item)
    {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [&] { return closed_ || !queue_.empty(); });
        if (queue_.empty())
            return false;
        item = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }

    bool try_pop(T &item)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (queue_.empty())
            return false;
        item = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }

    void close()
    {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            closed_ = true;
        }
        cv_.notify_all();
    }

private:
    std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<T> queue_;
    bool closed_ = false;
};

class HostGroupBudget
{
public:
    explicit HostGroupBudget(std::size_t limit) : limit_(limit) {}

    void acquire(std::size_t groups)
    {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [&] {
            if (groups > limit_)
                return used_ == 0;
            return used_ + groups <= limit_;
        });
        used_ += groups;
    }

    void release(std::size_t groups)
    {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            used_ = (groups > used_) ? 0 : (used_ - groups);
        }
        cv_.notify_all();
    }

private:
    std::mutex mtx_;
    std::condition_variable cv_;
    std::size_t limit_;
    std::size_t used_ = 0;
};

#endif
