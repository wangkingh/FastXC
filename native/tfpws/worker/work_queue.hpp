#ifndef TFPWS_WORK_QUEUE_HPP
#define TFPWS_WORK_QUEUE_HPP

#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

template <typename T>
class WorkQueue
{
public:
    void push(T item)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
        cv_.notify_one();
    }

    bool pop(T &item)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return closed_ || !queue_.empty(); });
        if (queue_.empty())
            return false;
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void close()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<T> queue_;
    bool closed_ = false;
};

#endif
