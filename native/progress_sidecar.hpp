#ifndef FASTXC_PROGRESS_SIDECAR_HPP
#define FASTXC_PROGRESS_SIDECAR_HPP

#include <cstdio>
#include <cstring>
#include <limits.h>
#include <pthread.h>
#include <string>
#include <vector>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

struct FastxcProgressRow
{
    std::string task;
    std::string status;
    size_t completed;
    size_t total;
    std::string unit;
    std::string detail;
};

class FastxcProgressSidecar
{
public:
    FastxcProgressSidecar() : path_(NULL), enabled_(false), initialized_(false)
    {
    }

    ~FastxcProgressSidecar()
    {
        destroy();
    }

    void init(const char *path)
    {
        destroy();
        if (!path_enabled(path))
            return;
        path_ = path;
        enabled_ = true;
        pthread_mutex_init(&mutex_, NULL);
        initialized_ = true;
    }

    bool enabled() const
    {
        return enabled_;
    }

    void set_rows(const std::vector<FastxcProgressRow> &rows)
    {
        if (!enabled_)
            return;
        pthread_mutex_lock(&mutex_);
        rows_ = rows;
        write_locked();
        pthread_mutex_unlock(&mutex_);
    }

    void update(const std::string &task,
                const std::string &status,
                size_t completed,
                size_t total,
                const std::string &unit,
                const std::string &detail)
    {
        if (!enabled_)
            return;
        pthread_mutex_lock(&mutex_);
        FastxcProgressRow *row = find_locked(task);
        if (!row)
        {
            rows_.push_back({task, status, completed, total, unit, detail});
        }
        else
        {
            row->status = status;
            row->completed = completed;
            row->total = total;
            row->unit = unit;
            row->detail = detail;
        }
        write_locked();
        pthread_mutex_unlock(&mutex_);
    }

    void add(const std::string &task, size_t delta, const std::string &detail)
    {
        if (!enabled_)
            return;
        pthread_mutex_lock(&mutex_);
        FastxcProgressRow *row = find_locked(task);
        if (row)
        {
            row->completed += delta;
            if (row->completed > row->total)
                row->completed = row->total;
            row->status = (row->total > 0 && row->completed >= row->total) ? "DONE" : "RUNNING";
            if (!detail.empty())
                row->detail = detail;
            write_locked();
        }
        pthread_mutex_unlock(&mutex_);
    }

    void finish(const std::string &status, bool complete)
    {
        if (!enabled_)
            return;
        pthread_mutex_lock(&mutex_);
        for (size_t i = 0; i < rows_.size(); ++i)
        {
            rows_[i].status = status;
            if (complete)
                rows_[i].completed = rows_[i].total;
        }
        write_locked();
        pthread_mutex_unlock(&mutex_);
    }

    void destroy()
    {
        if (initialized_)
        {
            pthread_mutex_destroy(&mutex_);
            initialized_ = false;
        }
        enabled_ = false;
        rows_.clear();
        path_ = NULL;
    }

private:
    const char *path_;
    bool enabled_;
    bool initialized_;
    pthread_mutex_t mutex_;
    std::vector<FastxcProgressRow> rows_;

    static bool path_enabled(const char *path)
    {
        return path && path[0] != '\0' && std::strcmp(path, "NONE") != 0;
    }

    FastxcProgressRow *find_locked(const std::string &task)
    {
        for (size_t i = 0; i < rows_.size(); ++i)
        {
            if (rows_[i].task == task)
                return &rows_[i];
        }
        return NULL;
    }

    static std::string clean_field(const std::string &text)
    {
        std::string out = text;
        for (size_t i = 0; i < out.size(); ++i)
        {
            if (out[i] == '\t' || out[i] == '\r' || out[i] == '\n')
                out[i] = ' ';
        }
        return out;
    }

    void write_locked()
    {
        if (!enabled_ || !path_)
            return;

        char tmp_path[PATH_MAX];
        int n = std::snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path_);
        if (n < 0 || n >= (int)sizeof(tmp_path))
        {
            enabled_ = false;
            return;
        }

        FILE *fp = std::fopen(tmp_path, "w");
        if (!fp)
        {
            enabled_ = false;
            return;
        }

        std::fprintf(fp, "task\tstatus\tcompleted\ttotal\tunit\tdetail\n");
        for (size_t i = 0; i < rows_.size(); ++i)
        {
            const FastxcProgressRow &row = rows_[i];
            std::fprintf(fp, "%s\t%s\t%zu\t%zu\t%s\t%s\n",
                         clean_field(row.task).c_str(),
                         clean_field(row.status).c_str(),
                         row.completed,
                         row.total,
                         clean_field(row.unit).c_str(),
                         clean_field(row.detail).c_str());
        }

        if (std::fclose(fp) != 0)
        {
            enabled_ = false;
            return;
        }
        if (std::rename(tmp_path, path_) != 0)
            enabled_ = false;
    }
};

#endif
