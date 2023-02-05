#ifndef THREAD_POOL
#define THREAD_POOL

#include <condition_variable>
#include <functional>
#include <thread>
#include <future>
#include <vector>
#include <mutex>
#include <queue>

class ThreadPool {
public:
    ThreadPool(size_t count) {
        for (size_t idx = 0; idx < count; ++idx) {
            threads.emplace_back([&]() {
                work();
             });
        }
    }

    ~ThreadPool() {
        await();
    }

    template <class Func, class... Args>
    std::future<typename std::result_of<Func(Args...)>::type> addJob(Func&& f, Args&&... args) {
        using returnType = typename std::result_of<Func(Args...)>::type;
        auto job = std::make_shared<std::packaged_task<returnType()>>(std::bind(std::forward<Func>(f), std::forward<Args>(args)...));
        std::future<returnType> jobResultFuture = job->get_future();

        m.lock();
        jobQueue.push([job]() {
            (*job)();
        });
        m.unlock();

        cv.notify_one();

        return jobResultFuture;
    }

private:
    void await() {
        isFinish = true;
        cv.notify_all();
        for (std::thread& thread : threads) {
            thread.join();
        }
    };

    void work() {
        while (true) {
            std::unique_lock<std::mutex> lock(m);

            cv.wait(lock, [&]() -> bool {
                return !jobQueue.empty() || isFinish;
            });

            if (jobQueue.empty() && isFinish) {
                lock.unlock();
                return;
            }
            else if (!jobQueue.empty()) {
                auto job = std::move(jobQueue.front());
                jobQueue.pop();
                lock.unlock();
                job();
            }
            else {
                lock.unlock();
            }
        }
    }

    std::mutex m;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> jobQueue;

    bool isFinish = false;
};

#endif // THREAD_POOL