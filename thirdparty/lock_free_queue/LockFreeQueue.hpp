#ifndef LOCK_FREE_QUEUE
#define LOCK_FREE_QUEUE

#include <list>
#include <cstddef>

enum class QueueType {
    MPMC, MPSC, SPMC, SPSC
};

template <typename T>
class LockFreeQueue {
public:
    LockFreeQueue(QueueType type): type(type) {};
    virtual ~LockFreeQueue() {};

    bool empty() const {
        return container.empty();
    }

    size_t size() const {
        return container.size();
    }

    template <typename TYPE>
    void push(TYPE&& object) {
        if (type == QueueType::SPSC) {
            container.push_back(std::forward<TYPE>(object));
        } else if (type == QueueType::SPMC) {
            container.push_back(std::forward<TYPE>(object));
        } else if (type == QueueType::MPSC) {

        } else if (type == QueueType::MPMC) {

        } 
    }
    
    bool pop(T& ref) {
        if (type == QueueType::SPSC) {
            if (!container.empty()) {
                ref = std::move(container.front());
                container.pop_front();
                return true;
            }
        } else if (type == QueueType::SPMC) {
            if (!container.empty()) {
                ref = std::move(container.front());
                container.pop_front();
                return true;
            }
        } else if (type == QueueType::MPSC) {

        } else if (type == QueueType::MPMC) {

        }

        return false;
    }
private:
    QueueType type;
    std::list<T> container;
};

#endif // LOCK_FREE_QUEUE