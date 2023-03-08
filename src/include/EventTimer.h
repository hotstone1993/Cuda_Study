
#ifndef EVENT_TIMER
#define EVENT_TIMER
#include <chrono>

class EventTimer {
public:
    EventTimer() {
    }

    ~EventTimer() {
    }

    void startTimer() {
        start = std::chrono::system_clock::now();
    }

    void stopTimer() {
        end = std::chrono::system_clock::now();
    }

    void printElapsedTime(std::string_view device) {
        std::cerr << device << " - execution time : " << std::chrono::duration<float, std::milli>(end - start).count() << "ms\n";
    }
    
private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};

#endif // EVENT_TIMER