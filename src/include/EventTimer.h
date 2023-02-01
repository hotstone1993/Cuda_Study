

#ifndef EVENT_TIMER
#define EVENT_TIMER

class EventTimer {
public:
    EventTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~EventTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void startTimer() {
        cudaEventRecord(start);
    }

    void stopTimer() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    }

    void printElapsedTime() {
        float milliseconds = 0.0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::cerr <<" SAXPY execution time : " << milliseconds << "ms\n";
    }
    
private:
    cudaEvent_t start, stop;
};

#endif // EVENT_TIMER