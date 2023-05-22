
#include "LockFreeQueue.hpp"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <vector>
#include <thread>
#include <iostream>

template <typename T>
void multiConsumerFunc(std::stop_token token, LockFreeQueue<T>& q, int threadIdx) {
    int count = 3;
    int holder;

    while (count > 0) {
        if (!q.empty()) {
            bool isSuccess = q.pop(holder);
            if (isSuccess) {
                --count;
                std::cout << "thread(" << threadIdx << "):"<< holder << std::endl;
            }
        }
        if (token.stop_requested()) {
            return;
        }
    }
}

TEST_CASE("LockFreeQueue Test", "LockFreeQueue") {

    SECTION( "SPSC" )
    {
        LockFreeQueue<int> queue(QueueType::SPSC);

        queue.push(1);
        queue.push(2);
        queue.push(3);

        int holder;

        if (queue.pop(holder)) {
            REQUIRE(holder == 1);
        }
        
        if (queue.pop(holder)) {
            REQUIRE(holder == 2);
        }
        
        if (queue.pop(holder)) {
            REQUIRE(holder == 3);
        }
    }

    SECTION( "SPMC" )
    {
        LockFreeQueue<int> queue(QueueType::SPMC);

        std::vector<std::jthread> threads;
        threads.reserve(5);

        for (int i = 0; i < 5; ++i) {
            threads.emplace_back(multiConsumerFunc<int>, std::ref(queue), i + 1);
        }
        
        for (int i = 0; i < 20; ++i) {
            queue.push(i);
        }

        while (queue.empty()) {}

        for (auto& thread: threads) {
            thread.request_stop();
        }

        REQUIRE(queue.empty());
    }
}