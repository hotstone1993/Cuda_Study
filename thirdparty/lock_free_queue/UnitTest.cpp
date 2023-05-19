
#include "LockFreeQueue.hpp"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

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
}