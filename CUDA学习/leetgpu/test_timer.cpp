#include "timer.hpp"
#include <math.h>

int main() {
    Timer timer;

    timer.start_cpu();

    for (int i = 0; i < 100000000; i++) {
        // do something
        sqrt(i);
    }

    timer.stop_cpu();

    std::string msg = "from cpu";

    timer.duration_cpu<Timer::ms>(msg);

    return 0;
}