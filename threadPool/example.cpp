#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <string>
#include "threadPool.h"

int main(){
    ThreadPool pool(4);

    std::vector<std::future<int>> futures;

    for(int i = 0; i < 8; i++){
        futures.emplace_back(
                pool.enqueue([i] {
                    std::this_thread::sleep_for(std::chrono::milliseconds(500 + i * 100));
                    std::cout<<"Task " << i << " done in thread "
                        << std::this_thread::get_id() << std::endl;
                    return i*i;}));
    return 0;
}
