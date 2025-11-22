#include <iostream>
#include <functional>
#include <thread>
#include <queue>
#include <utility>
#include <mutex>
#include <condition_variable>
#include <vector>
template <typename T>
class BlockingQueue{
public:
    BlockingQueue(bool nonblock = false) : nonblock_(nonblock){};

    void Push(const T& value){
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(value);
        not_empty_.notify_one();
    }

    bool Pop(T& value){
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this]()-> bool{
                return !queue_.empty()||nonblock_;
        });
        if(queue_.empty()){
            return false;
        }
        value = queue_.front();
        queue_.pop();
        return true;
    }

    void Cancel(){
        std::lock_guard<std::mutex> lock(mutex_);
        nonblock_ = true;
        not_empty_.notify_all();
    }
private:
    bool nonblock_;
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
};

template <typename T>
class BlockingQueuePro{
public:
    BlockingQueuePro(bool nonblock = false): nonblock_(nonblock){};

    void Push(const T& value){
        std::lock_guard<std::mutex> lock(producer_mutex_);
        producer_queue_.push(value);
        not_empty_.notify_one();
    }

    bool Pop(T& value){
        std::unique_lock<std::mutex> lock(consumer_mutex_);
       if(consumer_queue_.empty() && SwapQueue_() == 0){
          return false;
       } 
       value = consumer_queue_.front();
       consumer_queue_.pop();
       return true;
    }

    void Cancel(){
        std::lock_guard<std::mutex> lock(producer_mutex_);
        nonblock_ = true;
        not_empty_.notify_all();
    }

private:
    int SwapQueue_(){
        std::unique_lock<std::mutex> lock(producer_mutex_);
        not_empty_.wait(lock, [this]{ return !producer_queue_.empty() || nonblock_; });
        std::swap(producer_queue_, consumer_queue_);
        return consumer_queue_.size();
    }
    std::queue<T> consumer_queue_;
    std::queue<T> producer_queue_;
    std::mutex consumer_mutex_;
    std::mutex producer_mutex_;
    std::condition_variable not_empty_;
    bool nonblock_;
};
class ThreadPool{
public:
    explicit ThreadPool(int num_threads){
        for(size_t i = 0; i < num_threads; ++i){
            workers_.emplace_back([this]()->void { Worker();});
        }
    }
    ~ThreadPool(){
        task_queue_.Cancel();
        for(auto &worker:workers_){
            if(worker.joinable())
                worker.join();
        }
    }
    template<typename F, typename ...Args>
    void Post(F &&f, Args && ...args){
        auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...); 
        task_queue_.Push(task);
    }
private:
    void Worker(){
        while(true) {
            std::function<void()> task;
            if(!task_queue_.Pop(task)){
                break;
            }
            task();
        }
    }
    BlockingQueue<std::function<void()>> task_queue_;
    std::vector<std::thread> workers_;
};

