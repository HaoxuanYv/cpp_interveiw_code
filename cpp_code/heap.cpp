#include <iostream>
#include <iostream>
#include <vector>
using namespace std;
class MinHeap{
private:
    vector<int> heap;
    void shiftUp(int i){
        while(i > 0){
            // 0 1 2 -> 2-1 / 2 = 0
            int parent = (i - 1) / 2;
            if(heap[parent] <= heap[i]) break;
            swap(heap[parent], heap[i]);
            i = parent;
        }
    }
    void shiftDown(int i){
        int n = (int)heap.size();
        while(true){
            int l = 2 * i + 1;
            int r = 2 * i + 2;
            int smallest = i;

            if(l < n && heap[l] < heap[smallest]){
                smallest = l;
            }
            if(r < n && heap[r] < heap[smallest]){
                smallest = r;
            }
            if(smallest == i){
                break;
            }
            swap(heap[i], heap[smallest]);
            i = smallest;
        }
    }
public: 
    void push(int i){
        heap.push_back(i);
        shiftUp(heap.size() - 1);
    }
    void pop(){
        if(empty()) return;
        heap[0] = heap.back();
        heap.pop_back();
        if(!heap.empty()){
            shiftDown(0);
        }
    }
    int top(){
        return heap[0];
    }
    int size(){
        return heap.size();
    }
    bool empty(){
        return heap.size() == 0 ? true:false;
    }
    void buildFrom(vector<int>& nums){
        int n = nums.size();
        for(int i = (n - 2) / 2; i >= 0; i--){
            shiftDown(i);
        }
    }
}
void heapify(vector<int>& a, int i, int n){
    while(true){
        int smallest = i;
        int l = 2 * i + 1;
        int r = 2 * i + 2;
        if(l < n && a[smallest] > a[l]){
            smallest = l;
        }
        if(r < n && a[smallest] > a[r]){
            smallest = r;
        }
        if(smallest == i) break; // 最小值就是本身，不用轮换了
        swap(a[i], a[smallest]);
        i = smallest;
    }
}
void buildHeap(std::vector<int>& arr, int n){
  for(int i = (n - 2) / 2; i >= 0; i--){
    heapify(arr, i, n); 
  } 
  return;
}

