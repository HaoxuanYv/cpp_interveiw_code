/***
 * unique_ptr手撕
 * 核心： 独占资源管理；RAII；支持移动语义以及禁止拷贝构造
 * 接口：
 *      构造：接受裸指针，获取资源所有权
 *      析构：自动释放资源
 *      访问：opertator*, operator->
 *      移动：支持所有权转移
 *      重置：reset，release()
 *      查询：get(), operator bool()
 *
***/
template <typename T>
class unique_ptr{
private:
    T* ptr_;
public:
    explicit unique_ptr(T* ptr = nullptr) : ptr_(ptr){}
    ~unique_ptr(){
        delete ptr_;
    }

    unique_ptr(const unique_ptr&) = delete;
    unique_ptr& operator = (const unique_ptr&) = delete;

    unique_ptr(unique_ptr && other) noexcept : ptr_(other.ptr_){
        other.ptr_ = nullptr; 
    }
    unique_ptr& operator = (unique_ptr && other) noexcept{
        if(this != other){
            delete ptr_;
            ptr_ = std::exchange(other.ptr_, nullptr);
        }
        return this;
    }

    T& operator*() const{
        return *ptr_; //保持原声指针一致
    }
    T* operator ->() const{
        return ptr_;
    }
    T* get() const{
        return ptr_;
    }
    explicit operator bool () const {
        return ptr_!= nullptr;
    }
    
    T* release() {
        return std::exchange(ptr_, nullptr); // 返回原指针，将ptr_替换为null
                                            }
    void reset(T* p = nullptr){
        delete ptr_;
        ptr_ = p;
    }

    void swap(unique_ptr & other) noexcept{
        std::swap(ptr_, other.ptr_);
    }
}
template <typename T>
void swap(unique_ptr<T>& lhs, unique_ptr<T> & rhs) noexcept{
    lhs.swap(rhs);
}
template <typename T>
bool operator == (const unique_ptr<T>& lhs, const unique_ptr<T>& rhs){
    return lhs.get() == rhs.get();
}
template <typename T>
bool operator != (const unique_ptr<T>& lhs, const unique_ptr<T>& rhs){
    return !(lhs.get() == rhs.get());}

template <typename T>
bool operator == (const unique_ptr<T>& lhs, std::nullptr_t){
    return !lhs;
}

template <typename T>
bool operator != (const unique_ptr<T>& lhs, std::nullptr_t){
    return static_cast<bool>(lhs);
}

