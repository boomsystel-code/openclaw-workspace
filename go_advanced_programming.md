# Go语言高级编程

## 第一章：Go并发编程深入

### 1.1 Goroutine调度器

#### GMP模型
- **G（Goroutine）**：用户级线程
- **M（Machine）**：操作系统线程
- **P（Processor）**：处理器（调度上下文）

#### 调度策略
- **M:P:G = 1:1:N**
- **工作窃取（Work Stealing）**
  - 空闲处理器从其他队列窃取任务
  - 提高资源利用率

- **抢占式调度**
  - 基于时间片（10ms）
  - 防止单个Goroutine阻塞

### 1.2 通道（Channel）高级特性

#### 无缓冲通道
```go
// 同步通信
ch := make(chan int)

// 发送（阻塞直到有接收者）
ch <- value

// 接收（阻塞直到有发送者）
result := <-ch
```

#### 有缓冲通道
```go
// 带缓冲的通道
ch := make(chan int, 100)

// 非阻塞发送
select {
case ch <- value:
    fmt.Println("sent")
default:
    fmt.Println("buffer full")
}

// 非阻塞接收
select {
case result := <-ch:
    fmt.Println("received:", result)
default:
    fmt.Println("no data")
}
```

#### 单向通道
```go
// 只发送通道
func producer(out chan<- int) {
    for i := 0; i < 10; i++ {
        out <- i
    }
    close(out)
}

// 只接收通道
func consumer(in <-chan int) {
    for num := range in {
        fmt.Println(num)
    }
}
```

### 1.3 同步原语

#### sync.WaitGroup
```go
var wg sync.WaitGroup

for i := 0; i < 10; i++ {
    wg.Add(1)
    go func(id int) {
        defer wg.Done()
        process(id)
    }(i)
}

wg.Wait() // 等待所有goroutine完成
```

#### sync.Once
```go
var once sync.Once
var instance *Singleton

func GetInstance() *Singleton {
    once.Do(func() {
        instance = &Singleton{}
    })
    return instance
}
```

#### sync.Map
```go
var m sync.Map

// 存储
m.Store("key", "value")

// 加载
value, ok := m.Load("key")

// 加载或存储
value, loaded := m.LoadOrStore("key", "newValue")

// 删除
m.Delete("key")

// 遍历
m.Range(func(key, value interface{}) bool {
    fmt.Println(key, value)
    return true
})
```

#### sync.Cond
```go
cond := sync.NewCond(&sync.Mutex{})

go func() {
    cond.L.Lock()
    for !ready {
        cond.Wait()
    }
    cond.L.Unlock()
    // 执行任务
}()

// 通知一个
cond.Signal()

// 通知所有
cond.Broadcast()
```

### 1.4 上下文（Context）

#### Context接口
```go
type Context interface {
    Deadline() (deadline time.Time, ok bool)
    Done() <-chan struct{}
    Err() error
    Value(key interface{}) interface{}
}
```

#### 使用模式
```go
// 创建超时上下文
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

// 带值的上下文
ctx = context.WithValue(ctx, "user_id", 123)

// 在goroutine中使用
go func() {
    select {
    case <-ctx.Done():
        fmt.Println("Context cancelled:", ctx.Err())
        return
    case result := <-doWork():
        // 处理结果
    }
}()

// 在HTTP请求中使用
func handler(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()
    
    select {
    case <-ctx.Done():
        http.Error(w, "timeout", http.StatusGatewayTimeout)
        return
    default:
        // 处理请求
    }
}
```

---

## 第二章：Go内存模型与性能优化

### 2.1 内存逃逸分析

#### 逃逸场景
```go
// 逃逸到堆：返回指针
func escape() *int {
    x := 42
    return &x  // x逃逸到堆
}

// 逃逸到堆：切片增长
func growSlice() []int {
    s := make([]int, 0, 10)
    s = append(s, 1, 2, 3)
    return s  // 可能逃逸
}

// 不逃逸：局部使用
func noEscape() int {
    x := 42
    return x  // x在栈上
}
```

#### 避免逃逸
```go
// 预分配容量
func preAllocate() []int {
    s := make([]int, 0, 100)  // 预分配100
    s = append(s, 1, 2, 3)
    return s  // 避免重新分配
}

// 使用值而非指针
type Point struct {
    X, Y int
}

// 优先值传递
func processPoint(p Point) {
    // 拷贝传递，无逃逸
}
```

### 2.2 性能优化技巧

#### 减少内存分配

**对象池（sync.Pool）**
```go
var bufPool = sync.Pool{
    New: func() interface{} {
        return new(bytes.Buffer)
    },
}

func GetBuffer() *bytes.Buffer {
    return bufPool.Get().(*bytes.Buffer)
}

func PutBuffer(buf *bytes.Buffer) {
    buf.Reset()
    bufPool.Put(buf)
}
```

**strings.Builder**
```go
// 避免字符串拼接
var sb strings.Builder
for i := 0; i < 100; i++ {
    sb.WriteString("item")
    sb.WriteByte(',')
}
result := sb.String()
```

#### 避免反射

**使用接口或类型断言**
```go
// 类型断言优于反射
switch v := interface{}(x).(type) {
case int:
    // 处理int
case string:
    // 处理string
}

// 预编译类型信息
type Encoder interface {
    Encode(v interface{}) error
}

var encoderPool = sync.Pool{
    New: func() interface{} {
        return json.NewEncoder(ioutil.Discard)
    },
}

func EncodeJSON(v interface{}) ([]byte, error) {
    enc := encoderPool.Get().(*json.Encoder)
    defer encoderPool.Put(enc)
    enc.Reset(ioutil.Discard)
    return io.ReadAll(enc.Buffer)
}
```

### 2.3 并发模式

#### Fan-in/Fan-out
```go
// 多生产者，单消费者
func fanIn(channels ...<-chan int) <-chan int {
    out := make(chan int)
    
    var wg sync.WaitGroup
    wg.Add(len(channels))
    
    for _, ch := range channels {
        go func(c <-chan int) {
            defer wg.Done()
            for n := range c {
                out <- n
            }
        }(ch)
    }
    
    go func() {
        wg.Wait()
        close(out)
    }()
    
    return out
}
```

#### 流水线（Pipeline）
```go
func gen(nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        for _, n := range nums {
            out <- n
        }
        close(out)
    }()
    return out
}

func sq(in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        for n := range in {
            out <- n * n
        }
        close(out)
    }()
    return out
}

// 使用流水线
for n := range sq(gen(2, 3, 4)) {
    fmt.Println(n)
}
```

#### Worker Pool
```go
func worker(id int, jobs <-chan Job, results chan<- Result) {
    for job := range jobs {
        result := process(job)
        results <- result
    }
}

func main() {
    jobs := make(chan Job, 100)
    results := make(chan Result, 100)
    
    // 启动worker池
    for w := 1; w <= 3; w++ {
        go worker(w, jobs, results)
    }
    
    // 发送任务
    for j := 1; j <= 5; j++ {
        jobs <- Job{ID: j}
    }
    close(jobs)
    
    // 收集结果
    for a := 1; a <= 5; a++ {
        <-results
    }
}
```

---

## 第三章：Go设计模式

### 3.1 创建型模式

#### 单例模式（Singleton）
```go
type singleton struct {
    data int
}

var instance *singleton
var once sync.Once

func GetInstance() *singleton {
    once.Do(func() {
        instance = &singleton{}
    })
    return instance
}
```

#### 工厂模式
```go
type PaymentMethod interface {
    Pay(amount float64) error
}

type CreditCard struct{}
type PayPal struct{}

func (c *CreditCard) Pay(amount float64) error {
    return nil
}

func (p *PayPal) Pay(amount float64) error {
    return nil
}

func NewPaymentMethod(method string) (PaymentMethod, error) {
    switch method {
    case "credit":
        return &CreditCard{}, nil
    case "paypal":
        return &PayPal{}, nil
    default:
        return nil, fmt.Errorf("unknown method")
    }
}
```

### 3.2 结构型模式

#### 装饰器模式
```go
type Coffee interface {
    GetDescription() string
    GetCost() float64
}

type BaseCoffee struct{}

func (c *BaseCoffee) GetDescription() string {
    return "Coffee"
}

func (c *BaseCoffee) GetCost() float64 {
    return 5.0
}

type MilkDecorator struct {
    coffee Coffee
}

func (d *MilkDecorator) GetDescription() string {
    return d.coffee.GetDescription() + ", Milk"
}

func (d *MilkDecorator) GetCost() float64 {
    return d.coffee.GetCost() + 1.0
}
```

#### 适配器模式
```go
// 旧接口
type LegacyLogger struct{}

func (l *LegacyLogger) Log(message string, level string) {
    fmt.Printf("[%s] %s\n", level, message)
}

// 新接口
type Logger interface {
    Debug(msg string)
    Info(msg string)
    Error(msg string)
}

// 适配器
type LoggerAdapter struct {
    legacy *LegacyLogger
}

func (a *LoggerAdapter) Debug(msg string) {
    a.legacy.Log(msg, "DEBUG")
}

func (a *LoggerAdapter) Info(msg string) {
    a.legacy.Log(msg, "INFO")
}
```

### 3.3 行为型模式

#### 观察者模式
```go
type Subject interface {
    Register(observer Observer)
    Unregister(observer Observer)
    NotifyAll()
}

type Observer interface {
    Update(message string)
}

type ConcreteSubject struct {
    observers []Observer
    message   string
}

func (s *ConcreteSubject) Register(o Observer) {
    s.observers = append(s.observers, o)
}

func (s *ConcreteSubject) NotifyAll() {
    for _, o := range s.observers {
        o.Update(s.message)
    }
}
```

#### 策略模式
```go
type SortStrategy interface {
    Sort(data []int)
}

type BubbleSort struct{}

func (b *BubbleSort) Sort(data []int) {
    // 冒泡排序实现
}

type QuickSort struct{}

func (q *QuickSort) Sort(data []int) {
    // 快速排序实现
}

type Sorter struct {
    strategy SortStrategy
}

func (s *Sorter) SetStrategy(strategy SortStrategy) {
    s.strategy = strategy
}
```

---

## 第四章：Go标准库高级用法

### 4.1 IO操作

#### io.Reader/io.Writer
```go
// 组合Reader
type MultiReader struct {
    readers []io.Reader
}

func (m *MultiReader) Read(p []byte) (n int, err error) {
    for _, r := range m.readers {
        if n, err = r.Read(p); n > 0 {
            return n, nil
        }
    }
    return 0, io.EOF
}

// TeeReader：复制数据到Writer
func copyWithProgress(src io.Reader, dst io.Writer) error {
    tee := io.TeeReader(src, os.Stdout)
    _, err := io.ReadAll(tee)
    return err
}
```

#### io/fs（Go 1.16+）
```go
import "io/fs"

// 遍历目录
fs.WalkDir(dir, ".", func(path string, d fs.DirEntry, err error) error {
    if err != nil {
        return err
    }
    
    info, err := d.Info()
    if err != nil {
        return err
    }
    
    fmt.Println(path, info.Size())
    return nil
})

// 读取嵌入的文件
//go:embed data/*
var data embed.FS

func ReadDataFile(name string) ([]byte, error) {
    return fs.ReadFile(data, name)
}
```

### 4.2 网络编程

#### HTTP客户端高级用法
```go
// 自定义Transport
tr := &http.Transport{
    MaxIdleConns:        100,
    MaxIdleConnsPerHost:  10,
    IdleConnTimeout:      90 * time.Second,
    TLSHandshakeTimeout:  10 * time.Second,
}

client := &http.Client{
    Transport: tr,
    Timeout:   30 * time.Second,
    CheckRedirect: func(req *http.Request, via []*http.Request) error {
        // 自定义重定向处理
        return http.ErrUseLastResponse
    },
}

// 重试机制
func withRetry(client *http.Client, req *http.Request, maxRetries int) (*http.Response, error) {
    for i := 0; i < maxRetries; i++ {
        resp, err := client.Do(req)
        if err != nil {
            continue
        }
        if resp.StatusCode < 500 {
            return resp, nil
        }
        resp.Body.Close()
    }
    return nil, fmt.Errorf("max retries exceeded")
}
```

#### HTTP服务器中间件
```go
func middleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        
        // 日志
        log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
        
        // 压缩
        w.Header().Set("Content-Encoding", "gzip")
        
        next.ServeHTTP(w, r)
    })
}

// 链路追踪中间件
func tracing(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        ctx := r.Context()
        
        traceID := uuid.New().String()
        ctx = context.WithValue(ctx, "trace_id", traceID)
        
        w.Header().Set("X-Trace-ID", traceID)
        
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}
```

### 4.3 泛型编程（Go 1.18+）

#### 泛型函数
```go
// 类型参数
func Map[T any, R any](slice []T, fn func(T) R) []R {
    result := make([]R, len(slice))
    for i, v := range slice {
        result[i] = fn(v)
    }
    return result
}

// 使用泛型
numbers := []int{1, 2, 3}
strings := Map(numbers, func(n int) string {
    return strconv.Itoa(n)
})
```

#### 泛型类型
```go
// 泛型栈
type Stack[T any] struct {
    items []T
}

func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

func (s *Stack[T]) Pop() (T, bool) {
    if len(s.items) == 0 {
        var zero T
        return zero, false
    }
    item := s.items[len(s.items)-1]
    s.items = s.items[:len(s.items)-1]
    return item, true
}

// 泛型约束
type Number interface {
    int | int32 | int64 | float32 | float64
}

func Sum[T Number](values []T) T {
    var sum T
    for _, v := range values {
        sum += v
    }
    return sum
}
```

---

## 第五章：Go项目实践

### 5.1 Go项目结构

#### 标准项目布局
```
myproject/
├── cmd/
│   └── myapp/
│       └── main.go
├── internal/
│   ├── api/
│   ├── config/
│   ├── handler/
│   ├── model/
│   ├── repository/
│   └── service/
├── pkg/
│   ├── logger/
│   ├── cache/
│   └── utils/
├── test/
│   └── integration/
├── Dockerfile
├── docker-compose.yml
├── go.mod
├── go.sum
└── Makefile
```

### 5.2 依赖注入

#### 使用wire
```go
// wire.go
//go:build wireinject

package main

import (
    "github.com/google/wire"
)

func InitializeConfig() *Config {
    wire.Build(
        NewDatabase,
        NewCache,
        NewService,
        wire.Struct(new(Config), "*"),
    )
    return nil
}

// provider.go
func NewDatabase(cfg *Config) *Database {
    return &Database{URL: cfg.DBURL}
}
```

### 5.3 测试最佳实践

#### Table-Driven测试
```go
func TestAdd(t *testing.T) {
    tests := []struct {
        name     string
        a, b     int
        expected int
    }{
        {"positive numbers", 2, 3, 5},
        {"negative numbers", -1, -2, -3},
        {"mixed numbers", -5, 10, 5},
        {"zeros", 0, 0, 0},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := Add(tt.a, tt.b)
            if result != tt.expected {
                t.Errorf("Add(%d, %d) = %d, want %d", tt.a, tt.b, result, tt.expected)
            }
        })
    }
}
```

#### Mock测试
```go
// 使用mockgen生成mock
//go:generate mockgen -destination=repository_mock.go . Repository

type Repository interface {
    GetUser(id int) (*User, error)
    CreateUser(user *User) error
}

func TestService_CreateUser(t *testing.T) {
    mockRepo := NewMockRepository(ctrl)
    
    mockRepo.EXPECT().
        CreateUser(gomock.Any()).
        Return(nil)
    
    svc := NewService(mockRepo)
    err := svc.CreateUser(&User{Name: "test"})
    
    assert.NoError(t, err)
}
```

---

## 参考资源

### 官方文档
- Go语言官方文档：golang.org/doc
- Go语言规范：golang.org/ref/spec
- Go标准库：pkg.go.dev

### 进阶学习
- 《Effective Go》：golang.org/doc/effective_go
- 《Go语言并发模式》：golang.org/ref/mem
- 《Go Modules》：golang.org/ref/mod

### 工具推荐
- golangci-lint：代码检查
- Delve：调试器
- Air：热重载开发

---

*本知识文件最后更新：2026-02-07*
*涵盖Go并发编程、性能优化、设计模式*
