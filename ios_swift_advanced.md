# iOS与Swift高级开发

## 第一章：Swift高级特性

### 1.1 泛型编程

#### 泛型函数
```swift
// 泛型函数
func swapValues<T>(_ a: inout T, _ b: inout T) {
    let temp = a
    a = b
    b = temp
}

// 使用
var x = 5
var y = 10
swapValues(&x, &y)

// 泛型约束
protocol Summable {
    static func +(lhs: Self, rhs: Self) -> Self
}

extension Int: Summable {}
extension Double: Summable {}

func sum<T: Summable>(_ values: [T]) -> T {
    return values.reduce(.zero, +)
}

// 关联类型
protocol Container {
    associatedtype Item
    mutating func append(_ item: Item)
    var count: Int { get }
    subscript(i: Int) -> Item { get }
}

// where约束
func makeArray<Item>(repeating item: Item, numberOfTimes: Int) -> [Item] {
    return [Item](repeating: item, count: numberOfTimes)
}

extension Array: Container where Element: Equatable {
    // Element必须可比较
}
```

### 1.2 协议与面向协议编程

```swift
// 协议扩展
protocol Drawable {
    func draw()
}

extension Drawable {
    func draw() {
        print("Drawing")
    }
    
    var backgroundColor: String { get }
}

// 协议组合
protocol Named {
    var name: String { get }
}

protocol Aged {
    var age: Int { get }
}

struct Person: Named, Aged {
    var name: String
    var age: Int
}

func describe<T: Named & Aged>(_ entity: T) {
    print("\(entity.name) is \(entity.age) years old")
}

// 可失败初始化器
struct Product {
    let id: String
    let price: Double
    
    init?(id: String, price: Double) {
        guard price >= 0 else { return nil }
        self.id = id
        self.price = price
    }
}
```

---

## 第二章：Swift并发编程

### 2.1 Async/Await

```swift
// 异步函数
func fetchUser(id: Int) async -> User {
    let (data, _) = try await URLSession.shared.data(from: url)
    return try JSONDecoder().decode(User.self, from: data)
}

// 并行执行
async let user = fetchUser(id: 1)
async let profile = fetchProfile(id: 1)

let (userResult, profileResult) = await (user, profile)

// 任务组
async withTaskGroup(of: String.self) { group in
    for i in 1...5 {
        group.addTask {
            "Task \(i)"
        }
    }
    
    for await result in group {
        print(result)
    }
}

// 取消任务
Task {
    for await line in FileHandle.standardInput.lines {
        if Task.isCancelled { break }
        process(line)
    }
}
```

### 2.2 Actor

```swift
// Actor隔离
actor BankAccount {
    private var balance: Double
    
    init(initialBalance: Double) {
        self.balance = initialBalance
    }
    
    func deposit(_ amount: Double) {
        balance += amount
    }
    
    func transfer(to account: BankAccount, amount: Double) async {
        guard balance >= amount else { return }
        balance -= amount
        await account.deposit(amount)
    }
}

// Sendable协议
struct Message: Sendable {
    let content: String
    let timestamp: Date
}

// MainActor
@MainActor
class ViewModel {
    @Published var data: [Item] = []
    
    func loadData() async {
        data = await fetchItems()
    }
}
```

---

## 第三章：iOS架构设计

### 3.1 MVVM架构

```swift
import SwiftUI
import Combine

// Model
struct User: Identifiable, Codable {
    let id: Int
    let name: String
    let email: String
}

// ViewModel
@MainActor
class UserListViewModel: ObservableObject {
    @Published var users: [User] = []
    @Published var isLoading = false
    @Published var error: Error?
    
    private let service: UserServiceProtocol
    
    init(service: UserServiceProtocol = UserService()) {
        self.service = service
    }
    
    func loadUsers() async {
        isLoading = true
        error = nil
        
        do {
            users = try await service.fetchUsers()
        } catch {
            self.error = error
        }
        
        isLoading = false
    }
    
    func deleteUser(at offsets: IndexSet) async {
        for index in offsets {
            let user = users[index]
            do {
                try await service.deleteUser(id: user.id)
                users.remove(atOffsets: offsets)
            } catch {
                self.error = error
            }
        }
    }
}

// View
struct UserListView: View {
    @StateObject private var viewModel = UserListViewModel()
    
    var body: some View {
        Group {
            if viewModel.isLoading && viewModel.users.isEmpty {
                ProgressView("Loading...")
            } else if let error = viewModel.error {
                ErrorView(error: error) {
                    Task { await viewModel.loadUsers() }
                }
            } else {
                List {
                    ForEach(viewModel.users) { user in
                        UserRowView(user: user)
                    }
                    .onDelete { offsets in
                        Task {
                            await viewModel.deleteUser(at: offsets)
                        }
                    }
                }
                .refreshable {
                    await viewModel.loadUsers()
                }
            }
        }
        .task {
            await viewModel.loadUsers()
        }
    }
}
```

### 3.2 依赖注入

```swift
// 协议定义
protocol NetworkServiceProtocol {
    func fetch<T: Decodable>(_ type: T.Type, from url: URL) async throws -> T
}

struct NetworkService: NetworkServiceProtocol {
    func fetch<T: Decodable>(_ type: T.Type, from url: URL) async throws -> T {
        let (data, _) = try await URLSession.shared.data(from: url)
        return try JSONDecoder().decode(T.self, from: data)
    }
}

// 依赖注入容器
final class DIContainer {
    static let shared = DIContainer()
    
    private var services: [String: Any] = [:]
    
    func register<T>(_ service: T, for type: T.Type) {
        let key = "\(type)"
        services[key] = service
    }
    
    func resolve<T>(_ type: T.Type) -> T? {
        let key = "\(type)"
        return services[key] as? T
    }
}

// 使用
DIContainer.shared.register(NetworkService(), for: NetworkServiceProtocol.self)

class MyViewModel {
    private let networkService: NetworkServiceProtocol
    
    init(networkService: NetworkServiceProtocol = DIContainer.shared.resolve(NetworkServiceProtocol.self)!) {
        self.networkService = networkService
    }
}
```

---

## 第四章：Swift性能优化

### 4.1 内存管理

```swift
// 引用类型循环检测
class Node {
    var value: Int
    weak var parent: Node?
    var children: [Node] = []
    
    deinit {
        print("Node \(value) deinit")
    }
}

// 使用weak/unowned打破循环
class Tree {
    var root: Node
    
    init() {
        root = Node(value: 0)
    }
}

// 值类型使用
// Swift中Array、String、Struct是值类型
// 自动处理内存，避免循环引用
```

### 4.2 性能分析

```swift
import Instruments

// 性能标记
os_signpost(.begin, log: .default, name: "ExpensiveOperation")
// 耗时操作
os_signpost(.end, log: .default, name: "ExpensiveOperation")

// 内存使用监控
let memoryUsage = TaskInfo()
print("Memory used: \(memoryUsage.memoryUsed)")

// Instruments分析
// 1. Time Profiler：CPU时间分析
// 2. Allocations：内存分配
// 3. Leaks：内存泄漏
// 4. Core Animation：UI渲染性能
```

---

## 参考资源

### 官方文档
- Swift: docs.swift.org
- SwiftUI: developer.apple.com/swiftui
- Combine: developer.apple.com/documentation/combine

### 进阶资源
- Swift by Sundell
- Hacking with Swift
- Ray Wenderlich

### 开源库
- Alamofire：网络请求
- Kingfisher：图片缓存
- SnapKit：自动布局

---

*本知识文件最后更新：2026-02-07*
