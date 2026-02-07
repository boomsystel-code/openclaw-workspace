# Flutter跨平台开发实战

## 第一章：Flutter核心概念

### 1.1 Widget体系

#### StatelessWidget vs StatefulWidget
```dart
// 无状态组件
class MyStatelessWidget extends StatelessWidget {
  final String title;
  
  const MyStatelessWidget({
    super.key,
    required this.title,
  });
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(title),
      ),
      body: Center(
        child: Text(
          'Hello Flutter',
          style: Theme.of(context).textTheme.headlineMedium,
        ),
      ),
    );
  }
}

// 有状态组件
class CounterWidget extends StatefulWidget {
  const CounterWidget({super.key});
  
  @override
  State<CounterWidget> createState() => _CounterWidgetState();
}

class _CounterWidgetState extends State<CounterWidget> {
  int _counter = 0;
  
  void _increment() {
    setState(() {
      _counter++;
    });
  }
  
  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text('Count: $_counter'),
        ElevatedButton(
          onPressed: _increment,
          child: const Text('Increment'),
        ),
      ],
    );
  }
}
```

### 1.2 布局组件

```dart
// 常用布局组件
class LayoutDemo extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // 行布局
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Container(
                    width: 100,
                    height: 100,
                    color: Colors.blue,
                    child: const Center(child: Text('Box 1')),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Container(
                      height: 100,
                      color: Colors.green,
                      child: const Center(child: Text('Expanded')),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              
              // 网格布局
              GridView.builder(
                shrinkWrap: true,
                gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: 3,
                  childAspectRatio: 1.0,
                  crossAxisSpacing: 8,
                  mainAxisSpacing: 8,
                ),
                itemCount: 20,
                itemBuilder: (context, index) {
                  return Container(
                    color: Colors.primaries[index % Colors.primaries.length],
                    child: Center(
                      child: Text('Item $index'),
                    ),
                  );
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}
```

---

## 第二章：Dart异步编程

### 2.1 Future与Stream

```dart
// Future使用
Future<String> fetchUserData() async {
  // 模拟网络请求
  await Future.delayed(const Duration(seconds: 1));
  return '{"name": "John", "age": 30}';
}

// 使用FutureBuilder
class UserProfile extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return FutureBuilder<String>(
      future: fetchUserData(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const CircularProgressIndicator();
        }
        
        if (snapshot.hasError) {
          return Text('Error: ${snapshot.error}');
        }
        
        final userData = jsonDecode(snapshot.data!);
        return Text('User: ${userData['name']}');
      },
    );
  }
}

// Stream使用
Stream<int> numberStream() async* {
  for (int i = 1; i <= 10; i++) {
    await Future.delayed(const Duration(milliseconds: 500));
    yield i;
  }
}

// StreamBuilder
class StreamCounter extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return StreamBuilder<int>(
      stream: numberStream(),
      builder: (context, snapshot) {
        if (snapshot.hasData) {
          return Text('Number: ${snapshot.data}');
        }
        return const CircularProgressIndicator();
      },
    );
  }
}

// Stream控制器
class StreamControllerDemo {
  final _controller = StreamController<int>();
  
  Stream<int> get stream => _controller.stream;
  
  void addData(int value) {
    _controller.add(value);
  }
  
  void dispose() {
    _controller.close();
  }
}
```

### 2.2 Isolates并发

```dart
// 创建Isolate
Future<void> heavyComputation() async {
  final isolate = await Isolate.spawn(computeTask, 1000000);
  
  // 等待结果
  final result = await ReceivePort().first;
  print('Result: $result');
  
  isolate.kill();
}

// 独立函数作为入口点
void computeTask(int count) {
  int sum = 0;
  for (int i = 0; i < count; i++) {
    sum += i;
  }
  Isolate.exit(sum);
}

// 使用compute()简化
final result = await compute(calculateFibonacci, 40);

int calculateFibonacci(int n) {
  if (n <= 1) return n;
  return calculateFibonacci(n - 1) + calculateFibonacci(n - 2);
}
```

---

## 第三章：状态管理

### 3.1 Provider

```dart
// Model
class CounterModel extends ChangeNotifier {
  int _count = 0;
  
  int get count => _count;
  
  void increment() {
    _count++;
    notifyListeners();
  }
  
  void reset() {
    _count = 0;
    notifyListeners();
  }
}

// 注册Provider
void main() {
  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => CounterModel()),
        ChangeNotifierProvider(create: (_) => UserModel()),
      ],
      child: const MyApp(),
    ),
  );
}

// 使用Consumer
class CounterDisplay extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Consumer<CounterModel>(
      builder: (context, counter, child) {
        return Text('Count: ${counter.count}');
      },
    );
  }
}

// Selector（性能优化）
Selector<CounterModel, int>(
  selector: (context, model) => model.count,
  builder: (context, count, child) {
    return Text('Count: $count');
  },
);
```

### 3.2 Riverpod

```dart
// 定义Provider
final counterProvider = ChangeNotifierProvider<CounterNotifier>((ref) {
  return CounterNotifier();
});

class CounterNotifier extends ChangeNotifier {
  int _count = 0;
  int get count => _count;
  
  void increment() {
    _count++;
    notifyListeners();
  }
}

// 使用ConsumerWidget
class CounterScreen extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final counter = ref.watch(counterProvider);
    
    return Scaffold(
      body: Center(
        child: Text('Count: ${counter.count}'),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () => ref.read(counterProvider).increment(),
        child: const Icon(Icons.add),
      ),
    );
  }
}

// FutureProvider
final userProvider = FutureProvider<User>((ref) async {
  final api = ref.watch(apiProvider);
  return api.fetchUser();
});

// StreamProvider
final messagesProvider = StreamProvider<List<Message>>((ref) {
  return ref.watch(messageServiceProvider).streamMessages();
});
```

---

## 第四章：性能优化

### 4.1 渲染性能

```dart
// 避免不必要的重建
class OptimizedList extends StatelessWidget {
  const OptimizedList({super.key});
  
  @override
  Widget build(BuildContext context) {
    return ListView.builder(
      itemCount: 1000,
      itemBuilder: (context, index) {
        // 使用const构造函数
        return ListTile(
          key: Key('item_$index'),
          title: Text('Item $index'),
          leading: const Icon(Icons.circle, key: Key('icon_$index')),
        );
      },
    );
  }
}

// 使用RepaintBoundary
class SeparateAnimationWidget extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        RepaintBoundary(
          child: Container(
            // 动画Widget放这里
            child: const AnimationWidget(),
          ),
        ),
        const StaticWidget(),
      ],
    );
  }
}

// 图片优化
Image.network(
  url,
  loadingBuilder: (context, child, progress) {
    if (progress == null) return child;
    return CircularProgressIndicator();
  },
  errorBuilder: (context, error, stack) {
    return const Icon(Icons.error);
  },
)
```

### 4.2 内存管理

```dart
// 及时释放资源
class ResourceManager {
  List<StreamSubscription> _subscriptions = [];
  List<ChangeNotifier> _notifiers = [];
  
  void dispose() {
    for (var sub in _subscriptions) {
      sub.cancel();
    }
    _subscriptions.clear();
    
    for (var notifier in _notifiers) {
      notifier.dispose();
    }
    _notifiers.clear();
  }
}

// 使用valueKey保持状态
ListView.builder(
  itemCount: items.length,
  itemBuilder: (context, index) {
    return ListTile(
      key: ValueKey(items[index].id),
      title: Text(items[index].name),
    );
  },
)

// 图片缓存
ImageCache(
  maximumSize: 100,  // 最多100张图片
  maximumMemoryCache: 50 * 1024 * 1024,  // 50MB
)
```

---

## 参考资源

### 官方文档
- Flutter: docs.flutter.dev
- Dart: dart.dev/docs
- pub.dev/packages

### 进阶资源
- Flutter YouTube Channel
- Flutter Community
- Invertase Talks

### 工具推荐
- Dart DevTools：性能分析
- Flutter Inspector：UI调试
- Provider/Riverpod：状态管理

---

*本知识文件最后更新：2026-02-07*
