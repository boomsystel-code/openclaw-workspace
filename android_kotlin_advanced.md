# Android Kotlin高级开发

## 第一章：Kotlin核心特性

### 1.1 协程深入理解

#### 协程基础
```kotlin
// 启动协程
GlobalScope.launch {
    // 异步执行
    delay(1000L)
    println("Hello from coroutine")
}

// 结构化并发
fun main() = runBlocking {
    launch {
        delay(1000L)
        println("Task 1 completed")
    }
    
    launch {
        delay(500L)
        println("Task 2 completed")
    }
    
    println("Main program continues")
    delay(2000L) // 等待所有协程完成
}
```

#### 协程上下文与调度器
```kotlin
// 调度器类型
Dispatchers.Main      // UI线程
Dispatchers.IO        // IO操作
Dispatchers.Default   // CPU密集型
Dispatchers.Unconfined // 不指定

// 指定调度器
launch(Dispatchers.IO) {
    val data = withContext(Dispatchers.IO) {
        // IO操作
        database.query()
    }
    
    // 切换回Main更新UI
    withContext(Dispatchers.Main) {
        updateUI(data)
    }
}

// 自定义调度器
val myDispatcher = Executors.newFixedThreadPool(4).asCoroutineDispatcher()

// Job与协程取消
val job = CoroutineScope(Dispatchers.Default).launch {
    try {
        repeat(100) { i ->
            delay(100)
            if (isActive) {  // 检查是否取消
                println("Task $i")
            }
        }
    } finally {
        // 清理操作
        println("Cleanup")
    }
}

delay(500)
job.cancel()  // 取消协程
job.join()    // 等待完成
```

### 1.2 Flow响应式流

```kotlin
// 创建Flow
fun numbersFlow(): Flow<Int> = flow {
    for (i in 1..10) {
        delay(100)
        emit(i)
    }
}

// 收集Flow
fun main() = runBlocking {
    numbersFlow()
        .filter { it % 2 == 0 }
        .map { it * it }
        .collect { println(it) }
}

// Flow操作符
fun userFlow(): Flow<User> = userDao.getAllUsers()
    .map { list -> list.map { it.toDomain() } }
    .catch { e -> emitAll(fallbackFlow) }

fun main() = runBlocking {
    userFlow()
        .debounce(300)
        .distinctUntilChanged()
        .collect { users ->
            adapter.submitList(users)
        }
}

// StateFlow & SharedFlow
class ViewModel {
    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()
    
    private val _events = MutableSharedFlow<Event>()
    val events: SharedFlow<Event> = _events.asSharedFlow()
    
    fun processIntent(intent: Intent) {
        when (intent) {
            is Intent.LoadData -> loadData()
            is Intent.Refresh -> refresh()
        }
    }
}
```

---

## 第二章：Android架构组件

### 2.1 Hilt依赖注入

```kotlin
// Module
@Module
@InstallIn(SingletonComponent::class)
object AppModule {
    
    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): AppDatabase {
        return Room.databaseBuilder(
            context,
            AppDatabase::class.java,
            "app_database"
        ).build()
    }
    
    @Provides
    @Singleton
    fun provideUserRepository(database: AppDatabase): UserRepository {
        return UserRepositoryImpl(database.userDao())
    }
}

// ViewModel注入
@HiltViewModel
class UserListViewModel @Inject constructor(
    private val userRepository: UserRepository
) : ViewModel() {
    
    private val _uiState = MutableStateFlow<UserListState>(UserListState.Loading)
    val uiState: StateFlow<UserListState> = _uiState.asStateFlow()
    
    init {
        loadUsers()
    }
    
    private fun loadUsers() = viewModelScope.launch {
        _uiState.value = UserListState.Loading
        try {
            val users = userRepository.getUsers()
            _uiState.value = UserListState.Success(users)
        } catch (e: Exception) {
            _uiState.value = UserListState.Error(e.message)
        }
    }
}
```

### 2.2 Room数据库

```kotlin
// Entity
@Entity(tableName = "users")
data class UserEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    @ColumnInfo(name = "name")
    val name: String,
    @ColumnInfo(name = "email")
    val email: String,
    @ColumnInfo(name = "created_at")
    val createdAt: Long = System.currentTimeMillis()
)

// DAO
@Dao
interface UserDao {
    @Query("SELECT * FROM users ORDER BY created_at DESC")
    fun getAllUsers(): Flow<List<UserEntity>>
    
    @Query("SELECT * FROM users WHERE id = :id")
    suspend fun getUserById(id: Long): UserEntity?
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertUser(user: UserEntity): Long
    
    @Update
    suspend fun updateUser(user: UserEntity)
    
    @Delete
    suspend fun deleteUser(user: UserEntity)
    
    @Query("DELETE FROM users")
    suspend fun deleteAllUsers()
}

// Database
@Database(
    entities = [UserEntity::class],
    version = 1,
    exportSchema = false
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun userDao(): UserDao
}
```

---

## 第三章：Jetpack Compose

### 3.1 Compose基础

```kotlin
@Composable
fun UserCard(user: User) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {
                AsyncImage(
                    model = user.avatarUrl,
                    contentDescription = "User avatar",
                    modifier = Modifier
                        .size(60.dp)
                        .clip(CircleShape)
                )
                
                Spacer(modifier = Modifier.width(16.dp))
                
                Column {
                    Text(
                        text = user.name,
                        style = MaterialTheme.typography.titleMedium
                    )
                    Text(
                        text = user.email,
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Button(
                onClick = { /* Handle click */ },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("View Profile")
            }
        }
    }
}
```

### 3.2 状态管理

```kotlin
// Remember vs RememberSaveable
@Composable
fun Counter() {
    var count by remember { mutableIntStateOf(0) }
    
    Text("Count: $count")
    
    Button(onClick = { count++ }) {
        Text("Increment")
    }
}

// 状态提升
@Composable
fun ParentScreen() {
    var name by rememberSaveable { mutableStateOf("") }
    
    ChildScreen(
        name = name,
        onNameChange = { name = it }
    )
}

@Composable
fun ChildScreen(
    name: String,
    onNameChange: (String) -> Unit
) {
    OutlinedTextField(
        value = name,
        onValueChange = onNameChange
    )
}

// 副作用
@Composable
fun DataScreen(viewModel: DataViewModel = hiltViewModel()) {
    val data by viewModel.dataFlow.collectAsStateWithLifecycle()
    
    LaunchedEffect(key1 = data) {
        // 副作用：当data变化时执行
        analytics.trackViewData(data)
    }
    
    DisposableEffect(Unit) {
        onDispose {
            // 清理
            analytics.stopTracking()
        }
    }
}
```

---

## 第四章：性能优化

### 4.1 内存优化

```kotlin
// 避免内存泄漏
class MyFragment : Fragment() {
    private var viewModel: MyViewModel? = null
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewModel = ViewModelProvider(this)[MyViewModel::class.java]
    }
    
    override fun onDestroyView() {
        super.onDestroyView()
        viewModel = null // 清理引用
    }
}

// 使用弱引用
class Cache {
    private val cache = WeakHashMap<String, Any>()
    
    fun put(key: String, value: Any) {
        cache[key] = value
    }
    
    fun get(key: String): Any? {
        return cache[key]
    }
}
```

### 4.2 启动优化

```kotlin
// 延迟初始化
class MyApplication : Application() {
    private val database: AppDatabase by lazy {
        Room.databaseBuilder(
            applicationContext,
            AppDatabase::class.java,
            "app_database"
        ).build()
    }
    
    override fun onCreate() {
        super.onCreate()
        // 初始化必要组件
        initCrashReporter()
        initLogger()
    }
}

// 按需初始化
// build.gradle.kts
android {
    defaultConfig {
        vectorDrawables {
            useSupportLibrary = true
        }
    }
    
    buildTypes {
        debug {
            isDebuggable = true
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
}
```

---

## 参考资源

### 官方文档
- Kotlin: kotlinlang.org/docs
- Android: developer.android.com/docs
- Jetpack Compose: developer.android.com/jetpack/compose

### 进阶资源
- Android Developers Blog
- Google I/O Talks
- Ray Wenderlich Android

### 开源库
- Retrofit：网络请求
- Glide/Picasso：图片加载
- Moshi：JSON解析

---

*本知识文件最后更新：2026-02-07*
