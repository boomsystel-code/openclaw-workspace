# System Design 系统设计完全指南

## 第一章：系统设计基础

### 1.1 核心概念

#### 什么是系统设计？
系统设计是构建可扩展、高可用、高效软件系统的过程。它涉及：
- **架构决策**：选择合适的技术栈和模式
- **数据管理**：存储、缓存、数据库设计
- **可扩展性**：水平/垂直扩展策略
- **可靠性**：容错、冗余、恢复机制
- **性能**：延迟、吞吐量优化

#### 系统设计面试要点
1. **需求澄清**：明确功能和非功能需求
2. **高层次设计**：整体架构图
3. **核心组件设计**：详细设计关键模块
4. **扩展性考虑**：如何处理增长
5. **权衡分析**：利弊权衡

### 1.2 可扩展性原则

#### 水平扩展 vs 垂直扩展

**水平扩展（Horizontal Scaling）**
```
优点：
- 无限扩展
- 容错性好
- 按需扩展
- 成本可控

缺点：
- 分布式复杂性
- 数据一致性挑战
- 运维复杂度
```

**垂直扩展（Vertical Scaling）**
```
优点：
- 简单直接
- 无需修改代码
- 强一致性

缺点：
- 单点故障
- 扩展有上限
- 成本指数增长
```

#### 扩展性度量
- **吞吐量（Throughput）**：QPS/TPS
- **延迟（Latency）**：响应时间（P50/P95/P99）
- **可用性（Availability）**：正常运行时间（99.9%, 99.99%）
- **可扩展性（Scalability）**：扩展效率

---

## 第二章：数据存储设计

### 2.1 数据库选择

#### SQL vs NoSQL

**SQL数据库（关系型）**
```
适用场景：
- 强一致性要求
- 复杂查询
- ACID事务
- 固定Schema

代表：
- PostgreSQL
- MySQL
- MariaDB
- Cloud SQL

最佳实践：
- 正确使用索引
- 查询优化
- 分库分表
- 读写分离
```

**NoSQL数据库**

| 类型 | 特点 | 代表 | 使用场景 |
|------|------|------|---------|
| 文档型 | 灵活Schema | MongoDB | 内容管理 |
| 键值型 | 极致性能 | Redis | 缓存、会话 |
| 列式 | 列存储分析 | Cassandra | 时序数据 |
| 图数据库 | 关系查询 | Neo4j | 社交网络 |

#### 缓存策略

**多级缓存**
```
L1：本地缓存（如Guava Cache）
- 容量：MB级别
- 延迟：ns级别
- 适用：热点数据

L2：分布式缓存（Redis）
- 容量：GB级别
- 延迟：μs级别
- 适用：共享数据

L3：数据库缓存
- 容量：TB级别
- 延迟：ms级别
- 适用：持久化存储
```

**缓存模式**
```python
# Cache-Aside（旁路缓存）
def get_user(user_id):
    user = cache.get(user_id)
    if user:
        return user
    
    user = db.query(user_id)
    cache.set(user_id, user)
    return user

def update_user(user_id, data):
    db.update(user_id, data)
    cache.delete(user_id)  # 先更新DB，再删除缓存

# Write-Through（写穿透）
def save_user(user):
    cache.set(user.id, user)  # 同步写入缓存
    db.save(user)              # 写入数据库

# Write-Behind（写回）
def save_user(user):
    cache.set(user.id, user)  # 只写缓存
    async db.save(user)        # 异步落盘
```

### 2.2 数据模型设计

#### ER建模
```
用户（User）
├── 订单（Order）1:N
│   └── 订单项（OrderItem）N:1
└── 地址（Address）1:N

规范化的好处：
- 减少数据冗余
- 保证数据一致性
- 简化维护

反规范化的场景：
- 读多写少
- 需要高性能查询
- 容忍数据冗余
```

#### 分片策略

**哈希分片**
```python
# 一致性哈希
def get_shard(key):
    hash_value = md5(key)
    return hash_value % num_shards

# 虚拟节点
for i in range(VIRTUAL_NODES):
    virtual_key = f"{key}#{i}"
    shard = hash(virtual_key) % num_shards
```

**范围分片**
```python
# 按时间分片
def get_shard(time):
    if time < 2020:
        return "archive"
    elif time < 2024:
        return "recent"
    return "current"
```

---

## 第三章：分布式系统设计

### 3.1 CAP定理

#### CAP权衡
```
CAP：
- Consistency（一致性）：所有节点看到相同数据
- Availability（可用性）：每个请求都有响应
- Partition Tolerance（分区容错）：系统继续运行

只能同时满足两个：
CP（一致+分区容错）：Zookeeper, etcd
AP（可用+分区容错）：Cassandra, DynamoDB
CA（一致+可用）：单节点数据库
```

#### BASE理论
```
BASE：
- Basically Available：基本可用
- Soft State：软状态
- Eventual Consistency：最终一致

实践：
- 乐观锁
- 版本控制
- 冲突解决
- 重试机制
```

### 3.2 一致性模型

#### 强一致性
```
线性一致性（Linearizability）：
- 所有操作看起来是原子的
- 实时性保证
- 实现成本高

实现方式：
- 2PC（两阶段提交）
- Paxos/Raft共识
- 分布式锁
```

#### 最终一致性
```
最终一致的保证：
- 更新最终会传播
- 不要求实时一致
- 更高的可用性

应用场景：
- 社交媒体点赞
- 计数器
- 排行榜
```

### 3.3 分布式事务

#### 两阶段提交（2PC）
```
阶段1：准备（Prepare）
- 协调者询问所有参与者
- 参与者执行但不提交
- 参与者返回YES/NO

阶段2：提交（Commit）
- 所有返回YES：发送提交
- 任一返回NO：发送回滚
```

#### TCC（Try-Confirm-Cancel）
```
Try：预留资源
Confirm：确认使用
Cancel：取消释放

适用场景：
- 分布式事务
- 跨银行转账
- 订单系统
```

---

## 第四章：高可用设计

### 4.1 负载均衡

#### 算法类型
```
1. 轮询（Round Robin）
   - 简单公平
   - 适用场景：后端相同

2. 加权轮询（Weighted Round Robin）
   - 按权重分配
   - 适用场景：服务器性能不同

3. 最少连接（Least Connections）
   - 选择连接数最少
   - 适用场景：请求耗时不同

4. IP哈希（IP Hash）
   - 同一IP固定服务器
   - 适用场景：需要会话保持

5. 一致性哈希（Consistent Hash）
   - 最小化迁移
   - 适用场景：缓存分片
```

#### 健康检查
```
主动健康检查：
- TCP检查
- HTTP检查
- 自定义脚本

检查策略：
- 检查频率
- 超时时间
- 不健康阈值
- 健康阈值
```

### 4.2 容错设计

#### 断路器模式
```python
class CircuitBreaker:
    def __init__(self, threshold=5, timeout=60):
        self.failure_count = 0
        self.success_count = 0
        self.threshold = threshold
        self.timeout = timeout
        self.state = "CLOSED"
        self.last_failure = None
    
    def call(self, func):
        if self.state == "OPEN":
            if time.now() - self.last_failure > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitOpenError()
        
        try:
            result = func()
            self.on_success()
            return result
        except Exception:
            self.on_failure()
            raise
    
    def on_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.threshold:
            self.state = "OPEN"
            self.last_failure = time.now()
    
    def on_success(self):
        self.success_count += 1
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
```

#### 重试策略
```python
def retry(func, max_retries=3, delay=0.1, backoff=2):
    for i in range(max_retries):
        try:
            return func()
        except Exception as e:
            if i == max_retries - 1:
                raise
            sleep_time = delay * (backoff ** i)
            time.sleep(sleep_time)
```

### 4.3 高可用架构

#### 多活设计
```
数据中心级别：
- 同城多活
- 异地多活
- 全球部署

挑战：
- 数据同步延迟
- DNS配置
- 流量调度
```

---

## 第五章：系统设计案例

### 5.1 URL短链服务

#### 核心需求
```
功能需求：
- 短链生成
- 短链重定向
- 点击统计

非功能需求：
- 高可用
- 低延迟
- 存储效率
```

#### 设计方案
```
1. 长链转短链算法
   - Base62编码
   - 6位字符：62^6 ≈ 568亿
   
2. 数据存储
   - Redis：短链->长链映射
   - MySQL：持久化存储
   
3. 生成器服务
   - 预生成ID
   - 分布式锁
   
4. 重定向服务
   - Nginx直接跳转
   - 301重定向（SEO友好）
```

### 5.2 消息队列设计

#### 核心组件
```
生产者（Producer）
- 发送消息
- 消息确认

消费者（Consumer）
- 接收消息
- 消息确认

Broker
- 消息存储
- 分区管理
- 副本同步
```

#### 核心功能
```
消息持久化
- 顺序写入
- WAL（Write Ahead Log）
- 定期刷盘

消息语义
- At Most Once
- At Least Once
- Exactly Once

消费模型
- Push vs Pull
- 负载均衡
- 消费者组
```

### 5.3 分布式ID生成

#### 雪花算法（Snowflake）
```
64位ID结构：
- 1位：符号位（0）
- 41位：时间戳（毫秒）
- 10位：机器ID
- 12位：序列号

优点：
- 时间有序
- 分布式唯一
- 高性能

问题：
- 时钟回拨
- 机器ID分配
```

#### UUID
```
128位：
- 高度随机
- 局部有序

变体：
- UUID v1：时间+MAC
- UUID v4：纯随机
```

---

## 第六章：设计模式在系统设计中的应用

### 6.1 分布式设计模式

#### 代理模式
```
应用场景：
- API网关
- 负载均衡器
- 缓存代理

实现：
- 正向代理
- 反向代理
```

#### 事件驱动模式
```
应用场景：
- 异步处理
- 解耦组件
- 实时通知

技术选型：
- Kafka
- RabbitMQ
- Pulsar
```

#### CQRS模式
```
命令查询职责分离：
- Command：写操作
- Query：读操作

优势：
- 读写分离优化
- 灵活的数据模型
- 更好的扩展性
```

---

## 参考资源

### 书籍推荐
- 《Designing Data-Intensive Applications》- Martin Kleppmann
- 《System Design Interview》- Alex Xu
- 《Site Reliability Engineering》
- 《The Art of Scalability》

### 在线资源
- System Primer
- High Scalability Blog
- GitHub Engineering Blog
- Netflix Tech Blog

### 实践平台
- LeetCode System Design
- Exponent
- Pramp System Design

---

*本知识文件最后更新：2026-02-07*
*涵盖系统设计基础、数据存储、分布式系统、高可用架构*
