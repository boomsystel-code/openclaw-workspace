# 关系型数据库高级技术与实战

## 第一章：数据库性能优化

### 1.1 SQL查询优化

#### 执行计划分析
```sql
-- 使用EXPLAIN分析查询
EXPLAIN ANALYZE
SELECT u.username, o.order_id, o.total
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE o.created_at > '2024-01-01'
ORDER BY o.total DESC;

-- 分析结果关注点
-- 1. type: ALL（全表扫描）-> ref（索引）-> const（最优）
-- 2. rows: 扫描行数
-- 3. key: 使用的索引
-- 4. Extra: Using filesort, Using temporary
```

#### 索引优化策略
```sql
-- 复合索引：按查询频率排序
CREATE INDEX idx_user_order ON orders(user_id, created_at, status);

-- 覆盖索引：包含查询所需的所有列
CREATE INDEX idx_order_cover ON orders(status, created_at)
  INCLUDE (total, user_id);

-- 部分索引：只索引热点数据
CREATE INDEX idx_active_orders ON orders(created_at DESC)
  WHERE status = 'active';

-- 表达式索引：加速计算列查询
CREATE INDEX idx_user_lower ON users((LOWER(email)));
```

### 1.2 查询重写技巧

#### 子查询优化
```sql
-- IN子查询 -> JOIN
-- 低效
SELECT * FROM users
WHERE id IN (SELECT user_id FROM orders WHERE amount > 1000);

-- 高效
SELECT DISTINCT u.*
FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE o.amount > 1000;

-- EXISTS替代IN
-- EXISTS在找到匹配后立即停止
SELECT * FROM users u
WHERE EXISTS (
  SELECT 1 FROM orders o 
  WHERE o.user_id = u.id AND o.amount > 1000
);
```

#### 分页优化
```sql
-- 低效：OFFSET越大越慢
SELECT * FROM orders ORDER BY created_at DESC LIMIT 1000000, 20;

-- 高效：基于ID分页
SELECT * FROM orders
WHERE id < 1000000
ORDER BY id DESC LIMIT 20;

-- 使用游标
SELECT * FROM orders
WHERE created_at < '2024-01-01'
ORDER BY created_at DESC LIMIT 20;
```

---

## 第二章：事务与并发控制

### 2.1 事务隔离级别

#### 隔离级别详解
```sql
-- READ UNCOMMITTED（读未提交）
-- 最低隔离级别，可能读到脏数据

-- READ COMMITTED（读已提交）
-- Oracle默认，只能读到已提交数据
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;

-- REPEATABLE READ（可重复读）
-- MySQL InnoDB默认，同一事务内多次读取一致
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- SERIALIZABLE（串行化）
-- 最高隔离级别，完全串行执行
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
```

#### 隔离级别问题对比
| 隔离级别 | 脏读 | 不可重复读 | 幻读 |
|---------|------|-----------|------|
| READ UNCOMMITTED | 可能 | 可能 | 可能 |
| READ COMMITTED | 不可能 | 可能 | 可能 |
| REPEATABLE READ | 不可能 | 不可能 | InnoDB不可能 |
| SERIALIZABLE | 不可能 | 不可能 | 不可能 |

### 2.2 锁机制

#### 行锁与表锁
```sql
-- 行锁
SELECT * FROM orders WHERE id = 100 FOR UPDATE;

-- 表锁
LOCK TABLES orders READ, users WRITE;

-- 死锁检测与处理
-- InnoDB自动检测死锁，选择回滚影响最小的事务
SHOW ENGINE INNODB STATUS;

-- 减少死锁策略
-- 1. 按固定顺序访问表
-- 2. 避免在事务中用户交互
-- 3. 保持事务简短
-- 4. 使用较低的隔离级别
```

#### 乐观锁与悲观锁
```sql
-- 乐观锁（版本号）
UPDATE products
SET stock = stock - 1, version = version + 1
WHERE id = 100 AND version = 5;

-- 悲观锁（SELECT ... FOR UPDATE）
START TRANSACTION;
SELECT stock FROM products WHERE id = 100 FOR UPDATE;
-- 检查库存...
UPDATE products SET stock = 99 WHERE id = 100;
COMMIT;
```

---

## 第三章：分布式数据库

### 3.1 分库分表

#### 垂直拆分
```sql
-- 按业务字段拆分
-- 用户表拆分为 users_basic 和 users_extend

-- users_basic: 基本信息
CREATE TABLE users_basic (
    id BIGINT PRIMARY KEY,
    username VARCHAR(50),
    email VARCHAR(100),
    password_hash CHAR(60)
);

-- users_extend: 扩展信息
CREATE TABLE users_extend (
    user_id BIGINT PRIMARY KEY,
    phone VARCHAR(20),
    avatar_url VARCHAR(500),
    bio TEXT,
    settings JSON
);
```

#### 水平拆分
```sql
-- 按用户ID哈希分片
-- shard_key = user_id % 4

-- 订单表分片
CREATE TABLE orders_0 (
    id BIGINT AUTO_INCREMENT,
    user_id BIGINT,
    ...
    PRIMARY KEY (id),
    KEY idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE orders_1 (...);
CREATE TABLE orders_2 (...);
CREATE TABLE orders_3 (...);
```

### 3.2 分布式事务

#### 2PC两阶段提交
```
协调者                           参与者
   |                               |
   |---- Prepare ----------------->|
   |<---- Ready ------------------|
   |                               |
   |---- Commit ------------------>|
   |<---- ACK --------------------|
   |                               |
```

#### TCC事务
```python
# Try阶段：预留资源
def try_deduct_stock(order_id, quantity):
    # 冻结库存
    stock.freeze(product_id, quantity)
    return True

# Confirm阶段：确认使用
def confirm_deduct_stock(order_id, quantity):
    # 真正扣减库存
    stock.deduct(product_id, quantity)

# Cancel阶段：取消释放
def cancel_deduct_stock(order_id, quantity):
    # 释放冻结库存
    stock.unfreeze(product_id, quantity)
```

---

## 第四章：数据库架构设计

### 4.1 高可用架构

#### 主从复制
```sql
-- 主库配置
-- my.cnf
[mysqld]
server-id=1
log_bin=mysql-bin
binlog_format=ROW
sync_binlog=1

-- 从库配置
[mysqld]
server-id=2
relay_log=mysql-relay-bin
read_only=ON
```

#### 主从切换
```bash
# MHA主从切换流程
1. 检查复制延迟
2. 选择最新的从库作为新主库
3. 提升从库为主库
4. 其他从库同步新主库
5. 更新应用连接配置
```

### 4.2 数据库中间件

#### ShardingSphere
```yaml
# config.yaml
schemaName: sharding_db

dataSources:
  ds_0:
    url: jdbc:mysql://host1:3306/db0
    username: root
    password: password
  ds_1:
    url: jdbc:mysql://host2:3306/db1
    
shardingRules:
  tables:
    orders:
      actualDataNodes: ds_${0..1}.orders_${0..3}
      databaseStrategy:
        inline:
          shardingColumn: user_id
          algorithmExpression: ds_${user_id % 2}
      tableStrategy:
        inline:
          shardingColumn: order_id
          algorithmExpression: orders_${order_id % 4}
```

---

## 参考资源

### 官方文档
- MySQL: dev.mysql.com
- PostgreSQL: postgresql.org/docs
- Redis: redis.io/documentation

### 进阶书籍
- 《高性能MySQL》
- 《SQL查询优化艺术》
- 《数据库系统概念》

---

*本知识文件最后更新：2026-02-07*
