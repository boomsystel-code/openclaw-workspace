# 数据工程实战 - Spark与Flink

## 第一章：Apache Spark深度实践

### 1.1 Spark核心概念

#### Spark架构
```
Driver Program
    |
    +-- SparkContext
         |
         +-- Cluster Manager (YARN/Mesos/K8s)
              |
              +-- Executor 1
              |    +-- Task 1
              |    +-- Task 2
              |
              +-- Executor 2
                   +-- Task 1
                   +-- Task 2
```

#### RDD详解
```python
from pyspark import SparkConf, SparkContext

# 创建SparkContext
conf = SparkConf().setAppName("MyApp").setMaster("local[*]")
sc = SparkContext(conf=conf)

# 创建RDD
data = [1, 2, 3, 4, 5]
rdd = sc.parallelize(data)

# 转换操作
mapped_rdd = rdd.map(lambda x: x * 2)
filtered_rdd = rdd.filter(lambda x: x > 2)
flat_mapped_rdd = rdd.flatMap(lambda x: [x, -x])

# 行动操作
result = rdd.collect()           # 收集到Driver
count = rdd.count()              # 计数
first = rdd.first()             # 第一个元素
take = rdd.take(3)              # 取前3个
reduce_sum = rdd.reduce(lambda a, b: a + b)

# 持久化
rdd.persist(StorageLevel.MEMORY_ONLY)
rdd.unpersist()
```

### 1.2 DataFrame与Spark SQL

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, max, when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# 创建SparkSession
spark = SparkSession.builder \
    .appName("DataFrameDemo") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# 定义Schema
schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("name", StringType(), True),
    StructField("department", StringType(), True),
    StructField("salary", IntegerType(), True)
])

# 创建DataFrame
df = spark.read.json("data.json", schema=schema)

# DSL风格查询
result = df.filter(col("salary") > 5000) \
    .groupBy("department") \
    .agg(
        sum("salary").alias("total_salary"),
        avg("salary").alias("avg_salary"),
        max("salary").alias("max_salary")
    ) \
    .orderBy(col("avg_salary").desc())

# SQL风格查询
df.createOrReplaceTempView("employees")

result_sql = spark.sql("""
    SELECT department,
           SUM(salary) as total_salary,
           AVG(salary) as avg_salary
    FROM employees
    WHERE salary > 5000
    GROUP BY department
    ORDER BY avg_salary DESC
""")

# 添加列
df_with_bonus = df.withColumn(
    "bonus",
    when(col("salary") > 8000, col("salary") * 0.1)
    .when(col("salary") > 5000, col("salary") * 0.05)
    .otherwise(0)
)
```

### 1.3 Spark性能优化

#### 广播变量
```python
# 广播小表
small_table = {"A": 1, "B": 2, "C": 3}
broadcast_var = sc.broadcast(small_table)

# 使用广播变量
def lookup_department(dept_code):
    return broadcast_var.value.get(dept_code, "Unknown")

result = df.map(lambda row: (row.name, lookup_department(row.department)))
```

#### 分区优化
```python
# 重新分区
df_repartitioned = df.repartition(200, "department")

# 合并小分区
df_coalesced = df.coalesce(50)

# 自定义分区器
from pyspark import Partitioners

rdd_with_partitioner = rdd.keyBy(lambda x: x[0]) \
    .partitionBy(numPartitions=100, partitionFunc=lambda x: hash(x) % 100)
```

#### 缓存策略
```python
# 选择存储级别
df.persist(StorageLevel.MEMORY_AND_DISK)  # 内存放不下则磁盘
df.persist(StorageLevel.MEMORY_ONLY_SER) # 序列化存储，节省内存

# 何时缓存
# 1. DataFrame被多次引用
# 2. 迭代计算（如机器学习多次迭代）
# 3. 复杂转换链

# 避免缓存
# 1. 只使用一次的DataFrame
# 2. 大规模临时数据
```

---

## 第二章：Apache Flink流处理

### 2.1 Flink核心概念

#### 流处理架构
```
DataStream API:
Source → Transform → Sink

Table API / SQL:
Tables with time semantics

DataSet API (批处理):
DataSet → Map → Reduce → DataSet
```

#### Time语义
```java
// Processing Time（处理时间）
// 数据被处理的时间
env.setStreamTimeCharacteristic(TimeCharacteristic.ProcessingTime);

// Event Time（事件时间）
// 数据产生的时间
env.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);

// Ingestion Time（摄入时间）
// 数据进入Flink的时间
env.setStreamTimeCharacteristic(TimeCharacteristic.IngestionTime);

// 指定Event Time
DataStream<SensorReading> readings = env
    .addSource(new SensorSource())
    .assignTimestampsAndWatermarks(
        WatermarkStrategy.<SensorReading>forBoundedOutOfOrderness(Duration.ofSeconds(20))
            .withTimestampAssigner((event, timestamp) -> event.timestamp)
    );
```

### 2.2 DataStream API

```java
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.api.common.functions.MapFunction;

public class FlinkDemo {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        // Source
        DataStream<String> lines = env.socketTextStream("localhost", 9999);
        
        // Transform
        DataStream<SensorReading> readings = lines
            .map((MapFunction<String, SensorReading>) line -> {
                String[] fields = line.split(",");
                return new SensorReading(
                    fields[0],
                    Long.parseLong(fields[1]),
                    Double.parseDouble(fields[2])
                );
            })
            .keyBy(SensorReading::getId)
            .window(TumblingEventTimeWindows.of(Time.seconds(30)))
            .allowedLateness(Time.seconds(10))
            .process(new TemperatureAverager());
        
        // Sink
        readings.addSink(new InfluxDBSink());
        
        env.execute("Flink Stream Processing");
    }
}
```

### 2.3 Flink SQL

```java
// 创建表
Table orders = tableEnv.fromDataStream(
    ordersStream,
    $("order_id"),
    $("user_id"),
    $("amount"),
    $("timestamp").rowtime()
);

Table products = tableEnv.fromDataStream(
    productsStream,
    $("product_id"),
    $("category"),
    $("price")
);

// SQL查询
Table revenue = tableEnv.sqlQuery("""
    SELECT 
        TUMBLE_START(o.timestamp, INTERVAL '1' HOUR) AS window_start,
        o.category,
        SUM(o.amount) AS revenue,
        COUNT(DISTINCT o.user_id) AS unique_users
    FROM orders o
    JOIN products p ON o.product_id = p.product_id
    WHERE p.category IN ('Electronics', 'Clothing')
    GROUP BY 
        TUMBLE(o.timestamp, INTERVAL '1' HOUR'),
        o.category
    HAVING SUM(o.amount) > 10000
""");

// 输出到Kafka
tableEnv.executeSql("""
    CREATE TABLE sink_table (
        window_start TIMESTAMP,
        category STRING,
        revenue DOUBLE,
        unique_users BIGINT
    ) WITH (
        'connector' = 'jdbc',
        'url' = 'jdbc:mysql://localhost:3306/db',
        'table-name' = 'revenue_stats'
    )
""");

revenue.executeInsert("sink_table");
```

---

## 第三章：数据湖与数据仓库

### 3.1 Delta Lake

```python
from delta import DeltaTable
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# 创建Delta表
DeltaTable.createOrReplace(spark) \
    .addColumn("id", IntegerType()) \
    .addColumn("name", StringType()) \
    .addColumn("value", DoubleType()) \
    .partitionBy("date") \
    .location("/data/delta/table") \
    .execute()

# 读取Delta表
df = spark.read.format("delta").load("/data/delta/table")

# 时间旅行（Time Travel）
df_v1 = spark.read.format("delta").option("versionAsOf", 0).load("/data/delta/table")
df_today = spark.read.format("delta").option("timestampAsOf", "2024-01-01").load("/data/delta/table")

# 增量读取
changes = spark.read.format("delta") \
    .option("startingVersion", 100) \
    .load("/data/delta/table")

# 合并（MERGE）
deltaTable.merge(
    source="source_table s",
    condition="t.id = s.id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()

# 优化文件大小
deltaTable.optimize().executeZOrderBy("category")
```

### 3.2 Apache Iceberg

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.spark_catalog.type", "hive") \
    .getOrCreate()

# 创建Iceberg表
spark.sql("""
    CREATE TABLE my_catalog.my_db.orders (
        order_id BIGINT,
        user_id BIGINT,
        amount DECIMAL(10,2),
        created_at TIMESTAMP
    ) USING iceberg
    PARTITIONED BY (days(created_at))
""")

# 增量查询（Change Data Capture）
spark.sql("""
    SELECT * FROM my_catalog.my_db.orders.changes
    WHERE change_ordinal = 1
    AND change_type IN ('INSERT', 'UPDATE_AFTER')
""")

# Time Travel
df_old = spark.read \
    .format("iceberg") \
    .option("snapshot-id", 12345678901234) \
    .load("my_catalog.my_db.orders")

# 架构演进
spark.sql("""
    ALTER TABLE my_catalog.my_db.orders
    ADD COLUMNS (
        shipping_address STRING,
        status STRING AFTER amount
    )
""")
```

---

## 参考资源

### 官方文档
- Apache Spark: spark.apache.org/docs
- Apache Flink: nightlies.apache.org/flink
- Delta Lake: docs.delta.io
- Apache Iceberg: ice.apache.org/docs

### 进阶书籍
- 《Spark权威指南》
- 《Stream Processing with Apache Flink》
- 《Designing Data-Intensive Applications》

### 在线资源
- Databricks Academy
- Cloudera DataFlow
- O'Reilly Spark/Flink Courses

---

*本知识文件最后更新：2026-02-07*
