# Python高级编程与架构设计

## 第一章：Python并发编程

### 1.1 多线程编程

#### threading模块
```python
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# 基本线程创建
def worker(name, delay):
    print(f"Thread {name} starting")
    time.sleep(delay)
    print(f"Thread {name} finishing")

# 创建线程
thread = threading.Thread(target=worker, args=("Worker1", 2))
thread.start()
thread.join()

# 线程池
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(worker, i, 1) for i in range(8)]
    for future in as_completed(futures):
        result = future.result()
```

#### 线程同步
```python
import threading
from threading import Lock, RLock, Semaphore, Condition

# Lock（互斥锁）
lock = Lock()
counter = 0

def increment():
    global counter
    with lock:
        counter += 1

# RLock（可重入锁）- 同一线程可多次获取
rlock = RLock()

# Semaphore（信号量）- 控制并发数
semaphore = Semaphore(3)

# Condition（条件变量）
condition = Condition()

def consumer():
    with condition:
        while not data_available:
            condition.wait()
        data = data_queue.pop()
        return data

def producer():
    with condition:
        data_queue.append(item)
        data_available = True
        condition.notify_all()
```

#### 线程安全的数据结构
```python
from collections import Queue
from queue import Queue, LifoQueue, PriorityQueue

# 线程安全的队列
q = Queue(maxsize=100)

# 生产者
def producer():
    for i in range(100):
        q.put(i)

# 消费者
def consumer():
    while True:
        item = q.get()
        if item is None:  # 终止信号
            break
        process(item)
        q.task_done()
```

### 1.2 多进程编程

#### multiprocessing模块
```python
from multiprocessing import Process, Pool, Queue, Pipe
import multiprocessing

# 基本进程创建
def worker(name):
    print(f"Worker {name} starting")
    time.sleep(2)
    print(f"Worker {name} finishing")

if __name__ == "__main__":
    processes = []
    for i in range(4):
        p = Process(target=worker, args=(i,))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
```

#### 进程池
```python
from multiprocessing import Pool

def process_item(item):
    return item * item

if __name__ == "__main__":
    with Pool(processes=4) as pool:
        # 同步执行
        results = pool.map(process_item, range(10))
        
        # 异步执行
        future = pool.apply_async(process_item, (5,))
        result = future.get()
        
        #imap（迭代器方式）
        for result in pool.imap(process_item, range(10)):
            print(result)
```

#### 进程间通信
```python
# Queue（队列）
from multiprocessing import Queue

def producer(q):
    for i in range(10):
        q.put(i)

def consumer(q):
    while True:
        item = q.get()
        if item is None:
            break
        print(item)

# Pipe（管道）
parent_conn, child_conn = Pipe()
```

### 1.3 异步编程（asyncio）

#### 异步基础
```python
import asyncio
import aiohttp

async def fetch_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def main():
    urls = [
        'https://example.com',
        'https://google.com',
        'https://github.com'
    ]
    
    # 并发执行
    tasks = [fetch_url(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    for url, content in zip(urls, results):
        print(f"{url}: {len(content)} bytes")

asyncio.run(main())
```

#### asyncio高级用法
```python
import asyncio
from asyncio import Semaphore, Queue

# 限流器
semaphore = Semaphore(10)

async def limited_fetch(url):
    async with semaphore:
        return await fetch_url(url)

# 优先级队列
priority_queue = asyncio.PriorityQueue()

# 取消任务
async def long_task():
    try:
        await asyncio.sleep(100)
    except asyncio.CancelledError:
        print("Task cancelled")
        raise

task = asyncio.create_task(long_task())
await asyncio.sleep(5)
task.cancel()

# 超时控制
async with asyncio.timeout(10):
    await long_running_operation()
```

---

## 第二章：Python设计模式

### 2.1 创建型模式

#### 单例模式
```python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# 使用装饰器
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    pass
```

#### 工厂模式
```python
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        raise ValueError("Unknown animal type")
```

### 2.2 结构型模式

#### 适配器模式
```python
class OldCalculator:
    def operations(self, t1, t2, op):
        result = 0
        if op == 'add':
            result = t1 + t2
        elif op == 'sub':
            result = t1 - t2
        return result

class NewCalculator:
    def add(self, t1, t2):
        return t1 + t2
    
    def sub(self, t1, t2):
        return t1 - t2

class CalculatorAdapter:
    def __init__(self):
        self.new_calc = NewCalculator()
    
    def operations(self, t1, t2, op):
        if op == 'add':
            return self.new_calc.add(t1, t2)
        elif op == 'sub':
            return self.new_calc.sub(t1, t2)
```

#### 装饰器模式
```python
class Coffee:
    def get_description(self):
        return "Coffee"
    
    def get_cost(self):
        return 5

class CoffeeDecorator(Coffee):
    def __init__(self, coffee):
        self._coffee = coffee
    
    def get_description(self):
        return self._coffee.get_description()
    
    def get_cost(self):
        return self._coffee.get_cost()

class MilkDecorator(CoffeeDecorator):
    def get_description(self):
        return self._coffee.get_description() + ", Milk"
    
    def get_cost(self):
        return self._coffee.get_cost() + 1.0

# 使用
coffee = Coffee()
coffee_with_milk = MilkDecorator(coffee)
```

### 2.3 行为型模式

#### 观察者模式
```python
from abc import ABC, abstractmethod

class Subject:
    def __init__(self):
        self._observers = []
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    def notify(self, message):
        for observer in self._observers:
            observer.update(message)

class Observer(ABC):
    @abstractmethod
    def update(self, message):
        pass

class ConcreteObserver(Observer):
    def update(self, message):
        print(f"Received: {message}")
```

#### 策略模式
```python
from abc import ABC, abstractmethod

class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data):
        pass

class BubbleSort(SortStrategy):
    def sort(self, data):
        return sorted(data)  # 实际使用冒泡

class QuickSort(SortStrategy):
    def sort(self, data):
        return sorted(data)  # 实际使用快速排序

class Sorter:
    def __init__(self, strategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy):
        self._strategy = strategy
    
    def execute(self, data):
        return self._strategy.sort(data)
```

---

## 第三章：Python性能优化

### 3.1 性能分析

#### cProfile
```python
import cProfile
import pstats
from io import StringIO

def my_function():
    # 被分析的代码
    sum([i**2 for i in range(100000)])

profiler = cProfile.Profile()
profiler.enable()
my_function()
profiler.disable()

# 输出统计信息
stream = StringIO()
stats = pstats.Stats(profiler, stream=stream)
stats.sort_stats('cumulative')
stats.print_stats(20)  # 前20个函数
print(stream.getvalue())
```

#### line_profiler
```python
# 使用@profile装饰器
@profile
def my_slow_function():
    for i in range(10000):
        # 逐行分析
        result = i * i
    
# 运行分析
# kernprof -l -v script.py
```

### 3.2 内存优化

#### 生成器与迭代器
```python
# 生成器 - 惰性计算
def fibonacci_generator():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# 内存效率对比
# 列表：一次性加载所有数据
data = [i**2 for i in range(100000000)]  # ~800MB

# 生成器：按需计算
data = (i**2 for i in range(100000000))  # ~100bytes
```

#### __slots__优化
```python
class Point:
    __slots__ = ('x', 'y')  # 禁用__dict__，节省内存
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

#### 数据类型优化
```python
# 使用array模块
from array import array

# Python列表（每个元素是对象）
list_data = [1, 2, 3, 4, 5]  # 每个int ~28bytes

# array（连续内存）
array_data = array('i', [1, 2, 3, 4, 5])  # 每个int 4bytes

# 使用numpy进行数值计算
import numpy as np
numpy_array = np.array([1, 2, 3, 4, 5])  # 高效内存和向量化操作
```

### 3.3 C扩展

#### ctypes
```python
from ctypes import cdll, c_int

# 加载C动态库
libc = cdll.LoadLibrary("libc.so.6")

# 调用C函数
libc.printf(b"Hello from C!\n")

# 传递参数
libc.printf(b"Value: %d\n", c_int(42))
```

#### Cython
```python
# hello.pyx
def fib(int n):
    cdef int a = 0, b = 1
    cdef int i
    for i in range(n):
        a, b = b, a + b
    return a
```

---

## 第四章：Python Web框架

### 4.1 Flask高级用法

#### 蓝图（Blueprint）
```python
# app.py
from flask import Flask
from admin import admin_bp
from api import api_bp

app = Flask(__name__)
app.register_blueprint(admin_bp, url_prefix='/admin')
app.register_blueprint(api_bp, url_prefix='/api')

# admin.py
admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/dashboard')
def dashboard():
    return 'Admin Dashboard'
```

#### 上下文管理
```python
from flask import Flask, g, current_app

app = Flask(__name__)

@app.before_request
def before_request():
    g.user = get_current_user()
    g.db = get_db_connection()

@app.teardown_appcontext
def cleanup(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()
```

#### 扩展使用
```python
# Flask-SQLAlchemy
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))

# Flask-Migrate
from flask_migrate import Migrate
migrate = Migrate(app, db)

# Flask-Cache
from flask_caching import Cache
cache = Cache(config={'CACHE_TYPE': 'RedisCache'})
cache.init_app(app)
```

### 4.2 FastAPI深度实践

#### 依赖注入
```python
from fastapi import FastAPI, Depends

app = FastAPI()

# 依赖
def get_db():
    db = Database()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme)):
    user = decode_token(token)
    return user

# 使用依赖
@app.get("/users/me")
async def read_users_me(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return current_user
```

#### Pydantic模型
```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    password: str = Field(..., min_length=8)
    
    @validator('username')
    def username_alphanumeric(cls, v):
        assert v.isalnum(), 'must be alphanumeric'
        return v

class UserResponse(BaseModel):
    id: int
    username: str
    created_at: datetime
    
    class Config:
        orm_mode = True
```

### 4.3 Django企业级开发

#### settings配置
```python
# settings.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY')

DEBUG = os.environ.get('DEBUG', 'False') == 'True'

ALLOWED_HOSTS = ['*']

# 数据库配置
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('DB_NAME'),
        'USER': os.environ.get('DB_USER'),
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': os.environ.get('DB_HOST'),
        'PORT': os.environ.get('DB_PORT', 5432),
    }
}

# Redis缓存
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': os.environ.get('REDIS_URL'),
    }
}
```

#### Django REST Framework
```python
# serializers.py
from rest_framework import serializers
from .models import Article

class ArticleSerializer(serializers.ModelSerializer):
    author = serializers.ReadOnlyField(source='author.username')
    
    class Meta:
        model = Article
        fields = '__all__'

# views.py
from rest_framework import viewsets, permissions

class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    
    def perform_create(self, serializer):
        serializer.save(author=self.request.user)
```

---

## 第五章：Python机器学习工程化

### 5.1 ML Pipeline

#### 数据预处理Pipeline
```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# 数值特征处理
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 分类特征处理
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 组合
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 完整Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
```

#### 模型持久化
```python
import joblib
import pickle

# 保存模型
joblib.dump(model_pipeline, 'model_pipeline.joblib')
pickle.dump(model_pipeline, open('model.pkl', 'wb'))

# 加载模型
model = joblib.load('model_pipeline.joblib')

# 模型版本管理
MODEL_REGISTRY = {
    'v1.0': joblib.load('models/v1.0/model.joblib'),
    'v2.0': joblib.load('models/v2.0/model.joblib'),
}
```

### 5.2 MLOps实践

#### MLflow实验跟踪
```python
import mlflow
import mlflow.sklearn

# 启动实验
mlflow.start_run(run_name="experiment_001")

# 记录参数
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 10)

# 记录指标
mlflow.log_metric("accuracy", 0.95)
mlflow.log_metric("f1_score", 0.93)

# 记录模型
mlflow.sklearn.log_model(model, "model")

# 记录工件
mlflow.log_artifact("data/processed/test.csv")

mlflow.end_run()
```

#### BentoML模型服务
```python
import bentoml
from bentoml.mlir import infer

# 保存模型
bentoml.sklearn.save_model("classifier", model)

# 加载服务
svc = bentoml.Service(
    "classifier",
    runners=[
        bentoml.sklearn.get("classifier:latest").to_runner()
    ]
)

@svc.api(input=infer.JSON(), output=infer.JSON())
def classify(input_data):
    return runner.run(input_data)
```

---

## 参考资源

### 官方文档
- Python官方文档：docs.python.org
- FastAPI文档：fastapi.tiangolo.com
- Flask文档：flask.palletsprojects.com
- Django文档：www.djangoproject.com

### 进阶书籍
- 《Fluent Python》
- 《Python Cookbook》
- 《High Performance Python》
- 《Architecture Patterns with Python》

### 在线课程
- Real Python（realpython.com）
- Full Stack Python（fullstackpython.com）
- Talk Python Training

---

*本知识文件最后更新：2026-02-07*
*涵盖Python并发编程、设计模式、性能优化、Web框架*
