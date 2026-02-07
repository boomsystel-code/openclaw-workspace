# MicroServices Architecture 微服务架构完全指南

## 第一章：微服务基础

### 1.1 微服务定义与特征

#### 什么是微服务？
微服务是一种将应用程序构建为松耦合服务集合的架构风格。每个服务：
- **独立部署**：可独立开发、测试和部署
- **业务边界明确**：围绕业务能力组织
- **轻量级通信**：通常使用HTTP REST或消息队列
- **技术栈灵活**：不同服务可使用不同技术

#### 微服务特征
```
✓ 单一职责原则（SRP）
✓ 高度解耦
✓ 独立数据存储
✓ 自动化部署
✓ 容错设计
✓ 按业务组织团队
```

### 1.2 微服务 vs 单体架构

| 维度 | 单体架构 | 微服务架构 |
|------|---------|-----------|
| 开发速度 | 初期快，后期慢 | 持续快速 |
| 部署 | 整体部署 | 独立部署 |
| 扩展 | 整体扩展 | 按需扩展 |
| 容错 | 单点故障 | 隔离故障 |
| 技术栈 | 统一 | 多样化 |
| 复杂度 | 集中 | 分布式 |
| 数据管理 | 统一数据库 | 多数据源 |
| 团队协作 | 紧密 | 自治 |

---

## 第二章：微服务通信

### 2.1 同步通信

#### HTTP/REST
```yaml
# OpenAPI 3.0 定义
openapi: 3.0.0
info:
  title: User Service API
  version: 1.0.0
paths:
  /users/{userId}:
    get:
      summary: Get user by ID
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: integer
      responses:
        '200':
          description: User found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
```

#### gRPC
```protobuf
// user.proto
syntax = "proto3";

package user;

service UserService {
  rpc GetUser(GetUserRequest) returns (User);
  rpc CreateUser(CreateUserRequest) returns (User);
  rpc ListUsers(ListUsersRequest) returns (ListUsersResponse);
}

message User {
  int64 id = 1;
  string username = 2;
  string email = 3;
  int64 created_at = 4;
}

message GetUserRequest {
  int64 id = 1;
}
```

### 2.2 异步通信

#### 消息队列
```python
# 生产者
import pika

connection = pika.BlockingConnection(
    pika.ConnectionParameters('rabbitmq')
)
channel = connection.channel()

channel.queue_declare(queue='user_events')

channel.basic_publish(
    exchange='',
    routing_key='user_events',
    body=json.dumps({
        'event_type': 'USER_CREATED',
        'user_id': 123,
        'timestamp': time.time()
    })
)
```

#### 事件驱动
```python
# 使用Kafka
from confluent_kafka import Producer

def delivery_callback(err, msg):
    if err:
        print(f'Message failed delivery: {err}')

producer = Producer({
    'bootstrap.servers': 'kafka-broker:9092'
})

producer.produce(
    'user-events',
    key='user:123',
    value=json.dumps(event),
    callback=delivery_callback
)
```

---

## 第三章：服务发现与注册

### 3.1 服务注册

#### Consul服务注册
```python
import consul

c = consul.Consul()

# 注册服务
c.agent.service.register(
    'user-service',
    service_id='user-service-001',
    address='user-service.internal',
    port=8080,
    check={
        'http': 'http://user-service.internal:8080/health',
        'interval': '10s',
        'timeout': '5s'
    }
)

# 健康检查
def register_health_check():
    c.agent.check.register(
        'user-service-health',
        {
            'http': 'http://user-service.internal:8080/health',
            'interval': '10s',
            'timeout': '5s',
            ' deregistercriticalserviceafter': '30s'
        }
    )
```

### 3.2 服务发现

#### 客户端发现模式
```python
import consul

class ServiceDiscovery:
    def __init__(self):
        self.consul = consul.Consul()
    
    def get_service(self, service_name):
        _, services = self.consul.health.service(
            service_name,
            passing_only=True
        )
        
        if not services:
            raise NoAvailableService(f"No instances of {service_name}")
        
        # 简单轮询或随机选择
        service = random.choice(services)
        return f"http://{service['Address']}:{service['Port']}"
```

---

## 第四章：API网关

### 4.1 Kong网关配置

#### 服务定义
```bash
# 添加服务
curl -i -X POST http://kong:8001/services \
  --data "name=user-service" \
  --data "url=http://user-service:8080"

# 添加路由
curl -i -X POST http://kong:8001/services/user-service/routes \
  --data "paths[]=/api/users" \
  --data "strip_path=false"
```

#### 插件配置
```yaml
# 限流插件
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: rate-limit
config:
  minute: 100
  policy: local

# JWT认证插件
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: jwt-auth
config:
  key_claim_name: iss
  claims_to_verify:
    - exp
```

### 4.2 OAuth 2.0集成

#### Authorization Code Flow
```
流程：
1. 用户访问客户端应用
2. 重定向到授权服务器
3. 用户登录并授权
4. 返回授权码
5. 客户端用授权码换取Token
6. 使用Token访问资源
```

---

## 第五章：服务间认证

### 5.1 JWT认证

#### JWT Token结构
```
Header:
{
  "alg": "RS256",
  "typ": "JWT"
}

Payload:
{
  "sub": "user-123",
  "iss": "auth-service",
  "aud": "user-service",
  "exp": 1234567890,
  "scope": "user:read user:write"
}

Signature:
RS256(header, payload, private_key)
```

#### 服务间认证
```python
import jwt
from functools import wraps
from flask import request, jsonify

class AuthService:
    def verify_jwt(self, token):
        try:
            public_key = self.get_public_key()
            payload = jwt.decode(
                token,
                public_key,
                algorithms=['RS256'],
                audience='api-gateway'
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise TokenExpired()
        except jwt.InvalidTokenError:
            raise InvalidToken()

def require_service_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Missing token'}), 401
        
        token = auth_header.replace('Bearer ', '')
        payload = auth_service.verify_jwt(token)
        request.service = payload
        return f(*args, **kwargs)
    return decorated
```

---

## 第六章：微服务安全

### 6.1 安全最佳实践

#### 零信任架构
```
原则：
- 不信任内部网络
- 最小权限原则
- 持续验证
- 端到端加密

实施方式：
- mTLS（双向认证）
- 服务网格（Istio）
- API网关统一认证
```

#### mTLS配置
```yaml
# Istio PeerAuthentication
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: istio-system
spec:
  mtls:
    mode: STRICT
```

---

## 第七章：微服务测试

### 7.1 测试策略

#### 测试金字塔
```
         ┌─────────┐
        /  E2E    \      ← 少量，验收测试
       /───────────\
      /  Integration \   ← 中等，服务集成
     /─────────────────\
    /      Unit          \  ← 大量，单元测试
   /─────────────────────\
```

### 7.2 Contract Testing

#### Pact测试
```python
import pact

consumer = pact.Consumer('UserConsumer')
provider = pact.Provider('UserProvider')

consumer.given('user with id 123 exists').upon_receiving(
    'a request for user 123'
).with_request(
    method='GET',
    path='/users/123'
).will_respond_with(
    status=200,
    body={
        'id': 123,
        'username': 'testuser',
        'email': 'test@example.com'
    }
```

---

## 第八章：微服务监控

### 8.1 可观测性三支柱

#### 日志聚合
```yaml
# Fluentd配置
<source>
  @type tail
  path /var/log/**/*.log
  tag microservice.*
</source>

<match microservice.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name logs
</match>
```

#### 指标监控
```
四大黄金指标：
1. 延迟（Latency）
   - P50, P95, P99
   - 平均值

2. 流量（Traffic）
   - QPS
   - 吞吐量

3. 错误（Errors）
   - 错误率
   - 故障率

4. 饱和度（Saturation）
   - CPU使用率
   - 内存使用率
```

#### 分布式追踪
```python
from opentracing import tracer

def get_user(user_id):
    with tracer.start_span('get_user') as span:
        span.set_tag('user_id', user_id)
        
        with tracer.start_span('db_query'):
            user = db.query(user_id)
        
        return user
```

---

## 参考资源

### 官方文档
- Microservices.io patterns
- Microsoft Microservices Architecture
- AWS Microservices

### 书籍推荐
- 《微服务架构设计模式》
- 《Building Microservices》
- 《The Tao of Microservices》

### 工具推荐
- 服务发现：Consul, Eureka, etcd
- API网关：Kong, Istio, Apigee
- 消息队列：Kafka, RabbitMQ, Pulsar
- 监控：Prometheus, Grafana, Jaeger

---

*本知识文件最后更新：2026-02-07*
*涵盖微服务架构、服务通信、服务发现、安全、测试*
