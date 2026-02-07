# 云原生与Kubernetes深度实践

## 第一章：云原生架构基础

### 1.1 云原生定义与特征

#### 云原生概念
- **CNCF定义**：云原生技术使组织能够在现代、动态的环境（如公有云、私有云、混合云）中构建和运行可扩展的应用程序。
- **核心特征**：
  - 容器化：应用程序及其依赖打包
  - 微服务架构：松耦合服务
  - 声明式API：期望状态管理
  - 不可变基础设施：环境一致性
  - 弹性伸缩：自动扩缩容

#### 云原生优势
- **敏捷性**：快速开发部署
- **可移植性**：避免厂商锁定
- **资源效率**：弹性资源利用
- **可靠性**：自愈和高可用

### 1.2 容器技术深入

#### Docker底层原理

**命名空间（Namespaces）**
- **PID命名空间**：进程隔离
- **网络命名空间**：网络隔离
- **挂载命名空间**：文件系统隔离
- **用户命名空间**：用户ID隔离
- **UTS命名空间**：主机名隔离
- **IPC命名空间**：进程间通信隔离

**控制组（cgroups）**
- **资源限制**
  - CPU限制（cpu.shares, cpu.cfs_period）
  - 内存限制（memory.limit_in_bytes）
  - I/O限制（blkio.throttle.*）
  - 设备访问控制

**联合文件系统（UnionFS）**
- **分层存储**
  - 镜像层只读
  - 容器层可写
  - 写时复制（COW）

- **存储驱动**
  - overlay2（推荐）
  - devicemapper
  - aufs
  - vfs

#### Docker最佳实践

**镜像优化**
- **多阶段构建**
  ```dockerfile
  # 构建阶段
  FROM golang:1.21 AS builder
  WORKDIR /app
  COPY . .
  RUN go build -o main main.go
  
  # 运行阶段
  FROM alpine:latest
  WORKDIR /app
  COPY --from=builder /app/main .
  CMD ["./main"]
  ```

- **.dockerignore**
  ```
  .git
  node_modules
  *.log
  Dockerfile
  .dockerignore
  ```

- **镜像扫描**
  - Trivy
  - Clair
  - Snyk

**网络配置**
- **Bridge网络**：默认NAT
- **Host网络**：直接使用宿主机网络
- **Overlay网络**：跨主机通信
- **Macvlan网络**：物理网络直通

### 1.3 容器编排需求

#### 单主机局限
- **资源隔离不足**
  - 进程冲突
  - 端口冲突
  - 存储冲突

- **调度能力弱**
  - 缺乏智能调度
  - 负载均衡不足
  - 故障恢复差

- **扩缩容困难**
  - 手动扩缩容
  - 缺乏自动恢复
  - 资源利用率低

#### 编排系统功能
- **服务发现**
  - 动态注册
  - DNS解析
  - 健康检查

- **负载均衡**
  - 流量分发
  - 会话保持
  - 灰度发布

- **配置管理**
  - 密钥管理
  - 配置注入
  - 动态更新

- **自愈能力**
  - 自动重启
  - 节点替换
  - 副本保证

---

## 第二章：Kubernetes深度实践

### 2.1 Kubernetes架构

#### 控制平面组件

**API Server（kube-apiserver）**
- **核心职责**
  - 提供REST API
  - 认证授权
  -  admission控制
  - 资源验证

- **性能优化**
  - 水平扩展（多实例）
  - 缓存层（etcd）
  - 请求限流

**Scheduler（kube-scheduler）**
- **调度流程**
  1. 过滤（Predicate）
  2. 优先级（Priority）
  3. 选择最优节点

- **调度策略**
  - 资源亲和性
  - 污点容忍
  - 拓扑分布

**Controller Manager（kube-controller-manager）**
- **核心控制器**
  - ReplicaSet控制器
  - Deployment控制器
  - Service控制器
  - Endpoints控制器

- **工作循环**
  - 期望状态对比
  - 调整实际状态
  - 错误恢复

**etcd**
- **分布式键值存储**
  - 存储所有集群状态
  - Raft一致性协议
  - 备份恢复策略

#### 工作节点组件

**Kubelet**
- **核心职责**
  - Pod管理
  - 容器运行时交互
  - 资源监控
  - 健康检查

- **工作流程**
  1. 从API Server获取Pod spec
  2. 同步Pod状态
  3. 调用容器运行时
  4. 报告状态到API Server

**Kube-proxy**
- **网络代理**
  - Service负载均衡
  - 流量转发
  - iptables/IPVS规则

- **代理模式**
  - userspace（古老）
  - iptables（默认）
  - IPVS（高性能）

**容器运行时**
- Docker/containerd/cri-o
- CRI（容器运行时接口）
- 镜像管理

### 2.2 核心资源对象

#### Pod深度解析

**Pod生命周期**
- **Pending**：调度中
- **Running**：运行中
- **Succeeded**：成功完成
- **Failed**：失败
- **Unknown**：未知状态

**Init容器**
- **应用场景**
  - 应用依赖准备
  - 权限初始化
  - 注册中心注册

- **执行特点**
  - 按顺序执行
  - 全部成功才启动主容器
  - 可以阻塞或延迟启动

**容器探针**
- **存活探针（livenessProbe）**
  - 检测容器是否存活
  - 失败则重启容器

- **就绪探针（readinessProbe）**
  - 检测是否就绪
  - 失败从Service移除

- **启动探针（startupProbe）**
  - 慢启动应用
  - 存活和就绪探针延迟

**探针类型**
- **exec**：执行命令
- **httpGet**：HTTP请求
- **tcpSocket**：TCP连接

#### Deployment高级特性

**滚动更新策略**
```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 0
```

- **maxSurge**：额外Pod数量
- **maxUnavailable**：不可用Pod最大数
- **暂停和恢复**
  - kubectl rollout pause deployment/myapp
  - kubectl rollout resume deployment/myapp

**回滚**
- **查看历史**：kubectl rollout history deployment/myapp
- **回滚到上一版本**：kubectl rollout undo deployment/myapp
- **回滚到指定版本**：kubectl rollout undo deployment/myapp --to-revision=3

**金丝雀发布**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: canary-ingress
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "30"
```

#### Service深度配置

**Service类型**
- **ClusterIP**（默认）：集群内部访问
- **NodePort**：节点端口暴露
- **LoadBalancer**：云负载均衡器
- **ExternalName**：外部服务映射

**服务发现**
- **环境变量**：自动注入
- **DNS**：CoreDNS解析
  - 普通Service：<service>.<namespace>.svc.cluster.local
  - Headless Service：<pod-name>.<service>.<namespace>.svc.cluster.local

**External Traffic Policy**
- **Cluster**：流量可能跨节点
- **Local**：流量只在本地节点

#### ConfigMap与Secret

**ConfigMap使用**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  config.yaml: |
    database:
      host: localhost
      port: 5432
  properties: |
    logging.level=INFO
```

- **使用方式**
  - 环境变量
  - 命令行参数
  - Volume挂载

**Secret管理**
- **类型**
  - generic：通用Secret
  - docker-registry：镜像仓库认证
  - tls：TLS证书

- **加密存储**
  - 静态加密
  - KMS集成
  - Vault集成

### 2.3 持久化存储

#### PersistentVolume（PV）

**PV类型**
- **本地存储（HostPath）**
  - 单节点
  - 高性能
  - 数据持久化

- **网络存储**
  - NFS
  - CephFS
  - GlusterFS
  - 云存储（GCE PD, AWS EBS, Azure Disk）

- **临时存储**
  - emptyDir
  - 与Pod生命周期相同

#### PersistentVolumeClaim（PVC）

**PVC配置**
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mysql-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast
```

**访问模式**
- **ReadWriteOnce（RWO）**：单节点读写
- **ReadOnlyMany（ROX）**：多节点只读
- **ReadWriteMany（RWX）**：多节点读写

#### StorageClass

**动态供应**
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast
provisioner: pd.csi.storage.gke.io
parameters:
  type: pd-ssd
  replication-type: none
```

- **供应器**
  - in-tree：内置
  - CSI：容器存储接口

### 2.4 调度策略

#### 节点亲和性

**RequiredDuringSchedulingIgnoredDuringExecution**
```yaml
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: disktype
            operator: In
            values:
            - ssd
            - nvme
```

**PreferredDuringSchedulingIgnoredDuringExecution**
```yaml
preferredDuringSchedulingIgnoredDuringExecution:
- weight: 100
  preference:
    matchExpressions:
    - key: zone
      operator: In
      values:
      - us-west1-a
```

#### Pod亲和性与反亲和性

**Pod反亲和性（分散调度）**
```yaml
spec:
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - webserver
        topologyKey: kubernetes.io/hostname
```

**Pod亲和性（协同调度）**
```yaml
spec:
  affinity:
    podAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - cache
        topologyKey: topology.kubernetes.io/zone
```

#### 污点与容忍

**添加污点**
```bash
kubectl taint nodes node1 key=value:NoSchedule
kubectl taint nodes node1 key=value:NoExecute
kubectl taint nodes node1 key=value:PreferNoSchedule
```

**配置容忍**
```yaml
spec:
  tolerations:
  - key: "key"
    operator: "Equal"
    value: "value"
    effect: "NoSchedule"
  - key: "key"
    operator: "Exists"
    effect: "NoExecute"
    tolerationSeconds: 3600
```

---

## 第三章：服务网格与可观测性

### 3.1 Istio服务网格

#### Istio架构

**数据平面**
- **Envoy代理**
  - Sidecar注入
  - 流量拦截
  - 策略执行

**控制平面**
- **Istiod**
  - 配置管理
  - 证书管理
  - 策略分发

#### 流量管理

**VirtualService**
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews-route
spec:
  hosts:
  - reviews
  http:
  - match:
    - headers:
        end-user:
          exact: jason
    route:
    - destination:
        host: reviews
        subset: v2
  - route:
    - destination:
        host: reviews
        subset: v1
```

**DestinationRule**
```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: reviews-destination
spec:
  host: reviews
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        h2UpgradePolicy: UPGRADE
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
    loadBalancer:
      simple: LEAST_REQUEST
    tls:
      mode: ISTIO_MUTUAL
```

**Gateway**
```yaml
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: gateway
spec:
  servers:
  - port:
      number: 443
      name: https
      protocol: HTTPS
    tls:
      mode: ISTIO_MUTUAL
      credentialName: cert-secret
    hosts:
    - "*"
```

#### 安全管理

**PeerAuthentication**
```yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: istio-system
spec:
  mtls:
    mode: STRICT
```

**AuthorizationPolicy**
```yaml
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: httpbin-policy
spec:
  selector:
    matchLabels:
      app: httpbin
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/default/sa/sleep"]
    to:
    - operation:
        paths: ["/data"]
        methods: ["POST"]
```

### 3.2 可观测性体系

#### 指标监控（Prometheus）

**PromQL查询**
```promql
# CPU使用率
100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# 内存使用率
(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100

# HTTP请求率
sum(rate(http_requests_total[5m])) by (method, status)
```

**告警规则**
```yaml
groups:
- name: node-alerts
  rules:
  - alert: HighCPUUsage
    expr: 100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage on {{ $labels.instance }}"
```

#### 日志管理（Loki）

**日志聚合架构**
- **采集**：Fluent Bit / Promtail
- **存储**：Loki（低成本日志存储）
- **查询**：LogQL
- **展示**：Grafana

**LogQL查询**
```logql
# 查找包含ERROR的日志
{job="myapp"} |= "ERROR"

# 统计错误频率
sum(rate({job="myapp"} |= "ERROR"[5m]))

# 日志模式检测
{job="myapp"} | json | line_format "{{.timestamp}} {{.level}} {{.message}}"
```

#### 分布式追踪（Jaeger）

**追踪概念**
- **Trace**：完整请求链路
- **Span**：调用片段
- **Context**：传递信息

**采样策略**
- **固定采样**：按比例采样
- **限流采样**：每秒限制数
- **优先级采样**：按业务重要性

---

## 第四章：GitOps最佳实践

### 4.1 GitOps核心概念

#### GitOps原则
1. **声明式配置**：所有配置代码化
2. **版本控制**：Git作为唯一真相来源
3. **自动化同步**：CI/CD自动化部署
4. **持续协调**：系统自动同步期望状态

#### 工作流程
1. **代码开发**：功能开发与测试
2. **提交代码**：Git push到仓库
3. **CI构建**：自动化测试与镜像构建
4. **更新配置**：Git更新部署配置
5. **自动同步**：GitOps Operator同步到集群
6. **健康检查**：验证部署状态

### 4.2 ArgoCD实践

#### ArgoCD配置

**Application定义**
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: guestbook
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/argoproj/argocd-example-apps.git
    targetRevision: HEAD
    path: guestbook
  destination:
    server: https://kubernetes.default.svc
    namespace: guestbook
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
  revisionHistoryLimit: 10
```

**Sync同步策略**
- **Manual**：手动同步
- **Auto**：自动同步
- **Prune**：自动清理资源
- **Self-Heal**：自动修复漂移

### 4.3 基础设施即代码

#### Terraform基础

**Provider配置**
```hcl
provider "kubernetes" {
  config_path    = "~/.kube/config"
  config_context = "my-context"
}

provider "helm" {
  kubernetes {
    config_path = "~/.kube/config"
  }
}
```

**资源定义**
```hcl
resource "kubernetes_namespace" "example" {
  metadata {
    name = "example-namespace"
  }
}

resource "kubernetes_deployment" "example" {
  metadata {
    name = "example"
    namespace = kubernetes_namespace.example.metadata[0].name
  }
  
  spec {
    replicas = 3
    
    selector {
      match_labels = {
        app = "example"
      }
    }
    
    template {
      metadata {
        labels = {
          app = "example"
        }
      }
      
      spec {
        container {
          image = "nginx:1.21"
          name  = "example"
          
          resources {
            limits {
              cpu    = "500m"
              memory = "512Mi"
            }
            requests {
              cpu    = "250m"
              memory = "256Mi"
            }
          }
        }
      }
    }
  }
}
```

---

## 第五章：实战案例

### 5.1 微服务部署案例

#### 项目架构
- **前端服务**：React + Nginx
- **API网关**：Kong / Istio Gateway
- **用户服务**：Go / Java
- **订单服务**：Python / Node.js
- **支付服务**：Java（安全隔离）
- **消息队列**：RabbitMQ / Kafka
- **缓存服务**：Redis Cluster
- **数据库**：PostgreSQL + Patroni

#### 部署配置
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  labels:
    app: api-gateway
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
        version: v1
    spec:
      serviceAccountName: api-gateway
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      containers:
      - name: api-gateway
        image: registry.example.com/api-gateway:v1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 5.2 高可用架构设计

#### 多集群部署
```yaml
# 跨集群Service
apiVersion: v1
kind: Service
metadata:
  name: global-service
  annotations:
    networking.gke.io/pre-shared-certs: "global-cert"
spec:
  type: GlobalExternalHTTP(S)LoadBalancer
  ports:
  - port: 443
    targetPort: 8080
  backendConfig:
    name: global-backend
---
apiVersion: cloud.google.com/v1
kind: BackendConfig
metadata:
  name: global-backend
  namespace: default
spec:
  healthCheck:
    checkIntervalSec: 15
    timeoutSec: 5
    healthyThreshold: 2
    unhealthyThreshold: 3
    type: HTTP
    requestPath: /health
    port: 8080
```

#### 自动扩缩容
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-gateway-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-gateway
  minReplicas: 3
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
```

---

## 参考资源

### 官方文档
- Kubernetes官方文档：kubernetes.io
- Docker官方文档：docs.docker.com
- Istio官方文档：istio.io
- Prometheus官方文档：prometheus.io

### 进阶书籍
- 《Kubernetes权威指南》
- 《Kubernetes in Action》
- 《Site Reliability Engineering》
- 《Cloud Native Infrastructure》

### 在线课程
- CKAD/CKA/CKAD认证课程
- Linux Academy
- Coursera云原生专项课程

---

*本知识文件最后更新：2026-02-07*
*涵盖云原生架构、Kubernetes深度实践、GitOps*
