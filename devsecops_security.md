# DevSecOps安全实践

## 第一章：安全开发流程

### 1.1 安全开发生命周期

```
┌─────────────────────────────────────────────────────────────┐
│                    SDLC安全集成                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐──▶┌─────────┐──▶┌─────────┐──▶┌─────────┐     │
│  │  Plan   │   │ Design  │   │Develop │   │  Test   │     │
│  │ (计划)  │   │ (设计)  │   │ (开发)  │   │ (测试)  │     │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘     │
│       │             │             │             │          │
│       ▼             ▼             ▼             ▼          │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │威胁建模 │   │架构评审 │   │SAST/   │   │DAST/    │     │
│  │(TMD)   │   │(STRIDE) │   │Linting │   │IAST     │     │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘     │
│                                                              │
│  ┌─────────┐──▶┌─────────┐──▶┌─────────┐──▶┌─────────┐     │
│  │Release  │   │Deploy  │   │Operate │   │Monitor  │     │
│  │ (发布)  │   │ (部署)  │   │ (运营)  │   │ (监控)  │     │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘     │
│       │             │             │             │          │
│       ▼             ▼             ▼             ▼          │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │签署验证 │   │基础设施 │   │运行时  │   │SIEM/    │     │
│  │(SBoM)  │   │(IaC扫描)│   │防护    │   │SOC      │     │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 威胁建模

#### STRIDE威胁分析
```
STRIDE威胁分类：
┌──────────┬──────────────────┬─────────────────────┐
│ 威胁类型  │      描述        │      缓解措施       │
├──────────┼──────────────────┼─────────────────────┤
│ Spoofing│ 身份欺骗          │ 强认证、多因素认证  │
│ Tampering│ 数据篡改         │ 完整性校验、加密    │
│ Repudiation│ 抵赖行为       │ 审计日志、数字签名  │
│ Info Disclosure│ 信息泄露   │ 加密、访问控制     │
│ Denial of Service│ 拒绝服务 │ 限流、冗余设计     │
│ Elevation of Priv│ 权限提升 │ 最小权限原则       │
└──────────┴──────────────────┴─────────────────────┘

威胁建模流程：
1. 识别资产
2. 创建架构图
3. 识别入口点
4. 识别信任边界
5. 列出威胁
6. 缓解威胁
7. 验证缓解
```

---

## 第二章：安全工具链

### 2.1 SAST静态应用安全测试

#### SonarQube配置
```yaml
# sonar-project.properties
sonar.projectKey=my-project
sonar.projectName=My Application
sonar.projectVersion=1.0.0

sonar.sources=src
sonar.tests=tests
sonar.language=java
sonar.java.binaries=target/classes
sonar.java.test.binaries=target/test-classes

# 质量门禁
sonar.qualitygate.wait=true
sonar.qualitygate.timeout=300

# 排除
sonar.exclusions=**/test/**/*,**/*.spec.ts
sonar.test.exclusions=**/test/**/*

# 安全规则
sonar.security.escaping=true
sonar.webhook.name=jenkins
sonar.webhook.url=http://jenkins:8080/sonarqube-webhook
```

#### GitLab SAST
```yaml
# .gitlab-ci.yml
include:
  - template: Security/SAST.gitlab-ci.yml

variables:
  SAST_GOLANG_SKIP_BINARIES: "(^|/).*\.keep$"
  SAST_JAVA_SCAN_DEPTH: "2"

stages:
  - test
  - security
  - deploy

sast:
  stage: security
  variables:
    SEARCH_MAX_DEPTH: "4"
  allow_failure: false  # 发现高危漏洞时阻止合并
```

### 2.2 DAST动态应用安全测试

#### OWASP ZAP集成
```yaml
# GitLab CI中的ZAP扫描
zap_scan:
  stage: security
  image: owasp/zap2docker-stable
  variables:
    ZAP_TARGET_URL: "https://staging.example.com"
    ZAP_SCAN_TYPE: "spider"
    ZAP_REPORT_FILE: "zap_report.html"
  artifacts:
    paths:
      - zap_report.html
    when: always
  script:
    - |
      zap-baseline.py \
        -t $ZAP_TARGET_URL \
        -J zap_report.json \
        -r zap_report.html \
        --hook=/zap/auth_hook.py \
        -c auth_config.conf
  rules:
    - if: $CI_MERGE_REQUEST_IID
    - if: $CI_COMMIT_BRANCH == "main"
```

---

## 第三章：容器安全

### 3.1 Trivy镜像扫描

```yaml
# .gitlab-ci.yml
trivy_scan:
  stage: security
  image:
    name: aquasec/trivy:latest
    entrypoint: [""]
  variables:
    TRIVY_REPOSITORY: "docker.io/myapp"
    TRIVY_VERSION: "v1"
    TRIVY_SEVERITY: "HIGH,CRITICAL"
    TRIVY_EXIT_CODE: "1"  # 发现漏洞时失败
    TRIVY_CACHE_DIR: ".trivy"
  cache:
    paths:
      - .trivy/
  before_script:
    - trivy --version
  script:
    - |
      trivy image --exit-code 1 \
        --severity $TRIVY_SEVERITY \
        --format table \
        --output trivy-report.txt \
        $TRIVY_REPOSITORY:$CI_COMMIT_SHA
  after_script:
    - |
      cat trivy-report.txt
  artifacts:
    paths:
      - trivy-report.txt
    when: always
```

### 3.2 Kubernetes安全策略

```yaml
# Pod安全策略
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'secret'
    - 'persistentVolumeClaim'
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  supplementalGroups:
    rule: 'MustRunAs'
    ranges:
      - min: 1000
        max: 65535
  fsGroup:
    rule: 'MustRunAs'
    ranges:
      - min: 1000
        max: 65535

---
# NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: frontend
      ports:
        - protocol: TCP
          port: 8080
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: backend
      ports:
        - protocol: TCP
          port: 8080
```

---

## 第四章：Secrets管理

### 4.1 HashiCorp Vault集成

```python
import hvac
import os

# 初始化Vault客户端
client = hvac.Client(
    url=os.environ.get('VAULT_ADDR'),
    token=os.environ.get('VAULT_TOKEN')
)

# 读取密钥
def get_secret(path: str, key: str) -> str:
    try:
        response = client.secrets.kv.v2.read_secret_version(
            path=path,
            mount_point='secret'
        )
        return response['data']['data'][key]
    except Exception as e:
        raise ValueError(f"Failed to read secret: {e}")

# 写入密钥
def set_secret(path: str, key: str, value: str):
    client.secrets.kv.v2.create_or_update_secret(
        path=path,
        secret={key: value},
        mount_point='secret'
    )

# 使用动态密钥
def get_database_credentials():
    response = client.secrets.database.generate_credentials(
        name="my-postgres-role"
    )
    return {
        'username': response['data']['username'],
        'password': response['data']['password']
    }
```

### 4.2 GitOps Secrets

```yaml
# SOPS加密配置
# .sops.yaml
creation_rules:
  - age: "age1..."  # 接收者公钥
    path_regex: secrets/.*
    encrypted_regex: "^(apiKey|password|secret)$"

# ArgoCD中的Vault插件
# values.yaml
repoServer:
  env:
    - name: VAULT_ADDR
      value: "https://vault.example.com"
  volumes:
    - name: vault-token
      secret:
        secretName: vault-token
  volumeMounts:
    - name: vault-token
      mountPath: /vault/token

# 使用外部Secret
apiVersion: external-secrets.io/v1alpha1
kind: ClusterSecretStore
metadata:
  name: vault-backend
spec:
  provider:
    vault:
      server: "https://vault.example.com"
      version: "v2"
      auth:
        tokenSecretRef:
          name: vault-token
          key: token

---
apiVersion: external-secrets.io/v1alpha1
kind: ExternalSecret
metadata:
  name: database-credentials
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: ClusterSecretStore
  target:
    name: database-credentials
    creationPolicy: Owner
  data:
    - secretKey: username
      remoteRef:
        key: secret/database
        property: username
    - secretKey: password
      remoteRef:
        key: secret/database
        property: password
```

---

## 参考资源

### 官方文档
- OWASP: owasp.org
- NIST CSF: nist.gov
- CIS Benchmarks: cisecurity.org

### 工具文档
- SonarQube: docs.sonarqube.org
- Trivy: aquasecurity.github.io/trivy
- HashiCorp Vault: vaultproject.io/docs

### 认证考试
- CISSP
- CEH (Certified Ethical Hacker)
- OSCP (Offensive Security)
- AWS Security Specialty

---

*本知识文件最后更新：2026-02-07*
