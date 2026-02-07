# Jenkins与CI/CD实战

## 第一章：Jenkins基础

### 1.1 Jenkins架构

#### 主从架构
```
┌─────────────────────────────────────────────────────────────┐
│                      Jenkins Master                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Web UI    │  │  Scheduler  │  │  Build      │         │
│  │  (用户界面)  │  │  (调度器)   │  │  Queue     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                │                │
│         └────────────────┼────────────────┘                │
│                          │                                │
│                   ┌──────┴──────┐                        │
│                   │  API Server  │                        │
│                   └──────┬──────┘                        │
└──────────────────────────┼────────────────────────────────┘
                           │
                           │ SSH / JNLP
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     Jenkins Agents                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Agent 1     │  │ Agent 2     │  │ Agent 3     │         │
│  │ (Linux)     │  │ (Windows)   │  │ (Docker)    │         │
│  │ 2 executors │  │ 1 executor  │  │ 4 executors │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Jenkinsfile基础

```groovy
// Jenkinsfile (Declarative Pipeline)
pipeline {
    agent any
    
    environment {
        APP_NAME = 'my-application'
        DOCKER_REGISTRY = 'docker.io'
        VERSION = '1.0.0'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    commitHash = sh(
                        script: 'git rev-parse HEAD',
                        returnStdout: true
                    ).trim()
                }
            }
        }
        
        stage('Build') {
            steps {
                echo "Building ${APP_NAME}..."
                sh 'npm install'
                sh 'npm run build'
            }
        }
        
        stage('Test') {
            steps {
                echo "Running tests..."
                sh 'npm run test -- --coverage'
            }
            post {
                always {
                    junit '**/test-results/*.xml'
                    publishHTML([
                        reportDir: 'coverage',
                        reportFiles: 'index.html',
                        reportName: 'Coverage Report'
                    ])
                }
            }
        }
        
        stage('Security Scan') {
            steps {
                echo "Running security scans..."
                sh 'trivy image ${APP_NAME}:${VERSION}'
            }
        }
        
        stage('Deploy to Staging') {
            when {
                branch 'develop'
            }
            steps {
                echo "Deploying to staging..."
                sh 'kubectl set image deployment/${APP_NAME} ${APP_NAME}=${DOCKER_REGISTRY}/${APP_NAME}:${VERSION}'
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                echo "Deploying to production..."
                input message: 'Deploy to production?'
                sh 'kubectl set image deployment/${APP_NAME} ${APP_NAME}=${DOCKER_REGISTRY}/${APP_NAME}:${VERSION}'
            }
        }
    }
    
    post {
        success {
            echo "Pipeline completed successfully!"
            office365ConnectorSend(
                webhookUrl: "${OFFICE365_WEBHOOK}",
                messageText: "Build succeeded: ${env.BUILD_URL}"
            )
        }
        failure {
            echo "Pipeline failed!"
            office365ConnectorSend(
                webhookUrl: "${OFFICE365_WEBHOOK}",
                messageText: "Build failed: ${env.BUILD_URL}"
            )
        }
        always {
            cleanWs()
        }
    }
}
```

---

## 第二章：高级Pipeline

### 2.1 矩阵构建

```groovy
// Matrix Pipeline
pipeline {
    agent none
    
    environment {
        APP_NAME = 'cross-platform-app'
    }
    
    stages {
        stage('Build Matrix') {
            matrix {
                axes {
                    axis {
                        name 'PLATFORM'
                        values 'linux', 'windows', 'macos'
                    }
                    axis {
                        name 'NODE_TYPE'
                        values 'standard', 'high-memory'
                    }
                }
                excludes {
                    exclude {
                        axis {
                            name 'PLATFORM'
                            values 'windows'
                        }
                        axis {
                            name 'NODE_TYPE'
                            values 'high-memory'
                        }
                    }
                }
                agent {
                    label "${PLATFORM}"
                }
                stages {
                    stage('Build') {
                        steps {
                            echo "Building on ${PLATFORM} with ${NODE_TYPE}"
                            sh "make build PLATFORM=${PLATFORM}"
                        }
                    }
                    stage('Test') {
                        steps {
                            echo "Testing on ${PLATFORM}..."
                            sh "make test PLATFORM=${PLATFORM}"
                        }
                    }
                }
            }
        }
    }
}
```

### 2.2 共享库

```groovy
// vars/deploy.groovy
def call(Map config) {
    pipeline {
        agent any
        
        stages {
            stage('Deploy') {
                steps {
                    script {
                        echo "Deploying ${config.app} to ${config.env}"
                        
                        switch(config.env) {
                            case 'staging':
                                sh "kubectl apply -f k8s/staging/"
                                break
                            case 'production':
                                sh "kubectl apply -f k8s/production/"
                                break
                            default:
                                error "Unknown environment: ${config.env}"
                        }
                    }
                }
            }
            
            stage('Smoke Test') {
                steps {
                    sh "kubectl run test-pod --image=${config.app}:${config.version} --restart=Never -- ${config.testCommand}"
                }
            }
        }
    }
}

// vars/notify.groovy
def success(String channel) {
    office365ConnectorSend(
        webhookUrl: "${OFFICE365_WEBHOOK}",
        messageText: "✅ Build succeeded: ${env.BUILD_URL}"
    )
}

def failure(String channel) {
    office365ConnectorSend(
        webhookUrl: "${OFFICE365_WEBHOOK}",
        messageText: "❌ Build failed: ${env.BUILD_URL}"
    )
}
```

---

## 第三章：Blue Ocean与可视化

### 3.1 Blue Ocean配置

```groovy
// Jenkinsfile with Blue Ocean优化
pipeline {
    agent any
    
    options {
        buildDiscarder(logRotator(
            numToKeepStr: '10',
            artifactNumToKeepStr: '5'
        ))
        timeout(time: 1, unit: 'HOURS')
        disableConcurrentBuilds()
        skipDefaultCheckout()
    }
    
    stages {
        stage('Initialize') {
            steps {
                echo 'Initializing build environment...'
                sh 'printenv | sort'
            }
        }
        
        stage('Build') {
            steps {
                echo "Building application..."
                sh 'make build'
            }
        }
        
        stage('Unit Tests') {
            steps {
                echo 'Running unit tests...'
                sh 'make test-unit'
            }
        }
        
        stage('Integration Tests') {
            steps {
                echo 'Running integration tests...'
                sh 'make test-integration'
            }
        }
        
        stage('Package') {
            steps {
                echo 'Packaging application...'
                sh 'make package'
                archiveArtifacts artifacts: 'dist/**/*', fingerprint: true
            }
        }
    }
}
```

---

## 第四章：Jenkins最佳实践

### 4.1 性能优化

```groovy
// 使用并行阶段优化构建时间
pipeline {
    agent any
    
    stages {
        stage('Build') {
            steps {
                parallel(
                    'Frontend': {
                        sh 'cd frontend && npm install && npm run build'
                    },
                    'Backend': {
                        sh 'cd backend && go build -o app'
                    },
                    'Database': {
                        sh 'make db-migrations'
                    }
                )
            }
        }
        
        stage('Test') {
            steps {
                parallel(
                    'Unit Tests': {
                        sh 'make test-unit'
                    },
                    'Integration Tests': {
                        sh 'make test-integration'
                    },
                    'E2E Tests': {
                        sh 'make test-e2e'
                    }
                )
            }
        }
    }
}

// 增量构建
pipeline {
    agent {
        label 'docker'
    }
    
    stages {
        stage('Detect Changes') {
            steps {
                script {
                    changes = sh(
                        script: 'git diff --name-only HEAD~1..HEAD',
                        returnStdout: true
                    ).trim()
                    env.CHANGED_FILES = changes ?: ''
                }
            }
        }
        
        stage('Build Changed') {
            steps {
                script {
                    if (env.CHANGED_FILES.contains('frontend/')) {
                        sh 'cd frontend && npm install && npm run build'
                    }
                    if (env.CHANGED_FILES.contains('backend/')) {
                        sh 'cd backend && go build'
                    }
                }
            }
        }
    }
}
```

### 4.2 安全性配置

```groovy
// Jenkins安全性配置
pipeline {
    agent any
    
    options {
        // 禁用脚本安全（仅演示，生产环境应谨慎）
        // scriptApproval.addSignature('*')
        
        // 构建参数
        parameters {
            choice(
                name: 'DEPLOY_ENV',
                choices: ['staging', 'production'],
                description: 'Select deployment environment'
            )
            booleanParam(
                name: 'RUN_SECURITY_SCAN',
                defaultValue: true,
                description: 'Run security scan'
            )
        }
    }
    
    stages {
        stage('Security Scan') {
            when {
                expression { params.RUN_SECURITY_SCAN }
            }
            steps {
                sh '''
                    # SAST扫描
                    semgrep --config=auto .
                    
                    # 依赖扫描
                    safety check -r requirements.txt
                    
                    # 容器扫描
                    trivy image ${IMAGE_NAME}
                '''
            }
        }
        
        stage('Credentials Check') {
            steps {
                script {
                    // 避免在日志中打印敏感信息
                    withCredentials([string(credentialsId: 'api-token', variable: 'API_TOKEN')]) {
                        sh 'echo $API_TOKEN | wc -c'  // 不打印实际值
                    }
                }
            }
        }
    }
}
```

---

## 参考资源

### 官方文档
- Jenkins: www.jenkins.io/doc
- Jenkinsfile: www.jenkins.io/doc/book/pipeline/jenkinsfile/
- Blue Ocean: www.jenkins.io/doc/book/blueocean/

### 进阶资源
- Pipeline Syntax Reference
- Jenkins Plugin Documentation
- Jenkins Community Wiki

### 最佳实践
- Jenkins Best Practices
- Pipeline Groovy Syntax
- Shared Library Development

---

*本知识文件最后更新：2026-02-07*
