# MLOps机器学习工程化实战

## 第一章：MLOps基础架构

### 1.1 MLOps核心概念

#### MLOps生命周期
```
┌──────────────────────────────────────────────────────────────┐
│                      MLOps 生命周期                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│  │  Design │───▶│  Develop│───▶│ Deploy  │───▶│ Monitor │ │
│  │   (设计)│    │  (开发) │    │ (部署)  │    │  (监控) │ │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘ │
│       │              │              │              │        │
│       ▼              ▼              ▼              ▼        │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│  │  业务   │    │ 特征工程 │    │  模型   │    │ 性能    │ │
│  │  理解   │    │  管道   │    │ 服务    │    │ 监控    │ │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘ │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

#### MLOps vs DevOps
```
MLOps额外考虑：
- 数据版本管理
- 模型版本管理
- 特征存储
- 模型注册
- A/B测试
- 模型漂移检测
- 数据质量监控
```

### 1.2 MLflow实验跟踪

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# 设置MLflow追踪服务器
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("classification_experiment")

# 启动实验运行
with mlflow.start_run(run_name="random_forest_v1") as run:
    # 记录参数
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("random_state", 42)
    
    # 训练模型
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # 记录指标
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    
    # 记录模型
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # 记录工件
    mlflow.log_artifact("feature_importance.png")
    
    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")
```

---

## 第二章：模型部署与服务化

### 2.1 FastAPI模型服务

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="ML Prediction API",
    description="机器学习模型预测服务",
    version="1.0.0"
)

# 加载模型
model = joblib.load("models/gradient_boosting.pkl")
scaler = joblib.load("models/scaler.pkl")

# 请求模型
class PredictionRequest(BaseModel):
    features: list[float]

# 响应模型
class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    confidence: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # 特征预处理
        features = np.array(request.features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # 预测
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        confidence = float(max(probability))
        
        # 解释置信度
        if confidence >= 0.9:
            level = "very_high"
        elif confidence >= 0.7:
            level = "high"
        elif confidence >= 0.5:
            level = "medium"
        else:
            level = "low"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=confidence,
            confidence=level
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/model/info")
async def model_info():
    return {
        "model_type": type(model).__name__,
        "n_features": model.n_features_in_,
        "classes": model.classes_.tolist()
    }
```

### 2.2 KServe模型服务

```yaml
# inference.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: ml-model-service
  namespace: ml-models
spec:
  predictor:
    sklearn:
      storageUri: gs://model-bucket/ml-model/
      resources:
        requests:
          cpu: "1"
          memory: "2Gi"
        limits:
          cpu: "2"
          memory: "4Gi"
      readinessProbe:
        initialDelaySeconds: 30
        periodSeconds: 10
      livenessProbe:
        initialDelaySeconds: 60
        periodSeconds: 30
  transformer:
    custom:
      container:
        image: ml-model-transformer:latest
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"

---
# 批量预测
apiVersion: batchpredict.kserve.io/v1alpha1
kind: BatchJob
metadata:
  name: batch-prediction-job
spec:
  model:
    name: ml-model-service
  inputs:
    source: gs://input-data/batch-input.json
    format: json
  outputs:
    destination: gs://output-data/batch-output/
  parallelism: 3
  maxRetryCount: 3
```

---

## 第三章：特征存储

### 3.1 Feast特征存储

```python
from feast import FeatureStore
import pandas as pd

# 初始化特征存储
store = FeatureStore(repo_path="feature_repo/")

# 定义特征视图
# feature_repo/user_features.py
from feast import Entity, FeatureView, ValueType
from feast.data_source import FileSource

user = Entity(name="user_id", value_type=ValueType.INT64)

user_stats_source = FileSource(
    path="data/user_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

user_stats_fv = FeatureView(
    name="user_stats",
    entities=["user_id"],
    features=[
        Feature(name="total_orders", dtype=ValueType.INT64),
        Feature(name="average_order_value", dtype=ValueType.FLOAT),
        Feature(name="last_order_date", dtype=ValueType.STRING),
        Feature(name="user_tier", dtype=ValueType.STRING),
    ],
    source=user_stats_source,
    online=True
)

# 注册特征
store.apply([user, user_stats_source, user_stats_fv])

# 获取特征
feature_service = store.get_service(
    name="user_service",
    features=[
        "user_stats:total_orders",
        "user_stats:average_order_value"
    ]
)

# 训练时获取特征
training_df = store.get_historical_features(
    entity_df=entity_df,
    feature_refs=[
        "user_stats:total_orders",
        "user_stats:average_order_value"
    ]
).to_df()

# 在线服务获取特征
online_features = store.get_online_features(
    feature_refs=[
        "user_stats:total_orders",
        "user_stats:average_order_value"
    ],
    entity_rows=[{"user_id": 1001}, {"user_id": 1002}]
)
```

---

## 第四章：模型监控

### 4.1 Evidently AI监控

```python
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, NumTargetDriftTab, RegressionPerformanceTab
from evidently.pipeline.column_mapping import ColumnMapping

# 数据漂移检测
column_mapping = ColumnMapping()
column_mapping.target = "target"
column_mapping.prediction = "prediction"
column_mapping.numerical_features = ["feature_1", "feature_2"]

data_drift_dashboard = Dashboard(tabs=[
    DataDriftTab(),
    NumTargetDriftTab()
])

data_drift_dashboard.calculate(
    reference_data=reference_df,
    current_data=current_df,
    column_mapping=column_mapping
)

data_drift_dashboard.save("reports/data_drift_report.html")

# 模型性能监控
performance_dashboard = Dashboard(tabs=[
    RegressionPerformanceTab()
])

performance_dashboard.calculate(
    reference_data=train_df,
    current_data=test_df,
    column_mapping=column_mapping
)

performance_dashboard.save("reports/model_performance.html")
```

### 4.2 Prometheus监控

```python
from prometheus_client import Counter, Histogram, Summary, start_http_server
import random

# 定义指标
PREDICTION_COUNTER = Counter(
    'model_predictions_total',
    'Total number of predictions',
    ['model_name', 'prediction_class']
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_name'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

MODEL_ACCURACY = Summary(
    'model_accuracy',
    'Model accuracy score',
    ['model_name']
)

# 启动监控服务器
start_http_server(8000)

# 在预测函数中记录指标
def predict_with_metrics(features):
    start_time = time.time()
    
    prediction = model.predict(features)
    
    latency = time.time() - start_time
    PREDICTION_LATENCY.labels(model_name='classifier').observe(latency)
    
    PREDICTION_COUNTER.labels(
        model_name='classifier',
        prediction_class=str(prediction)
    ).inc()
    
    return prediction

# 定期更新准确率
def update_accuracy_score(model_name, accuracy):
    MODEL_ACCURACY.labels(model_name=model_name).observe(accuracy)
```

---

## 参考资源

### 官方文档
- MLflow: mlflow.org/docs
- Kubeflow: kubeflow.org/docs
- Feast: feast.dev/docs
- KServe: kserve.github.io

### 进阶资源
- Google MLOps Guidelines
- Microsoft MLOps Framework
- AWS Sagemaker Best Practices

### 开源工具
- MLflow, Kubeflow, Feast
- DVC (Data Version Control)
- Weights & Biases, Comet ML

---

*本知识文件最后更新：2026-02-07*
