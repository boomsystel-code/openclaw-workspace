# 机器学习核心概念

*精选的机器学习核心概念和算法*

---

## 1. 监督学习

### 1.1 线性模型

**线性回归**：
$$y = wx + b$$

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**逻辑回归**（二分类）：
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)
```

### 1.2 决策树

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)
```

### 1.3 集成学习

**随机森林**：
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

**梯度提升（XGBoost）**：
```python
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
```

### 1.4 支持向量机（SVM）

```python
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0)
model.fit(X_train, y_train)
```

---

## 2. 无监督学习

### 2.1 聚类

**K-means**：
```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(X)
labels = model.labels_
```

**DBSCAN**（密度聚类）：
```python
from sklearn.cluster import DBSCAN
model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X)
```

### 2.2 降维

**PCA**：
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  # 保留95%方差
X_reduced = pca.fit_transform(X)
```

**t-SNE**（非线性降维）：
```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
X_2d = tsne.fit_transform(X)
```

---

## 3. 模型评估

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 交叉验证
scores = cross_val_score(model, X, y, cv=5)

# 评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
```

---

## 4. 特征工程

### 4.1 特征缩放

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 归一化
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
```

### 4.2 编码

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 独热编码
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])])
X_encoded = ct.fit_transform(X)
```

---

## 5. 交叉验证

```python
from sklearn.model_selection import KFold, StratifiedKFold

# K折交叉验证
kfold = KFold(n_splits=5, shuffle=True)

# 分层K折
skfold = StratifiedKFold(n_splits=5, shuffle=True)
```

---

## 6. 超参调优

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# 网格搜索
param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

---

## 7. 偏差-方差权衡

- **高偏差**（欠拟合）：模型太简单
  - 表现：训练误差和测试误差都很高
  - 解决：增加模型复杂度、增加特征

- **高方差**（过拟合）：模型太复杂
  - 表现：训练误差很低，测试误差很高
  - 解决：增加数据、正则化、dropout

---

*机器学习核心概念整理完成！* 📚
