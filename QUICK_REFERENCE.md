# 🚀 快速参考指南

> 最后更新: 2026-02-07
> 常用命令、代码片段速查

---

## 💻 常用命令速查

### Git
| 命令 | 说明 |
|------|------|
| `git init` | 初始化仓库 |
| `git clone <url>` | 克隆仓库 |
| `git add .` | 添加所有文件 |
| `git commit -m "msg"` | 提交 |
| `git push` | 推送到远程 |
| `git pull` | 拉取更新 |
| `git status` | 查看状态 |
| `git checkout -b <branch>` | 创建并切换分支 |

### Docker
| 命令 | 说明 |
|------|------|
| `docker build -t name .` | 构建镜像 |
| `docker run -p 80:80 name` | 运行容器 |
| `docker ps` | 查看运行中的容器 |
| `docker stop <id>` | 停止容器 |
| `docker exec -it <id> bash` | 进入容器 |
| `docker logs -f <id>` | 查看日志 |

### Python
| 命令 | 说明 |
|------|------|
| `python -m venv .venv` | 创建虚拟环境 |
| `source .venv/bin/activate` | 激活环境 |
| `pip install -r req.txt` | 安装依赖 |
| `pip list` | 列出已安装包 |
| `python main.py` | 运行脚本 |

### Kubernetes
| 命令 | 说明 |
|------|------|
| `kubectl get pods` | 查看Pod |
| `kubectl get svc` | 查看服务 |
| `kubectl apply -f file.yaml` | 应用配置 |
| `kubectl delete -f file.yaml` | 删除资源 |
| `kubectl logs <pod>` | 查看日志 |
| `kubectl exec -it <pod> bash` | 进入Pod |

---

## 📝 代码片段库

### Python基础
```python
# 读取JSON
import json
with open('file.json') as f:
    data = json.load(f)

# 写入JSON
with open('file.json', 'w') as f:
    json.dump(data, f, indent=2)

# 列表推导
[x for x in items if x > 0]

# 字典合并
{**dict1, **dict2}

# 装饰器
def timer(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Time: {time.time()-start}s")
        return result
    return wrapper
```

### Pandas数据处理
```python
import pandas as pd

# 读取CSV
df = pd.read_csv('file.csv')

# 基本统计
df.describe()

# 筛选
df[df['column'] > value]

# 分组统计
df.groupby('column').sum()

# 排序
df.sort_values('column', ascending=False)

# 新增列
df['new_col'] = df['col1'] + df['col2']
```

### 数据可视化
```python
import matplotlib.pyplot as plt

# 折线图
plt.plot(x, y)
plt.title('Title')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# 柱状图
plt.bar(labels, values)
plt.title('Title')
plt.show()
```

### API请求
```python
import requests

# GET请求
response = requests.get(url, params={'key': 'value'})
data = response.json()

# POST请求
response = requests.post(url, json={'key': 'value'})
data = response.json()
```

---

## 🔧 快捷键速查

### VS Code
| 快捷键 | 功能 |
|--------|------|
| `Cmd+B` | 侧边栏 |
| `Cmd+Shift+P` | 命令面板 |
| `Cmd+P` | 快速打开文件 |
| `Cmd+Shift+\` | 跳转到匹配括号 |
| `Cmd+Shift+L` | 选择所有匹配 |
| `F12` | 跳转到定义 |

### Chrome
| 快捷键 | 功能 |
|--------|------|
| `Cmd+T` | 新标签页 |
| `Cmd+W` | 关闭标签页 |
| `Cmd+Shift+T` | 恢复关闭的标签 |
| `Cmd+L` | 跳转到地址栏 |
| `Cmd+Option+I` | 开发者工具 |

### macOS通用
| 快捷键 | 功能 |
|--------|------|
| `Cmd+C` | 复制 |
| `Cmd+V` | 粘贴 |
| `Cmd+A` | 全选 |
| `Cmd+Z` | 撤销 |
| `Cmd+Shift+Z` | 重做 |
| `Cmd+S` | 保存 |
| `Cmd+N` | 新建 |
| `Cmd+Q` | 退出 |

---

## 📊 数学公式速查

### 统计指标
| 公式 | 说明 |
|------|------|
| 均值: $\bar{x} = \frac{1}{n}\sum x_i$ | 平均值 |
| 方差: $\sigma^2 = \frac{1}{n}\sum(x_i-\bar{x})^2$ | 离散程度 |
| 标准差: $\sigma = \sqrt{\sigma^2}$ | 方差的平方根 |
| 相关系数: $\rho_{xy} = \frac{Cov(X,Y)}{\sigma_x\sigma_y}$ | 线性相关 |

### 机器学习
| 公式 | 说明 |
|------|------|
| 线性回归: $y = wx + b$ | 预测函数 |
| 损失函数: $MSE = \frac{1}{n}\sum(y_i-\hat{y}_i)^2$ | 均方误差 |
| 激活函数(Sigmoid): $\sigma(x) = \frac{1}{1+e^{-x}}$ | 0-1映射 |
| 交叉熵: $H(p,q) = -\sum p(x)\log q(x)$ | 分类损失 |

---

## 🌍 技术术语速查

### AI/ML
| 术语 | 含义 |
|------|------|
| LLM | 大语言模型 |
| NLP | 自然语言处理 |
| CV | 计算机视觉 |
| RL | 强化学习 |
| RAG | 检索增强生成 |
| Transformer | 注意力机制模型 |

### DevOps
| 术语 | 含义 |
|------|------|
| CI/CD | 持续集成/持续部署 |
| K8s | Kubernetes |
| IaC | 基础设施即代码 |
| SRE | 站点可靠性工程 |

### 金融
| 术语 | 含义 |
|------|------|
| ROI | 投资回报率 |
| CAGR | 复合年增长率 |
| P/E | 市盈率 |
| ATR | 平均真实波幅 |

---

## 📞 常用链接

### 开发资源
| 资源 | 链接 |
|------|------|
| GitHub | https://github.com |
| Stack Overflow | https://stackoverflow.com |
| npm | https://www.npmjs.com |
| PyPI | https://pypi.org |
| Docker Hub | https://hub.docker.com |

### AI资源
| 资源 | 链接 |
|------|------|
| Hugging Face | https://huggingface.co |
| arXiv | https://arxiv.org |
| Papers With Code | https://paperswithcode.com |

### 工具
| 资源 | 链接 |
|------|------|
| MDN Web Docs | https://developer.mozilla.org |
| Python Docs | https://docs.python.org |
| Git Docs | https://git-scm.com/doc |

---

## 🔍 故障排查

### 问题 | 解决方案
------|----------
Python导入错误 | `pip install package_name`
端口被占用 | `lsof -i :8080` → `kill <pid>`
Docker权限错误 | `sudo usermod -aG docker $USER`
Git合并冲突 | `git status` → 编辑冲突 → `git add .` → `git commit`
npm安装慢 | `npm config set registry https://registry.npmmirror.com`

---

*创建时间: 2026-02-07*
