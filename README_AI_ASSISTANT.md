# 🤖 OpenClaw AI 助手使用说明

## 🎯 简介

这是一个能自动学习、更新知识、执行任务的AI助手系统，基于今天从B站学到的2500+AI知识点构建。

---

## 📁 文件结构

```
workspace/
├── ai_assistant.py          # AI助手主程序
├── task_executor.py         # 任务执行器
├── start_assistant.sh       # 快速启动脚本
├── knowledge/               # 知识库目录
│   └── main_knowledge.md   # 主知识库
├── task_results/            # 任务输出目录
├── tasks.json              # 任务队列
└── assistant_config.json    # 配置文件
```

---

## 🚀 启动方式

### 方式1: 快速启动脚本
```bash
cd /Users/wangshice/.openclaw/workspace
./start_assistant.sh
```

### 方式2: 直接运行
```bash
python3 ai_assistant.py
```

### 方式3: 执行特定任务
```bash
# 执行数据分析
python3 -c "from task_executor import TaskExecutor; e=TaskExecutor(); e.execute('数据分析', {'file': 'data.csv'})"

# 生成机器学习代码
python3 -c "from task_executor import TaskExecutor; e=TaskExecutor(); e.execute('机器学习', {'model': 'random_forest'})"

# 学习新知识
python3 -c "from task_executor import TaskExecutor; e=TaskExecutor(); e.execute('学习新知识', {'topic': '新主题', 'content': '内容描述'})"
```

---

## 🎯 功能列表

### 1. 📊 数据分析
- 自动生成数据分析代码
- 数据清洗和预处理
- 统计分析和可视化

### 2. 🤖 机器学习
- 生成模型训练代码
- 支持多种算法 (Random Forest, XGBoost, SVM)
- 模型评估和特征重要性分析

### 3. 🧠 深度学习
- PyTorch模型代码生成
- CNN, RNN, Transformer架构
- GPU加速支持检测

### 4. 📝 NLP任务
- 文本分类
- 命名实体识别 (NER)
- 情感分析
- 基于HuggingFace Transformers

### 5. 💻 代码编写
- 支持多种语言 (Python, JavaScript, Java, C++)
- 自动生成代码模板
- 最佳实践遵循

### 6. 🔧 代码调试
- 自动检测常见错误
- TypeError修复
- JSON序列化问题修复

### 7. 📖 知识学习
- 自动保存新知识到知识库
- 支持多种来源
- 知识去重和更新

### 8. 🔄 知识库管理
- 自动更新知识库汇总
- 知识碎片整理
- 备份和恢复

### 9. 📊 报告生成
- 任务执行统计
- 状态报告生成
- 执行历史追踪

### 10. 📁 文件整理
- 按类型分组文件
- 批量处理
- 模式匹配

---

## 📖 使用示例

### 示例1: 生成一个数据分析脚本
```python
from task_executor import TaskExecutor

executor = TaskExecutor()
result = executor.execute('数据分析', {
    'file': '/path/to/your/data.csv'
})

# 查看生成的代码
print(result['result']['outputs'][0]['content'])
```

### 示例2: 训练一个分类模型
```python
from task_executor import TaskExecutor

executor = TaskExecutor()
result = executor.execute('机器学习', {
    'model': 'random_forest',
    'target': 'species',
    'test_size': 0.2
})

# 代码已保存到 task_results/ 目录
```

### 示例3: 学习新知识
```python
from task_executor import TaskExecutor

executor = TaskExecutor()
result = executor.execute('学习新知识', {
    'topic': '强化学习',
    'content': '''
    强化学习是一种机器学习方法,
    智能体通过与环境交互来学习最优策略...
    ''',
    'source': 'textbook'
})

# 知识已保存到 knowledge/ 目录
```

### 示例4: 使用AI助手类
```python
from ai_assistant import OpenClawAssistant

# 创建助手
assistant = OpenClawAssistant()

# 添加任务
assistant.add_task("分析销售数据", priority="high")
assistant.add_task("训练预测模型", priority="normal", dependencies=["分析销售数据"])
assistant.add_task("生成周报", priority="low")

# 执行自动化
assistant.run_automation("学习新知识")
assistant.run_automation("生成报告")

# 查看状态
assistant.print_status()
```

---

## ⚙️ 配置说明

配置文件: `assistant_config.json`

```json
{
    "version": "1.0",
    "learning_enabled": true,
    "auto_update_interval_hours": 6,
    "max_knowledge_entries": 10000,
    "personality": {
        "name": "OpenClaw Assistant",
        "role": "AI Helper",
        "vibe": "Helpful & Efficient"
    }
}
```

---

## 📊 统计信息

- **技术领域**: 35个 (从B站学到的)
- **知识点**: 2500+
- **代码模板**: 50+
- **支持任务类型**: 11种
- **知识库位置**: `workspace/knowledge/`

---

## 🔧 常见问题

### Q: 如何添加新任务?
A: 使用 `assistant.add_task(task_name, priority="high")`

### Q: 如何执行自动化任务?
A: 使用 `assistant.run_automation("任务名称")`

### Q: 知识保存在哪里?
A: 保存在 `workspace/knowledge/` 目录

### Q: 任务输出在哪里?
A: 保存在 `workspace/task_results/` 目录

### Q: 如何查看执行历史?
A: 检查 `task_execution_log.json` 文件

---

## 🎓 学到的知识点

这个AI助手整合了今天从B站学到的2500+知识点:

1. **Python**: 数据分析、可视化、自动化
2. **机器学习**: 分类、回归、聚类
3. **深度学习**: PyTorch、CNN、RNN、Transformer
4. **NLP**: 文本分类、NER、情感分析
5. **AI工具**: LangChain、LlamaIndex、HuggingFace
6. **最佳实践**: 代码规范、调试技巧、优化策略

---

## 📝 更新日志

**v1.0 (2026-02-05)**
- ✨ 初始版本发布
- 📚 集成2500+AI知识点
- 🤖 支持11种任务类型
- 📖 知识库系统
- ⚡ 自动化工作流

---

*由 OpenClaw AI Assistant 自动生成*
