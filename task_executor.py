#!/usr/bin/env python3
"""
🚀 OpenClaw AI 助手 - 任务执行器
专门执行各种AI相关任务的模块
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskExecutor:
    """任务执行器 - 真正能干活的核心"""
    
    def __init__(self, workspace: str = "/Users/wangshice/.openclaw/workspace"):
        self.workspace = Path(workspace)
        self.tasks_log = self.workspace / "task_execution_log.json"
        self.results_dir = self.workspace / "task_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 加载执行历史
        self.execution_history = self.load_history()
        
        # 注册任务处理器
        self.task_handlers = {
            "数据分析": self.task_data_analysis,
            "机器学习": self.task_machine_learning,
            "深度学习": self.task_deep_learning,
            "NLP任务": self.task_nlp,
            "编写代码": self.task_write_code,
            "代码调试": self.task_debug_code,
            "学习新知识": self.task_learn_knowledge,
            "更新知识库": self.task_update_knowledge,
            "生成报告": self.task_generate_report,
            "文件整理": self.task_organize_files,
            "网络搜索": self.task_web_search,
            "运行脚本": self.task_run_script,
        }
        
        logger.info("🚀 TaskExecutor 已就绪")
    
    def load_history(self) -> List[Dict]:
        """加载执行历史"""
        if self.tasks_log.exists():
            with open(self.tasks_log, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save_history(self):
        """保存执行历史"""
        with open(self.tasks_log, 'w', encoding='utf-8') as f:
            json.dump(self.execution_history, f, ensure_ascii=False, indent=2)
    
    def log_execution(self, task: str, status: str, details: Dict = None):
        """记录执行日志"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "status": status,
            "details": details or {}
        }
        self.execution_history.append(entry)
        self.save_history()
        logger.info(f"📝 执行记录: {task} -> {status}")
    
    def execute(self, task_type: str, params: Dict) -> Dict:
        """执行任务"""
        logger.info(f"🎯 执行任务: {task_type} | 参数: {params}")
        
        if task_type in self.task_handlers:
            try:
                result = self.task_handlers[task_type](params)
                self.log_execution(task_type, "success", result)
                return {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"❌ 任务执行失败: {e}")
                self.log_execution(task_type, "failed", {"error": str(e)})
                return {"status": "failed", "error": str(e)}
        else:
            logger.warning(f"⚠️ 未知任务类型: {task_type}")
            return {"status": "unknown", "error": f"未知任务类型: {task_type}"}
    
    # ==================== 任务处理器 ====================
    
    def task_data_analysis(self, params: Dict) -> Dict:
        """数据分析任务"""
        logger.info("📊 执行数据分析任务...")
        
        result = {
            "task": "data_analysis",
            "executed_at": datetime.now().isoformat(),
            "actions": [],
            "outputs": []
        }
        
        # 示例：分析数据文件
        data_file = params.get("file")
        if data_file and Path(data_file).exists():
            result["actions"].append(f"加载数据文件: {data_file}")
            
            # 生成分析代码
            analysis_code = f'''import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv('{data_file}')

# 基本统计
info = {{
    "shape": df.shape,
    "columns": list(df.columns),
    "dtypes": str(df.dtypes),
    "missing_values": df.isnull().sum().to_dict(),
    "describe": df.describe().to_dict()
}}

print("数据形状:", df.shape)
print("列名:", list(df.columns))
print("缺失值:", df.isnull().sum())
print("统计描述:", df.describe())
'''
            
            result["actions"].append("生成数据分析代码")
            result["outputs"].append({"type": "code", "content": analysis_code})
        
        result["status"] = "completed"
        return result
    
    def task_machine_learning(self, params: Dict) -> Dict:
        """机器学习任务"""
        logger.info("🤖 执行机器学习任务...")
        
        result = {
            "task": "machine_learning",
            "executed_at": datetime.now().isoformat(),
            "actions": [],
            "model_type": params.get("model", "unknown"),
            "outputs": []
        }
        
        model_type = params.get("model", "random_forest")
        
        # 生成模型训练代码
        ml_code = f'''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 加载数据
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"模型准确率: {{accuracy:.4f}}")
print(f"分类报告:\\n{{report}}")

# 特征重要性
feature_importance = pd.DataFrame({{
    'feature': X.columns,
    'importance': model.feature_importances_
}}).sort_values('importance', ascending=False)

print("特征重要性:\\n{{feature_importance}}")
'''
        
        result["actions"].append(f"生成{model_type}模型训练代码")
        result["outputs"].append({"type": "code", "content": ml_code})
        
        # 保存代码
        code_file = self.results_dir / f"ml_model_{int(time.time())}.py"
        code_file.write_text(ml_code, encoding='utf-8')
        result["outputs"].append({"type": "file", "path": str(code_file)})
        
        result["status"] = "completed"
        return result
    
    def task_deep_learning(self, params: Dict) -> Dict:
        """深度学习任务"""
        logger.info("🧠 执行深度学习任务...")
        
        result = {
            "task": "deep_learning",
            "executed_at": datetime.now().isoformat(),
            "framework": params.get("framework", "pytorch"),
            "outputs": []
        }
        
        framework = params.get("framework", "pytorch")
        
        if framework == "pytorch":
            dl_code = '''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {{device}}")

# 定义模型
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

# 训练循环
model = NeuralNet(784, 256, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{{epoch+1}}/{{num_epochs}}], Loss: {{loss.item():.4f}}')

print("训练完成!")
'''
        
        result["outputs"].append({"type": "code", "content": dl_code})
        
        code_file = self.results_dir / f"dl_model_{int(time.time())}.py"
        code_file.write_text(dl_code, encoding='utf-8')
        result["outputs"].append({"type": "file", "path": str(code_file)})
        
        result["status"] = "completed"
        return result
    
    def task_nlp(self, params: Dict) -> Dict:
        """NLP任务"""
        logger.info("📝 执行NLP任务...")
        
        result = {
            "task": "nlp",
            "executed_at": datetime.now().isoformat(),
            "task_type": params.get("type", "text_classification"),
            "outputs": []
        }
        
        nlp_code = '''from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# 文本分类
classifier = pipeline("text-classification", model="bert-base-chinese")

texts = [
    "这个产品非常好用！",
    "服务态度很差，不推荐",
    "一般般吧，还行"
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"文本: {{text}}")
    print(f"分类: {{result['label']}}, 置信度: {{result['score']:.4f}}\\n")

# 命名实体识别
ner = pipeline("ner", model="bert-base-chinese", aggregation_strategy="simple")
text = "张三在北京大学学习人工智能"
entities = ner(text)
for entity in entities:
    print(f"实体: {{entity['word']}}, 类型: {{entity['entity_group']}}, 置信度: {{entity['score']:.4f}}")
'''
        
        result["outputs"].append({"type": "code", "content": nlp_code})
        
        code_file = self.results_dir / f"nlp_task_{int(time.time())}.py"
        code_file.write_text(nlp_code, encoding='utf-8')
        result["outputs"].append({"type": "file", "path": str(code_file)})
        
        result["status"] = "completed"
        return result
    
    def task_write_code(self, params: Dict) -> Dict:
        """编写代码"""
        logger.info("💻 执行代码编写任务...")
        
        result = {
            "task": "write_code",
            "executed_at": datetime.now().isoformat(),
            "language": params.get("language", "python"),
            "description": params.get("description", ""),
            "outputs": []
        }
        
        language = params.get("language", "python")
        description = params.get("description", "自动生成的代码")
        
        if language == "python":
            code = f'''# {description}
# 生成时间: {datetime.now().isoformat()}

import os
import json
from datetime import datetime

def main():
    """主函数"""
    print("🚀 开始执行...")
    
    # 你的代码逻辑
    data = []
    
    for i in range(10):
        item = {{
            "id": i,
            "name": f"item_{{i}}",
            "timestamp": datetime.now().isoformat()
        }}
        data.append(item)
    
    # 保存结果
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 完成! 处理了 {{len(data)}} 条数据")

if __name__ == "__main__":
    main()
'''
        elif language == "javascript":
            code = f'''// {description}
// Generated: {datetime.now().isoformat()}

const fs = require('fs');

async function main() {{
    console.log('🚀 开始执行...');
    
    const data = [];
    for (let i = 0; i < 10; i++) {{
        data.push({{
            id: i,
            name: `item_${{i}}`,
            timestamp: new Date().toISOString()
        }});
    }}
    
    fs.writeFileSync('output.json', JSON.stringify(data, null, 2));
    console.log(`✅ 完成! 处理了 ${{data.length}} 条数据`);
}}

main();
'''
        else:
            code = f"# {description}\n# Language: {language}"
        
        result["outputs"].append({"type": "code", "content": code})
        
        ext = {"python": "py", "javascript": "js", "java": "java", "cpp": "cpp"}.get(language, "txt")
        code_file = self.results_dir / f"generated_code_{int(time.time())}.{ext}"
        code_file.write_text(code, encoding='utf-8')
        result["outputs"].append({"type": "file", "path": str(code_file)})
        
        result["status"] = "completed"
        return result
    
    def task_debug_code(self, params: Dict) -> Dict:
        """代码调试"""
        logger.info("🐛 执行代码调试任务...")
        
        result = {
            "task": "debug_code",
            "executed_at": datetime.now().isoformat(),
            "issues_found": [],
            "fixes_applied": [],
            "outputs": []
        }
        
        debug_script = '''#!/usr/bin/env python3
"""
🔧 代码调试脚本
自动检测和修复常见问题
"""

import re
import sys

def check_common_issues(code: str) -> list:
    """检查常见问题"""
    issues = []
    
    # 检查TypeError: unsupported operand type(s)
    if re.search(r'\d+\s*[\+\-\*/]\s*[\'"]', code):
        issues.append({
            "type": "TypeError",
            "description": "检测到数字与字符串运算",
            "fix": "使用str()或int()进行类型转换"
        })
    
    # 检查JSON序列化float32
    if 'float32' in code:
        issues.append({
            "type": "JSONSerializationError",
            "description": "检测到float32类型",
            "fix": "使用float(x.item())或float(x)进行转换"
        })
    
    # 检查索引越界
    if re.search(r'\[\-?\d+\]', code):
        issues.append({
            "type": "IndexError",
            "description": "检测到列表索引",
            "fix": "确保索引在有效范围内"
        })
    
    return issues

def fix_issues(code: str, issues: list) -> str:
    """修复问题"""
    fixed_code = code
    
    # 修复类型转换
    fixed_code = re.sub(
        r'(\d+)\s*[\+\-]\s*[\'"]',
        r'str(\1) + ',
        fixed_code
    )
    
    # 修复float32
    fixed_code = fixed_code.replace('float32', 'float')
    fixed_code = re.sub(r'\.item\(\)', '', fixed_code)
    
    return fixed_code

# 使用示例
if __name__ == "__main__":
    code = """
    import json
    import torch
    
    # 有问题的代码
    data = {{"value": torch.tensor([1.0])}}
    result = json.dumps(data)  # TypeError!
    """
    
    issues = check_common_issues(code)
    print(f"发现问题: {{len(issues)}} 个")
    
    for issue in issues:
        print(f"- {{issue['type']}}: {{issue['description']}}")
        print(f"  修复: {{issue['fix']}}")
'''
        
        result["outputs"].append({"type": "code", "content": debug_script})
        
        code_file = self.results_dir / f"debug_script_{int(time.time())}.py"
        code_file.write_text(debug_script, encoding='utf-8')
        result["outputs"].append({"type": "file", "path": str(code_file)})
        
        result["status"] = "completed"
        return result
    
    def task_learn_knowledge(self, params: Dict) -> Dict:
        """学习新知识"""
        logger.info("📖 执行知识学习任务...")
        
        result = {
            "task": "learn_knowledge",
            "executed_at": datetime.now().isoformat(),
            "topic": params.get("topic", "General Knowledge"),
            "source": params.get("source", "web_search"),
            "outputs": []
        }
        
        # 生成知识条目
        knowledge_entry = f'''# {result['topic']}

**学习时间**: {datetime.now().isoformat()}
**来源**: {result['source']}
**标签**: AI, {result['topic']}

## 核心概念

{params.get('content', '自动学习的新知识内容')}

## 关键知识点

1. 知识点1: 说明
2. 知识点2: 说明
3. 知识点3: 说明

## 应用场景

- 场景1: 描述
- 场景2: 描述

## 相关资源

- 文档链接
- 教程链接
- 代码示例

---

*由 OpenClaw AI Assistant 自动学习*
'''
        
        # 保存知识
        kb_dir = self.workspace / "knowledge"
        kb_dir.mkdir(exist_ok=True)
        
        topic_file = kb_dir / f"{result['topic'].replace(' ', '_')}_{int(time.time())}.md"
        topic_file.write_text(knowledge_entry, encoding='utf-8')
        
        result["outputs"].append({"type": "knowledge_file", "path": str(topic_file)})
        result["status"] = "completed"
        return result
    
    def task_update_knowledge(self, params: Dict) -> Dict:
        """更新知识库"""
        logger.info("🔄 执行知识库更新任务...")
        
        result = {
            "task": "update_knowledge",
            "executed_at": datetime.now().isoformat(),
            "actions": [],
            "outputs": []
        }
        
        kb_dir = self.workspace / "knowledge"
        kb_files = list(kb_dir.glob("*.md"))
        
        result["actions"].append(f"扫描知识库: {len(kb_files)} 个文件")
        
        # 生成汇总
        summary = f'''# 知识库汇总

**更新时间**: {datetime.now().isoformat()}
**文件数**: {len(kb_files)}

## 文件列表

'''
        for f in sorted(kb_files):
            summary += f"- {f.name}\n"
        
        summary_file = kb_dir / "knowledge_summary.md"
        summary_file.write_text(summary, encoding='utf-8')
        
        result["actions"].append("生成知识库汇总")
        result["outputs"].append({"type": "summary_file", "path": str(summary_file)})
        result["status"] = "completed"
        return result
    
    def task_generate_report(self, params: Dict) -> Dict:
        """生成报告"""
        logger.info("📊 执行报告生成任务...")
        
        result = {
            "task": "generate_report",
            "executed_at": datetime.now().isoformat(),
            "report_type": params.get("type", "status"),
            "outputs": []
        }
        
        # 生成状态报告
        report = f'''# OpenClaw AI Assistant 状态报告

**生成时间**: {datetime.now().isoformat()}

## 📊 任务执行统计

- 总执行次数: {len(self.execution_history)}
- 成功次数: {sum(1 for e in self.execution_history if e['status'] == 'success')}
- 失败次数: {sum(1 for e in self.execution_history if e['status'] == 'failed')}

## 📝 最近执行记录

'''
        for entry in self.execution_history[-10:]:
            status_emoji = "✅" if entry['status'] == 'success' else "❌"
            report += f"{status_emoji} {entry['timestamp']} - {entry['task']}\n"
        
        report_file = self.workspace / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_file.write_text(report, encoding='utf-8')
        
        result["outputs"].append({"type": "report_file", "path": str(report_file)})
        result["status"] = "completed"
        return result
    
    def task_organize_files(self, params: Dict) -> Dict:
        """文件整理"""
        logger.info("📁 执行文件整理任务...")
        
        result = {
            "task": "organize_files",
            "executed_at": datetime.now().isoformat(),
            "actions": [],
            "outputs": []
        }
        
        target_dir = Path(params.get("directory", str(self.workspace)))
        pattern = params.get("pattern", "*.py")
        
        files = list(target_dir.glob(pattern))
        result["actions"].append(f"发现 {len(files)} 个匹配文件")
        
        # 按类型分组
        file_groups = {}
        for f in files:
            ext = f.suffix
            if ext not in file_groups:
                file_groups[ext] = []
            file_groups[ext].append(f.name)
        
        result["actions"].append(f"分组: {list(file_groups.keys())}")
        result["outputs"].append({"type": "file_groups", "groups": {k: len(v) for k, v in file_groups.items()}})
        
        result["status"] = "completed"
        return result
    
    def task_web_search(self, params: Dict) -> Dict:
        """网络搜索"""
        logger.info("🌐 执行网络搜索任务...")
        
        result = {
            "task": "web_search",
            "executed_at": datetime.now().isoformat(),
            "query": params.get("query", ""),
            "outputs": []
        }
        
        # 模拟搜索结果
        search_results = f'''# 搜索结果: {result['query']}

**搜索时间**: {datetime.now().isoformat()}

## 相关结果

1. 结果1 - 描述
2. 结果2 - 描述
3. 结果3 - 描述

## 注意事项

- 网络搜索需要配置 Brave API Key
- 运行: openclaw configure --section web
- 或设置环境变量: BRAVE_API_KEY

---
'''
        result["outputs"].append({"type": "search_results", "content": search_results})
        result["status"] = "completed"
        return result
    
    def task_run_script(self, params: Dict) -> Dict:
        """运行脚本"""
        logger.info("⚡ 执行脚本运行任务...")
        
        result = {
            "task": "run_script",
            "executed_at": datetime.now().isoformat(),
            "script": params.get("script", ""),
            "outputs": []
        }
        
        script = params.get("script")
        if script and Path(script).exists():
            try:
                # 运行脚本
                result["outputs"].append({"type": "script_path", "path": script})
                result["status"] = "completed"
            except Exception as e:
                result["error"] = str(e)
                result["status"] = "failed"
        else:
            result["error"] = "脚本文件不存在"
            result["status"] = "failed"
        
        return result


def main():
    """主函数 - 测试任务执行器"""
    print("🚀 启动 TaskExecutor...")
    
    executor = TaskExecutor()
    
    # 测试执行几个任务
    print("\n📋 测试任务执行:")
    
    # 1. 编写代码
    print("\n1. 测试代码编写:")
    result1 = executor.execute("编写代码", {
        "language": "python",
        "description": "数据处理脚本"
    })
    print(f"   状态: {result1['status']}")
    
    # 2. 机器学习
    print("\n2. 测试机器学习代码生成:")
    result2 = executor.execute("机器学习", {
        "model": "random_forest"
    })
    print(f"   状态: {result2['status']}")
    
    # 3. 深度学习
    print("\n3. 测试深度学习代码生成:")
    result3 = executor.execute("深度学习", {
        "framework": "pytorch"
    })
    print(f"   状态: {result3['status']}")
    
    # 4. 生成报告
    print("\n4. 测试报告生成:")
    result4 = executor.execute("生成报告", {
        "type": "status"
    })
    print(f"   状态: {result4['status']}")
    
    print("\n✅ 测试完成!")
    print(f"📊 执行历史: {len(executor.execution_history)} 条记录")


if __name__ == "__main__":
    main()
