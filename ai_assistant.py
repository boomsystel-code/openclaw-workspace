#!/usr/bin/env python3
"""
🤖 OpenClaw AI Assistant - 自我进化版
一个能自动学习、更新知识、执行任务的AI助手
"""

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenClawAssistant:
    """OpenClaw的AI助手 - 能学习、能干活"""
    
    def __init__(self, workspace: str = "/Users/wangshice/.openclaw/workspace"):
        self.workspace = Path(workspace)
        self.knowledge_base = self.workspace / "knowledge"
        self.memory_dir = self.workspace / "memory"
        self.tasks_file = self.workspace / "tasks.json"
        self.config_file = self.workspace / "assistant_config.json"
        
        # 初始化目录
        self.knowledge_base.mkdir(exist_ok=True)
        
        # 加载配置
        self.config = self.load_config()
        
        # 知识库状态
        self.knowledge = {}
        self.task_queue = []
        
        logger.info("🤖 OpenClaw AI Assistant 已启动")
        
    def load_config(self) -> Dict:
        """加载配置"""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "learning_enabled": True,
            "auto_update_interval_hours": 6,
            "max_knowledge_entries": 10000,
            "personality": {
                "name": "OpenClaw Assistant",
                "role": "AI Helper",
                "vibe": "Helpful & Efficient"
            }
        }
    
    def save_config(self):
        """保存配置"""
        self.config["last_update"] = datetime.now().isoformat()
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
    
    def load_knowledge(self):
        """加载知识库"""
        logger.info("📚 加载知识库...")
        
        # 加载主知识库
        main_kb = self.knowledge_base / "main_knowledge.md"
        if main_kb.exists():
            with open(main_kb, 'r', encoding='utf-8') as f:
                content = f.read()
                # 解析知识条目
                entries = content.split('\n## ')
                self.knowledge['main'] = {
                    'entries': len(entries),
                    'last_loaded': datetime.now().isoformat(),
                    'content_hash': hash(content)
                }
        
        # 加载其他知识文件
        for kb_file in self.knowledge_base.glob("*.md"):
            if kb_file.name != "main_knowledge.md":
                with open(kb_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.knowledge[kb_file.stem] = {
                        'entries': content.count('\n## '),
                        'last_loaded': datetime.now().isoformat(),
                        'content_hash': hash(content)
                    }
        
        total_entries = sum(v['entries'] for v in self.knowledge.values())
        logger.info(f"✅ 知识库加载完成: {total_entries} 个知识点")
        return total_entries
    
    def learn_new_content(self, topic: str, content: str, source: str = "manual"):
        """学习新内容"""
        logger.info(f"📖 学习新内容: {topic}")
        
        # 生成知识条目
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{topic.replace(' ', '_')}_{timestamp}.md"
        
        knowledge_entry = f"""# {topic}

**学习时间**: {datetime.now().isoformat()}
**来源**: {source}
**标签**: {topic}

## 内容

{content}

---

*由 OpenClaw AI Assistant 自动学习*
"""
        
        # 保存到知识库
        output_file = self.knowledge_base / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(knowledge_entry)
        
        logger.info(f"✅ 已保存: {output_file.name}")
        
        # 更新统计
        self.config["last_update"] = datetime.now().isoformat()
        self.save_config()
        
        return output_file
    
    def add_task(self, task: str, priority: str = "normal", 
                 deadline: Optional[str] = None, dependencies: List[str] = None):
        """添加任务"""
        task_entry = {
            "id": len(self.task_queue) + 1,
            "task": task,
            "priority": priority,  # low, normal, high, urgent
            "status": "pending",  # pending, in_progress, completed, failed
            "created_at": datetime.now().isoformat(),
            "deadline": deadline,
            "dependencies": dependencies or [],
            "result": None,
            "error": None
        }
        
        self.task_queue.append(task_entry)
        self.save_tasks()
        logger.info(f"📝 添加任务: {task} (优先级: {priority})")
        return task_entry
    
    def get_next_task(self) -> Optional[Dict]:
        """获取下一个要执行的任务"""
        for task in self.task_queue:
            if task["status"] == "pending":
                # 检查依赖
                if task["dependencies"]:
                    dep_tasks = [t for t in self.task_queue 
                                if t["task"] in task["dependencies"] 
                                and t["status"] != "completed"]
                    if dep_tasks:
                        continue
                
                # 检查优先级
                task["status"] = "in_progress"
                task["started_at"] = datetime.now().isoformat()
                self.save_tasks()
                logger.info(f"🎯 开始执行任务: {task['task']}")
                return task
        
        return None
    
    def complete_task(self, task_id: int, result: str = None, error: str = None):
        """完成任务"""
        for task in self.task_queue:
            if task["id"] == task_id:
                task["status"] = "completed" if not error else "failed"
                task["completed_at"] = datetime.now().isoformat()
                task["result"] = result
                task["error"] = error
                self.save_tasks()
                
                status = "✅ 完成" if not error else "❌ 失败"
                logger.info(f"{status}: {task['task']}")
                
                # 检查是否有依赖此任务的其他任务
                for t in self.task_queue:
                    if task["task"] in t.get("dependencies", []) and t["status"] == "pending":
                        logger.info(f"🔗 触发依赖任务: {t['task']}")
                
                return task
        
        return None
    
    def save_tasks(self):
        """保存任务队列"""
        with open(self.tasks_file, 'w', encoding='utf-8') as f:
            json.dump(self.task_queue, f, ensure_ascii=False, indent=2)
    
    def load_tasks(self):
        """加载任务队列"""
        if self.tasks_file.exists():
            with open(self.tasks_file, 'r', encoding='utf-8') as f:
                self.task_queue = json.load(f)
            logger.info(f"📋 已加载 {len(self.task_queue)} 个任务")
    
    def run_automation(self, task_name: str) -> bool:
        """执行自动化任务"""
        automations = {
            "学习新知识": self._automation_learn,
            "更新知识库": self._automation_update_knowledge,
            "清理临时文件": self._automation_cleanup,
            "备份数据": self._automation_backup,
            "生成报告": self._automation_report,
            "优化性能": self._automation_optimize,
        }
        
        if task_name in automations:
            logger.info(f"🚀 执行自动化任务: {task_name}")
            try:
                automations[task_name]()
                return True
            except Exception as e:
                logger.error(f"❌ 自动化任务失败: {e}")
                return False
        else:
            logger.warning(f"⚠️ 未知自动化任务: {task_name}")
            return False
    
    def _automation_learn(self):
        """自动化学习任务"""
        # 示例：学习OpenClaw知识
        self.learn_new_content(
            topic="OpenClaw Assistant Capabilities",
            content="""
## 能力列表

1. **知识管理**
   - 自动学习新内容
   - 知识库管理与检索
   - 知识去重与更新

2. **任务管理**
   - 添加/管理任务队列
   - 优先级调度
   - 依赖关系处理

3. **自动化**
   - 定时学习新知识
   - 自动更新知识库
   - 数据备份与清理

4. **持续进化**
   - 记录学习历史
   - 追踪知识增长
   - 自我优化
            """,
            source="automation"
        )
    
    def _automation_update_knowledge(self):
        """自动化更新知识库"""
        # 合并碎片知识
        kb_files = list(self.knowledge_base.glob("*.md"))
        logger.info(f"📦 发现 {len(kb_files)} 个知识文件")
        
        # 生成汇总
        summary = f"""# 知识库汇总

**更新时间**: {datetime.now().isoformat()}
**文件数**: {len(kb_files)}

## 文件列表

"""
        for f in sorted(kb_files):
            summary += f"- {f.name}\n"
        
        (self.knowledge_base / "knowledge_summary.md").write_text(summary, encoding='utf-8')
        logger.info("✅ 知识库汇总已更新")
    
    def _automation_cleanup(self):
        """自动化清理"""
        temp_files = list(self.workspace.glob("*.tmp"))
        log_files = list(self.workspace.glob("*.log"))
        
        logger.info(f"🧹 清理临时文件: {len(temp_files)} 个")
        for f in temp_files:
            f.unlink()
        
        logger.info(f"📝 清理日志文件: {len(log_files)} 个")
    
    def _automation_backup(self):
        """自动化备份"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
        backup_dir = self.workspace / "backups" / backup_name
        
        # 创建备份
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 备份知识库
        for f in self.knowledge_base.glob("*.md"):
            (backup_dir / f.name).write_text(f.read_text(), encoding='utf-8')
        
        logger.info(f"💾 备份已保存: {backup_dir}")
    
    def _automation_report(self):
        """自动化生成报告"""
        report = f"""# OpenClaw Assistant 状态报告

**生成时间**: {datetime.now().isoformat()}

## 📊 统计信息

- **知识库文件数**: {len(list(self.knowledge_base.glob('*.md')))}
- **待完成任务数**: {sum(1 for t in self.task_queue if t['status'] == 'pending')}
- **进行中任务数**: {sum(1 for t in self.task_queue if t['status'] == 'in_progress')}
- **已完成任务数**: {sum(1 for t in self.task_queue if t['status'] == 'completed')}

## 📋 任务队列

"""
        
        for task in self.task_queue[:10]:  # 只显示前10个
            status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "failed": "❌"}
            emoji = status_emoji.get(task["status"], "📋")
            report += f"{emoji} {task['task']} ({task['priority']})\n"
        
        report_path = self.workspace / "assistant_report.md"
        report_path.write_text(report, encoding='utf-8')
        logger.info(f"📊 报告已生成: {report_path}")
    
    def _automation_optimize(self):
        """自动化优化"""
        logger.info("⚡ 优化性能...")
        
        # 清理内存占用
        self.load_knowledge()
        
        # 优化配置
        self.save_config()
        
        logger.info("✅ 性能优化完成")
    
    def get_status(self) -> Dict:
        """获取助手状态"""
        return {
            "name": self.config["personality"]["name"],
            "version": self.config["version"],
            "knowledge_entries": self.load_knowledge(),
            "pending_tasks": sum(1 for t in self.task_queue if t['status'] == 'pending'),
            "completed_tasks": sum(1 for t in self.task_queue if t['status'] == 'completed'),
            "last_update": self.config["last_update"],
            "learning_enabled": self.config["learning_enabled"]
        }
    
    def print_status(self):
        """打印状态"""
        status = self.get_status()
        print("\n" + "="*50)
        print(f"🤖 {status['name']} v{status['version']}")
        print("="*50)
        print(f"📚 知识条目: {status['knowledge_entries']}")
        print(f"📋 待完成任务: {status['pending_tasks']}")
        print(f"✅ 已完成任务: {status['completed_tasks']}")
        print(f"🕐 最后更新: {status['last_update']}")
        print(f"📖 学习已启用: {status['learning_enabled']}")
        print("="*50 + "\n")


def main():
    """主函数"""
    print("🚀 启动 OpenClaw AI Assistant...")
    
    # 创建助手实例
    assistant = OpenClawAssistant()
    
    # 加载现有数据
    assistant.load_tasks()
    assistant.load_knowledge()
    
    # 打印状态
    assistant.print_status()
    
    # 添加一些示例任务
    assistant.add_task("学习Python数据分析技巧", priority="high")
    assistant.add_task("更新知识库", priority="normal")
    assistant.add_task("生成状态报告", priority="low")
    
    # 执行自动化任务
    assistant.run_automation("学习新知识")
    assistant.run_automation("生成报告")
    
    # 打印状态
    assistant.print_status()
    
    return assistant


if __name__ == "__main__":
    main()
