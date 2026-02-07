#!/usr/bin/env python3
"""
错误日志系统 - 自动记录和追踪所有错误

使用方法:
  python error_logger.py --log "错误描述" --category "technical"
  python error_logger.py --resolve ERROR-001 --solution "解决方案"
  python error_logger.py --report --weekly
"""

import json
import os
import sys
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
import argparse

# 配置
WORKSPACE = os.path.expanduser("~/.openclaw/workspace")
ERROR_LOGS_DIR = Path(os.path.join(WORKSPACE, "error-logs"))
ERROR_COUNTER_FILE = os.path.join(WORKSPACE, "error-logs", ".counter")

class ErrorLogger:
    def __init__(self):
        self.errors_dir = Path(os.path.join(ERROR_LOGS_DIR, "errors", datetime.now().strftime("%Y-%m-%d")))
        self.errors_dir.mkdir(parents=True, exist_ok=True)
    
    def get_next_id(self) -> str:
        """获取下一个错误ID"""
        counter_file = Path(ERROR_COUNTER_FILE)
        if counter_file.exists():
            with open(counter_file, 'r') as f:
                counter = int(f.read().strip()) + 1
        else:
            counter = 1
        
        with open(counter_file, 'w') as f:
            f.write(str(counter))
        
        return f"ERROR-{counter:03d}"
    
    def log_error(self, error_msg: str, category: str = "other", severity: str = "medium",
                  context: str = "", solution: str = "", status: str = "open") -> str:
        """记录一个新错误"""
        error_id = self.get_next_id()
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 创建错误详情
        error_file = os.path.join(self.errors_dir, f"{error_id.lower()}.md")
        
        content = f"""---
title: "{error_id}"
date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
status: {status}
severity: {severity}
category: {category}
---

## 错误摘要

**错误代码:** {error_id}
**发生时间:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**严重程度:** {severity}
**状态:** {status}

## 错误信息

```
{error_msg}
```

## 发生场景

### 上下文描述
{context if context else '未提供上下文信息'}

## 解决方案

{solution if solution else '暂无解决方案'}

---

## 检查清单

- [x] 错误信息已记录
- [ ] 解决方案已找到
- [ ] 经验教训已提炼
- [ ] 已更新patterns.md
"""
        
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 更新每日汇总
        self._update_daily_summary(error_id, error_msg, category, severity)
        
        print(f"✅ 错误已记录: {error_id}")
        print(f"📄 文件: {error_file}")
        
        return error_id
    
    def _update_daily_summary(self, error_id: str, error_msg: str, category: str, severity: str):
        """更新每日错误汇总"""
        summary_file = os.path.join(self.errors_dir, "summary.md")
        
        entry = f"""### {error_id} [{severity}] [{category}]
- **时间:** {datetime.now().strftime("%H:%M:%S")}
- **错误:** {error_msg[:100]}...
"""
        
        if os.path.exists(summary_file):
            # 找到 ## 错误列表 部分并插入
            with open(summary_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if "## 错误列表" in content:
                # 在错误列表后插入
                content = content.replace("## 错误列表\n", f"## 错误列表\n{entry}\n", 1)
            else:
                content += f"\n{entry}"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            summary = f"""# {datetime.now().strftime("%Y-%m-%d")} 错误汇总

## 统计

- 总错误数: 1
- 严重错误: {1 if severity == 'critical' else 0}
- 已解决: 0

## 错误列表

{entry}
"""
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
    
    def resolve_error(self, error_id: str, solution: str):
        """标记错误为已解决"""
        error_file = os.path.join(self.errors_dir, f"{error_id.lower()}.md")
        
        if not os.path.exists(error_file):
            print(f"❌ 错误文件不存在: {error_file}")
            return
        
        with open(error_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 更新状态和解决方案
        content = content.replace('status: open', 'status: resolved')
        content = content.replace('## 解决方案\n\n暂无解决方案', f'## 解决方案\n\n{solution}')
        
        # 添加解决时间
        content += f"\n\n**解决时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 错误已解决: {error_id}")
        print(f"📝 解决方案: {solution}")
    
    def generate_report(self, report_type: str = "weekly"):
        """生成错误报告"""
        if report_type == "weekly":
            self._generate_weekly_report()
        elif report_type == "monthly":
            self._generate_monthly_report()
        else:
            self._generate_daily_report()
    
    def _generate_weekly_report(self):
        """生成每周报告"""
        # 获取上周的数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # 统计错误
        error_count = 0
        by_category = {}
        by_severity = {}
        resolved_count = 0
        
        report_file = os.path.join(ERROR_LOGS_DIR, "analysis", f"weekly-summary-{end_date.strftime('%Y-%m-%d')}.md")
        
        report = f"""# 每周错误分析报告

**周期:** {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}
**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 统计概览

- 总错误数: {error_count}
- 已解决: {resolved_count}
- 解决率: {f"{resolved_count/max(error_count,1)*100:.1f}%" if error_count > 0 else "N/A"}

## 按类别分布

| 类别 | 数量 | 占比 |
|------|------|------|
"""
        
        # TODO: 实现完整的统计逻辑
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📊 每周报告已生成: {report_file}")
    
    def _generate_daily_report(self):
        """生成每日报告"""
        self._update_daily_summary("", "", "", "")
    
    def _generate_monthly_report(self):
        """生成每月报告"""
        print("📊 月度报告功能待实现")
    
    def list_recent_errors(self, limit: int = 10) -> List[Dict]:
        """列出最近的错误"""
        errors = []
        for i, error_file in enumerate(sorted(Path(self.errors_dir).glob("ERROR-*.md"), reverse=True)):
            if i >= limit:
                break
            with open(error_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取基本信息
            error_id = error_file.stem
            status = "resolved" if "status: resolved" in content else "open"
            
            errors.append({
                "id": error_id,
                "status": status,
                "file": str(error_file)
            })
        
        return errors
    
    def show_statistics(self):
        """显示错误统计"""
        stats_file = os.path.join(ERROR_LOGS_DIR, "statistics", "metrics.json")
        
        # 简单的统计
        total_errors = 0
        open_errors = 0
        resolved_errors = 0
        
        for errors_date_dir in Path(os.path.join(ERROR_LOGS_DIR, "errors")).glob("*"):
            if errors_date_dir.is_dir():
                for error_file in errors_date_dir.glob("ERROR-*.md"):
                    total_errors += 1
                    with open(error_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if "status: resolved" in content:
                        resolved_errors += 1
                    else:
                        open_errors += 1
        
        print("\n📊 错误统计")
        print("=" * 40)
        print(f"总错误数: {total_errors}")
        print(f"未解决: {open_errors}")
        print(f"已解决: {resolved_errors}")
        print(f"解决率: {f"{resolved_errors/max(total_errors,1)*100:.1f}%" if total_errors > 0 else 'N/A'}")
        print()


def main():
    parser = argparse.ArgumentParser(description="错误日志系统 - 自动记录和追踪所有错误")
    parser.add_argument("--log", "-l", help="记录新错误")
    parser.add_argument("--category", "-c", default="other", choices=["technical", "user-interface", "automation", "integration", "other"],
                       help="错误类别")
    parser.add_argument("--severity", "-s", default="medium", choices=["low", "medium", "high", "critical"],
                       help="严重程度")
    parser.add_argument("--context", "-ctx", default="", help="错误上下文")
    parser.add_argument("--resolve", "-r", help="解决错误 (提供错误ID)")
    parser.add_argument("--solution", "-S", default="", help="解决方案")
    parser.add_argument("--list", "-L", action="store_true", help="列出最近的错误")
    parser.add_argument("--report", choices=["daily", "weekly", "monthly"], help="生成报告")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")
    
    args = parser.parse_args()
    
    logger = ErrorLogger()
    
    if args.stats:
        logger.show_statistics()
    elif args.log:
        error_id = logger.log_error(
            error_msg=args.log,
            category=args.category,
            severity=args.severity,
            context=args.context
        )
        print(f"\n🎯 下一步:")
        print(f"  1. 分析错误原因")
        print(f"  2. 找到解决方案")
        print(f"  3. 运行: python error_logger.py --resolve {error_id} --solution '你的解决方案'")
    elif args.resolve:
        if not args.solution:
            print("❌ 请提供解决方案: --solution 'xxx'")
            sys.exit(1)
        logger.resolve_error(args.resolve, args.solution)
    elif args.list:
        errors = logger.list_recent_errors()
        print("\n📋 最近的错误:")
        for error in errors:
            status = "✅" if error["status"] == "resolved" else "🔴"
            print(f"  {status} {error['id']} - {error['file']}")
    elif args.report:
        logger.generate_report(args.report)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
