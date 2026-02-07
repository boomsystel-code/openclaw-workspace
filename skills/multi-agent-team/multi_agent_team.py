#!/usr/bin/env python3
"""
多智能体协作系统演示
Multi-Agent Team Collaboration System
"""

import asyncio
import json
import random
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass


# ============ 配置 ============

class AgentRole(Enum):
    COMMANDER = "commander"
    RESEARCHER = "researcher"
    WRITER = "writer"
    REVIEWER = "reviewer"


@dataclass
class Agent:
    name: str
    role: AgentRole
    specialty: str
    system_prompt: str


# 团队成员配置
TEAM = {
    AgentRole.COMMANDER: Agent(
        name="指挥官",
        role=AgentRole.COMMANDER,
        specialty="任务分解与调度",
        system_prompt="你是一个项目指挥官，负责分解复杂任务并调度团队执行。"
    ),
    AgentRole.RESEARCHER: Agent(
        name="研究员",
        role=AgentRole.RESEARCHER,
        specialty="信息搜索与分析",
        system_prompt="你是一个专业研究员，负责搜索、收集和整理信息。"
    ),
    AgentRole.WRITER: Agent(
        name="写手",
        role=AgentRole.WRITER,
        specialty="内容创作",
        system_prompt="你是一个专业作家，负责撰写清晰、结构良好的内容。"
    ),
    AgentRole.REVIEWER: Agent(
        name="审核员",
        role=AgentRole.REVIEWER,
        specialty="质量把关",
        system_prompt="你是一个严格的质量审核员，负责检查内容的准确性和质量。"
    )
}


# ============ 模拟 LLM 调用 ============

async def mock_llm_call(system_prompt: str, user_input: str) -> str:
    """模拟 LLM 调用（实际使用时替换为真实 API）"""
    
    # 模拟不同 Agent 的响应
    if "指挥官" in system_prompt:
        return json.dumps({
            "tasks": [
                "收集 BTC 最新价格数据",
                "搜索分析师 BTC 走势预测",
                "整理宏观经济影响因素"
            ],
            "output_format": "Markdown 报告",
            "estimated_time": "2-3 分钟"
        }, ensure_ascii=False)
    
    elif "研究员" in system_prompt:
        return json.dumps({
            "topic": user_input[:50] + "...",
            "sources_found": 5,
            "key_findings": [
                "BTC 近期波动较大",
                "机构投资者持续入场",
                "宏观政策影响显著"
            ],
            "data_points": {
                "current_price": 70000,
                "market_cap": "1.3T",
                "fear_greed_index": 55
            }
        }, ensure_ascii=False)
    
    elif "写手" in system_prompt:
        return f"""# BTC 走势分析报告

## 摘要
{user_input[:100]}...

## 1. 市场概况
当前 BTC 价格约 $70,000，市值 1.3 万亿美元。

## 2. 影响因素
- 宏观经济走势
- 机构资金动向
- 监管政策变化

## 3. 机构观点
多数分析师认为 2026 年 BTC 有望突破新高。

## 4. 风险提示
- 波动性较大
- 政策不确定性
- 市场情绪影响

---
*本文由多智能体团队生成*
"""
    
    elif "审核员" in system_prompt:
        return json.dumps({
            "score": 85,
            "issues": [
                "建议添加更多数据来源引用",
                "部分观点可以更谨慎"
            ],
            "suggestions": [
                "补充具体的价格预测数据",
                "增加风险提示的详细说明"
            ],
            "overall": "内容质量良好，结构清晰"
        }, ensure_ascii=False)
    
    return user_input


# ============ Agent 执行器 ============

class AgentExecutor:
    """Agent 执行器"""
    
    def __init__(self):
        self.execution_log = []
    
    async def run_agent(self, role: AgentRole, task: str) -> Dict:
        """运行指定 Agent 执行任务"""
        
        agent = TEAM[role]
        print(f"\n🤖 {agent.name} 正在工作...")
        print(f"   职责：{agent.specialty}")
        print(f"   任务：{task[:50]}...")
        
        # 模拟 LLM 调用
        response = await mock_llm_call(
            system_prompt=agent.system_prompt,
            user_input=task
        )
        
        # 解析响应
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {"content": response}
        
        # 记录执行日志
        self.execution_log.append({
            "agent": agent.name,
            "task": task[:50],
            "timestamp": datetime.now().isoformat(),
            "result": result
        })
        
        print(f"   ✅ {agent.name} 完成")
        
        return result
    
    async def run_parallel(self, tasks: Dict[AgentRole, str]) -> Dict:
        """并行运行多个 Agent"""
        print(f"\n🚀 并行执行 {len(tasks)} 个任务...")
        
        results = await asyncio.gather(
            *[self.run_agent(role, task) for role, task in tasks.items()]
        )
        
        return dict(zip(tasks.keys(), results))


# ============ 多智能体团队 ============

class MultiAgentTeam:
    """多智能体协作团队"""
    
    def __init__(self):
        self.executor = AgentExecutor()
    
    async def handle_request(self, request: str) -> Dict:
        """处理用户请求"""
        
        print(f"\n" + "="*50)
        print(f"📋 收到请求：{request}")
        print(f"="*50)
        
        # Step 1: Commander 分析并规划
        print(f"\n👨‍💼 Commander 分析需求...")
        plan = await self.executor.run_agent(
            role=AgentRole.COMMANDER,
            task=f"分析以下需求并制定执行计划：{request}"
        )
        
        tasks = plan.get("tasks", [])
        
        if not tasks:
            # 简单任务直接执行
            print(f"\n✨ 简单任务，直接生成结果...")
            return await self.executor.run_agent(
                role=AgentRole.WRITER,
                task=request
            )
        
        # Step 2: Researcher 并行收集信息
        print(f"\n🔍 研究员并行收集信息...")
        research_tasks = {
            AgentRole.RESEARCHER: f"调研主题：{request}",
        }
        research_results = await self.executor.run_parallel(research_tasks)
        
        # 转换 AgentRole 为字符串以便 JSON 序列化
        research_results_serializable = {
            str(k.value): v for k, v in research_results.items()
        }
        
        # Step 3: Writer 生成内容
        print(f"\n✍️ 写手生成内容...")
        writer_input = f"""
主题：{request}

调研结果：
{json.dumps(research_results_serializable, ensure_ascii=False, indent=2)}
"""
        draft = await self.executor.run_agent(
            role=AgentRole.WRITER,
            task=writer_input
        )
        
        # Step 4: Reviewer 审核
        print(f"\n🔍 审核员审核内容...")
        review = await self.executor.run_agent(
            role=AgentRole.REVIEWER,
            task=draft.get("content", str(draft))
        )
        
        # Step 5: Commander 汇总最终输出
        print(f"\n👨‍💼 Commander 汇总最终报告...")
        
        final_output = {
            "request": request,
            "plan": plan,
            "research": research_results,
            "draft": draft,
            "review": review,
            "execution_log": self.executor.execution_log,
            "generated_at": datetime.now().isoformat()
        }
        
        return final_output
    
    def print_result(self, result: Dict):
        """打印最终结果"""
        print(f"\n" + "="*50)
        print(f"✅ 最终报告已生成")
        print(f"="*50)
        
        # 打印执行摘要
        print(f"\n📊 执行摘要：")
        print(f"   执行步骤：{len(result['execution_log'])}")
        print(f"   参与 Agent：{set(log['agent'] for log in result['execution_log'])}")
        print(f"   生成时间：{result['generated_at']}")
        
        # 打印审核结果
        review = result.get("review", {})
        if isinstance(review, dict):
            print(f"\n🔍 审核评分：{review.get('score', 'N/A')}/100")
            if review.get("issues"):
                print(f"   待改进：{len(review['issues'])} 项")
        
        # 尝试打印草稿内容
        draft = result.get("draft", {})
        if isinstance(draft, dict) and "content" in draft:
            print(f"\n📝 报告内容：")
            print("-" * 40)
            print(draft["content"][:500] + "..." if len(draft.get("content", "")) > 500 else draft["content"])
        
        print(f"\n" + "="*50)
        print(f"✨ 由 OpenClaw 多智能体团队生成")
        print(f"="*50)


# ============ CLI 界面 ============

async def main():
    """CLI 入口"""
    import sys
    
    team = MultiAgentTeam()
    
    # 获取用户输入
    if len(sys.argv) > 1:
        # 从命令行参数获取
        request = " ".join(sys.argv[1:])
    else:
        # 交互式输入
        print("""
🤖 多智能体协作团队演示

输入你的需求，我来调度团队完成！
示例：
  python multi_agent_team.py "分析 BTC 2026 年走势"
  python multi_agent_team.py "写一篇关于 AI Agent 的科普文章"
  python multi_agent_team.py "调研一下最新科技趋势"

请输入你的需求：
""")
        request = input("> ").strip()
    
    if not request:
        print("❌ 请输入有效需求")
        return
    
    # 执行
    result = await team.handle_request(request)
    
    # 输出结果
    team.print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
