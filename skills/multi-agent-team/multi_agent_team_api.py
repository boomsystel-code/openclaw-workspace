#!/usr/bin/env python3
"""
多智能体协作系统 - 真实 API 版本
Multi-Agent Team with Real API Support
"""

import asyncio
import json
import os
import random
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass

import requests


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
        system_prompt="""你是一个项目指挥官。
职责：
1. 理解用户需求
2. 将复杂任务分解为子任务
3. 分配给合适的 Agent 执行
4. 汇总结果并输出最终答案

请用 JSON 格式回复，包含：tasks (列表), output_format, estimated_time"""
    ),
    AgentRole.RESEARCHER: Agent(
        name="研究员",
        role=AgentRole.RESEARCHER,
        specialty="信息搜索与分析",
        system_prompt="""你是一个专业研究员。
职责：
1. 搜索和收集相关信息
2. 整理和归类数据
3. 提取关键信息
4. 提供结构化的调研报告

请用 JSON 格式回复，包含：topic, sources_found, key_findings (列表), data_points"""
    ),
    AgentRole.WRITER: Agent(
        name="写手",
        role=AgentRole.WRITER,
        specialty="内容创作",
        system_prompt="""你是一个专业作家。
职责：
1. 根据调研结果撰写内容
2. 语言流畅、结构清晰
3. 适当添加案例和说明
4. 生成可读性强的文档

请直接返回文章内容，使用 Markdown 格式。"""
    ),
    AgentRole.REVIEWER: Agent(
        name="审核员",
        role=AgentRole.REVIEWER,
        specialty="质量把关",
        system_prompt="""你是一个严格的质量审核员。
职责：
1. 核查内容的准确性
2. 检查逻辑的一致性
3. 发现并指出问题
4. 提出改进建议

请用 JSON 格式回复，包含：score (0-100), issues (列表), suggestions (列表), overall"""
    )
}


# ============ LLM API 调用 ============

class LLMClient:
    """LLM 客户端"""
    
    def __init__(self):
        self.api_key = os.environ.get("MINIMAX_API_KEY", "")
        self.base_url = "https://api.minimaxi.com/v1"
        self.model = "MiniMax-M2.1"
    
    async def call(self, system_prompt: str, user_input: str, 
                   temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """调用 LLM API"""
        
        # 如果没有 API Key，使用模拟模式
        if not self.api_key:
            return await self._mock_call(system_prompt, user_input)
        
        # 真实 API 调用
        try:
            response = requests.post(
                f"{self.base_url}/messages",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=30
            )
            
            data = response.json()
            return data.get("content", data.get("message", {}).get("content", str(data)))
            
        except Exception as e:
            print(f"⚠️ API 调用失败: {e}")
            return await self._mock_call(system_prompt, user_input)
    
    async def _mock_call(self, system_prompt: str, user_input: str) -> str:
        """模拟 LLM 响应（当 API 不可用时）"""
        
        if "指挥官" in system_prompt:
            return json.dumps({
                "tasks": ["收集 BTC 最新价格数据", "搜索分析师预测", "整理宏观因素"],
                "output_format": "Markdown 报告",
                "estimated_time": "2-3 分钟"
            }, ensure_ascii=False)
        
        elif "研究员" in system_prompt:
            return json.dumps({
                "topic": user_input[:50] + "...",
                "sources_found": 5,
                "key_findings": ["BTC 近期波动较大", "机构投资者持续入场", "宏观政策影响显著"],
                "data_points": {"current_price": 70000, "market_cap": "1.3T", "fear_greed_index": 55}
            }, ensure_ascii=False)
        
        elif "写手" in system_prompt:
            return f"""# BTC 走势分析报告

## 摘要
基于最新调研数据，本文对 BTC 2026 年走势进行分析。

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
*本文由 OpenClaw 多智能体团队生成*
"""
        
        elif "审核员" in system_prompt:
            return json.dumps({
                "score": 85,
                "issues": ["建议添加更多数据来源引用", "部分观点可以更谨慎"],
                "suggestions": ["补充具体的价格预测数据", "增加风险提示的详细说明"],
                "overall": "内容质量良好，结构清晰"
            }, ensure_ascii=False)
        
        return user_input


# ============ Agent 执行器 ============

class AgentExecutor:
    """Agent 执行器"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.execution_log = []
    
    async def run_agent(self, role: AgentRole, task: str) -> Dict:
        """运行指定 Agent 执行任务"""
        
        agent = TEAM[role]
        print(f"\n🤖 {agent.name} 正在工作...")
        print(f"   职责：{agent.specialty}")
        print(f"   任务：{task[:60]}...")
        
        # 调用 LLM
        response = await self.llm.call(
            system_prompt=agent.system_prompt,
            user_input=task
        )
        
        # 解析响应
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {"content": response, "raw": response}
        
        # 记录日志
        self.execution_log.append({
            "agent": agent.name,
            "task": task[:60],
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
        self.llm = LLMClient()
        self.executor = AgentExecutor(self.llm)
    
    async def handle_request(self, request: str) -> Dict:
        """处理用户请求"""
        
        print(f"\n" + "="*50)
        print(f"📋 收到请求：{request}")
        print(f"="*50)
        
        # 检查 API 状态
        if not self.llm.api_key:
            print(f"\n⚠️ 未检测到 MINIMAX_API_KEY 环境变量")
            print(f"   使用模拟模式运行")
        
        # Step 1: Commander 分析并规划
        print(f"\n👨‍💼 Commander 分析需求...")
        plan = await self.executor.run_agent(
            role=AgentRole.COMMANDER,
            task=f"分析以下需求并制定执行计划：{request}"
        )
        
        # Step 2: Researcher 并行收集信息
        print(f"\n🔍 研究员并行收集信息...")
        research_tasks = {
            AgentRole.RESEARCHER: f"请详细调研：{request}"
        }
        research_results = await self.executor.run_parallel(research_tasks)
        
        # 转换结果格式
        research_results_serializable = {
            str(k.value): v for k, v in research_results.items()
        }
        
        # Step 3: Writer 生成内容
        print(f"\n✍️ 写手生成内容...")
        writer_input = f"""
主题：{request}

调研结果：
{json.dumps(research_results_serializable, ensure_ascii=False, indent=2)}

请基于以上调研结果，撰写一篇完整的分析报告。
"""
        draft = await self.executor.run_agent(
            role=AgentRole.WRITER,
            task=writer_input
        )
        
        # Step 4: Reviewer 审核
        print(f"\n🔍 审核员审核内容...")
        draft_content = draft.get("content", str(draft))
        review = await self.executor.run_agent(
            role=AgentRole.REVIEWER,
            task=draft_content
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
        
        # 执行摘要
        print(f"\n📊 执行摘要：")
        print(f"   执行步骤：{len(result['execution_log'])}")
        print(f"   参与 Agent：{set(log['agent'] for log in result['execution_log'])}")
        print(f"   生成时间：{result['generated_at']}")
        
        # 审核结果
        review = result.get("review", {})
        if isinstance(review, dict):
            print(f"\n🔍 审核评分：{review.get('score', 'N/A')}/100")
            if review.get("issues"):
                print(f"   待改进：{len(review['issues'])} 项")
        
        # 报告内容
        draft = result.get("draft", {})
        content = draft.get("content", str(draft))
        
        print(f"\n📝 报告内容：")
        print("-" * 40)
        print(content[:1500])
        if len(content) > 1500:
            print(f"\n... (共 {len(content)} 字)")
        print("-" * 40)
        
        print(f"\n✨ 由 OpenClaw 多智能体团队生成")


# ============ CLI ============

async def main():
    import sys
    
    team = MultiAgentTeam()
    
    # 获取输入
    if len(sys.argv) > 1:
        request = " ".join(sys.argv[1:])
    else:
        print("""
🤖 多智能体协作团队

用法：
  python multi_agent_team_api.py "分析 BTC 2026 年走势"
  python multi_agent_team_api.py "写一篇关于 AI 的科普文章"

请输入你的需求：
""")
        request = input("> ").strip()
    
    if not request:
        print("❌ 请输入有效需求")
        return
    
    # 执行
    result = await team.handle_request(request)
    team.print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
