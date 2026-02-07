#!/usr/bin/env python3
"""
多智能体协作系统 - 完整版（6个专业 Agent）
Multi-Agent Team - Full Version with 6 Specialized Agents
"""

import asyncio
import json
import os
import random
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass


# ============ Agent 角色定义 ============

class AgentRole(Enum):
    # 核心团队
    COMMANDER = "commander"
    RESEARCHER = "researcher"
    WRITER = "writer"
    REVIEWER = "reviewer"
    # 扩展专家
    DATA_ANALYST = "data_analyst"
    LEGAL_ADVISOR = "legal_advisor"
    FINANCIAL_ADVISOR = "financial_advisor"
    TRANSLATOR = "translator"
    CREATIVE_DESIGNER = "creative_designer"


@dataclass
class Agent:
    name: str
    role: AgentRole
    specialty: str
    system_prompt: str
    color: str = "🔹"


# ============ 完整 Agent 团队 ============

TEAM = {
    # === 核心团队 ===
    AgentRole.COMMANDER: Agent(
        name="指挥官",
        role=AgentRole.COMMANDER,
        specialty="任务分解与调度",
        color="👑",
        system_prompt="""你是一个项目指挥官，负责领导多智能体团队完成复杂任务。

职责：
1. 理解用户需求
2. 将复杂任务分解为子任务
3. 根据任务类型选择合适的 Agent
4. 协调团队协作，汇总结果
5. 输出最终答案

请用 JSON 格式回复任务规划。
"""
    ),
    AgentRole.RESEARCHER: Agent(
        name="研究员",
        role=AgentRole.RESEARCHER,
        specialty="信息搜索与分析",
        color="🔍",
        system_prompt="""你是一个专业研究员，负责收集和整理信息。

职责：
1. 搜索和收集相关信息
2. 整理和归类数据
3. 提取关键信息
4. 提供结构化的调研报告

请用 JSON 格式回复。
"""
    ),
    AgentRole.WRITER: Agent(
        name="写手",
        role=AgentRole.WRITER,
        specialty="内容创作",
        color="✍️",
        system_prompt="""你是一个专业作家，负责撰写清晰、结构良好的内容。

职责：
1. 根据调研结果撰写内容
2. 语言流畅、结构清晰
3. 适当添加案例和说明
4. 生成可读性强的文档

请直接返回 Markdown 格式的内容。
"""
    ),
    AgentRole.REVIEWER: Agent(
        name="审核员",
        role=AgentRole.REVIEWER,
        specialty="质量把关",
        color="✅",
        system_prompt="""你是一个严格的质量审核员，负责检查内容的准确性和质量。

职责：
1. 核查内容的准确性
2. 检查逻辑的一致性
3. 发现并指出问题
4. 提出改进建议

请用 JSON 格式回复审核结果。
"""
    ),
    
    # === 扩展专家 ===
    AgentRole.DATA_ANALYST: Agent(
        name="数据分析师",
        role=AgentRole.DATA_ANALYST,
        specialty="数据处理与可视化",
        color="📊",
        system_prompt="""你是一个专业数据分析师，负责数据处理、统计分析和可视化建议。

职责：
1. 分析数据趋势
2. 计算统计数据
3. 发现数据中的模式
4. 提供可视化建议
5. 给出数据驱动的建议

请用 JSON 格式回复分析结果。
"""
    ),
    AgentRole.LEGAL_ADVISOR: Agent(
        name="法律顾问",
        role=AgentRole.LEGAL_ADVISOR,
        specialty="法律合规审查",
        color="⚖️",
        system_prompt="""你是一个专业法律顾问，负责法律合规审查。

职责：
1. 识别潜在法律风险
2. 检查合规性问题
3. 提供法律建议
4. 评估合同/协议条款

请用 JSON 格式回复法律分析。
"""
    ),
    AgentRole.FINANCIAL_ADVISOR: Agent(
        name="财务顾问",
        role=AgentRole.FINANCIAL_ADVISOR,
        specialty="投资理财建议",
        color="💰",
        system_prompt="""你是一个专业财务顾问，负责投资理财分析和建议。

职责：
1. 分析投资机会
2. 评估风险收益
3. 提供资产配置建议
4. 分析财务数据
5. 给出投资建议

请用 JSON 格式回复财务分析。
"""
    ),
    AgentRole.TRANSLATOR: Agent(
        name="翻译官",
        role=AgentRole.TRANSLATOR,
        specialty="多语言翻译",
        color="🌐",
        system_prompt="""你是一个专业翻译官，负责多语言翻译和本地化。

职责：
1. 高质量翻译
2. 保持原文风格
3. 考虑文化差异
4. 本地化建议

请直接返回翻译结果。
"""
    ),
    AgentRole.CREATIVE_DESIGNER: Agent(
        name="创意设计师",
        role=AgentRole.CREATIVE_DESIGNER,
        specialty="创意策划与设计",
        color="🎨",
        system_prompt="""你是一个创意设计师，负责创意策划和设计方案。

职责：
1. 提供创意想法
2. 设计视觉方案
3. 策划营销活动
4. 创新解决方案

请用 JSON 格式回复创意方案。
"""
    ),
}


# ============ LLM 客户端 ============

class LLMClient:
    """LLM 客户端"""
    
    def __init__(self):
        self.api_key = os.environ.get("MINIMAX_API_KEY", "")
        self.base_url = "https://api.minimaxi.com/v1"
        self.model = "MiniMax-M2.1"
    
    async def call(self, system_prompt: str, user_input: str, 
                   temperature: float = 0.7, max_tokens: int = 3000) -> str:
        """调用 LLM API"""
        
        if not self.api_key:
            return await self._mock_call(system_prompt, user_input)
        
        try:
            import requests
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
                timeout=60
            )
            
            data = response.json()
            return data.get("content", data.get("message", {}).get("content", str(data)))
            
        except Exception as e:
            print(f"⚠️ API 调用失败: {e}")
            return await self._mock_call(system_prompt, user_input)
    
    async def _mock_call(self, system_prompt: str, user_input: str) -> str:
        """模拟 LLM 响应"""
        
        role_map = {
            "指挥官": {"tasks": ["收集信息", "分析数据", "撰写报告"], "output_format": "Markdown"},
            "研究员": {"sources_found": 5, "key_findings": ["趋势1", "趋势2", "趋势3"]},
            "审核员": {"score": 88, "issues": [], "overall": "内容优秀"},
            "数据分析师": {"trend": "上涨", "avg_change": "+5.2%", "confidence": 85},
            "法律顾问": {"risk_level": "低", "compliance": "正常", "recommendations": []},
            "财务顾问": {"recommendation": "建议关注", "risk": "中等", "expected_return": "8-12%"},
            "翻译官": "这里是对应的中文翻译。",
            "创意设计师": {"concepts": ["概念A", "概念B", "概念C"], "recommended": "概念A"}
        }
        
        for role_name, mock_data in role_map.items():
            if role_name in system_prompt:
                if isinstance(mock_data, str):
                    return mock_data
                return json.dumps(mock_data, ensure_ascii=False)
        
        return user_input


# ============ Agent 执行器 ============

class AgentExecutor:
    """Agent 执行器"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.execution_log = []
    
    async def run_agent(self, role: AgentRole, task: str) -> Dict:
        """运行指定 Agent"""
        
        agent = TEAM[role]
        print(f"\n{agent.color} {agent.name} 正在工作...")
        print(f"   专长：{agent.specialty}")
        print(f"   任务：{task[:50]}...")
        
        response = await self.llm.call(
            system_prompt=agent.system_prompt,
            user_input=task
        )
        
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {"content": response}
        
        self.execution_log.append({
            "agent": agent.name,
            "role": role.value,
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


# ============ 任务调度器 ============

class TaskScheduler:
    """智能任务调度器"""
    
    @staticmethod
    def suggest_agents(request: str) -> List[AgentRole]:
        """根据请求建议需要的 Agent"""
        
        request_lower = request.lower()
        agents = [AgentRole.COMMANDER]
        
        if any(kw in request_lower for kw in ["分析", "趋势", "数据", "统计", "报告"]):
            agents.extend([AgentRole.RESEARCHER, AgentRole.DATA_ANALYST])
        
        if any(kw in request_lower for kw in ["翻译", "英文", "多语言"]):
            agents.append(AgentRole.TRANSLATOR)
        
        if any(kw in request_lower for kw in ["法律", "合规", "合同", "风险"]):
            agents.append(AgentRole.LEGAL_ADVISOR)
        
        if any(kw in request_lower for kw in ["投资", "理财", "财务", "收益", "BTC", "股票"]):
            agents.extend([AgentRole.FINANCIAL_ADVISOR, AgentRole.DATA_ANALYST])
        
        if any(kw in request_lower for kw in ["创意", "设计", "营销", "方案"]):
            agents.append(AgentRole.CREATIVE_DESIGNER)
        
        if AgentRole.WRITER not in agents:
            agents.append(AgentRole.WRITER)
        if AgentRole.REVIEWER not in agents:
            agents.append(AgentRole.REVIEWER)
        
        return list(set(agents))
    
    @staticmethod
    def create_tasks(request: str, agents: List[AgentRole]) -> Dict[AgentRole, str]:
        """为每个 Agent 创建任务"""
        
        tasks = {}
        
        for agent in agents:
            task_map = {
                AgentRole.COMMANDER: f"分析需求并制定执行计划：{request}",
                AgentRole.RESEARCHER: f"收集相关信息：{request}",
                AgentRole.DATA_ANALYST: f"分析相关数据：{request}",
                AgentRole.FINANCIAL_ADVISOR: f"提供财务/投资分析：{request}",
                AgentRole.LEGAL_ADVISOR: f"进行法律合规审查：{request}",
                AgentRole.TRANSLATOR: f"翻译以下内容：{request}",
                AgentRole.CREATIVE_DESIGNER: f"提供创意设计方案：{request}",
                AgentRole.WRITER: f"撰写关于 {request} 的内容",
                AgentRole.REVIEWER: "审核已完成的内容，检查质量和准确性",
            }
            tasks[agent] = task_map.get(agent, f"处理：{request}")
        
        return tasks


# ============ 多智能体团队 ============

class MultiAgentTeam:
    """多智能体协作团队"""
    
    def __init__(self):
        self.llm = LLMClient()
        self.executor = AgentExecutor(self.llm)
        self.scheduler = TaskScheduler()
    
    async def handle_request(self, request: str) -> Dict:
        """处理用户请求"""
        
        print(f"\n" + "="*60)
        print(f"📋 收到请求：{request}")
        print(f"="*60)
        
        # 1. 分析需求，建议 Agent
        suggested_agents = self.scheduler.suggest_agents(request)
        
        print(f"\n👥 建议团队阵容：")
        for agent in suggested_agents:
            info = TEAM[agent]
            print(f"   {info.color} {info.name} - {info.specialty}")
        
        # 2. 创建任务
        tasks = self.scheduler.create_tasks(request, suggested_agents)
        
        # 3. 排除 Commander
        worker_tasks = {k: v for k, v in tasks.items() if k != AgentRole.COMMANDER}
        
        # 4. 并行执行
        print(f"\n⚡ 开始执行...")
        results = await self.executor.run_parallel(worker_tasks)
        
        # 5. Writer 整合
        print(f"\n✍️ 整合内容...")
        
        research_data = results.get(AgentRole.RESEARCHER, {})
        analyst_data = results.get(AgentRole.DATA_ANALYST, {})
        
        writer_input = f"""
主题：{request}
研究结果：{json.dumps(research_data, ensure_ascii=False)}
数据分析：{json.dumps(analyst_data, ensure_ascii=False)}

请基于以上专家的意见，撰写一份综合报告。
"""
        
        final_report = await self.executor.run_agent(AgentRole.WRITER, writer_input)
        
        # 6. Reviewer 审核
        print(f"\n✅ 质量审核...")
        report_content = final_report.get("content", str(final_report))
        review = await self.executor.run_agent(AgentRole.REVIEWER, report_content)
        
        return {
            "request": request,
            "team": [TEAM[a].name for a in suggested_agents],
            "results": results,
            "final_report": final_report,
            "review": review,
            "execution_log": self.executor.execution_log,
            "generated_at": datetime.now().isoformat()
        }
    
    def print_result(self, result: Dict):
        """打印结果"""
        print(f"\n" + "="*60)
        print(f"✅ 任务完成！")
        print(f"="*60)
        
        print(f"\n📊 执行摘要：")
        print(f"   参与专家：{', '.join(result['team'])}")
        print(f"   执行步骤：{len(result['execution_log'])}")
        print(f"   完成时间：{result['generated_at']}")
        
        review = result.get("review", {})
        if isinstance(review, dict):
            print(f"\n🔍 质量评分：{review.get('score', 'N/A')}/100")
        
        print(f"\n📝 最终报告：")
        print("-" * 60)
        content = result.get("final_report", {}).get("content", "")
        print(content[:2000])
        if len(content) > 2000:
            print(f"\n... (共 {len(content)} 字)")
        print("-" * 60)
        
        print(f"\n✨ 由 OpenClaw 多智能体团队生成")


# ============ CLI ============

async def main():
    import sys
    
    team = MultiAgentTeam()
    
    if len(sys.argv) > 1:
        request = " ".join(sys.argv[1:])
    else:
        print("""
🤖 OpenClaw 多智能体协作团队 (9位专家版)

可用专家：
  👑 指挥官     - 任务分解与调度
  🔍 研究员     - 信息搜索与分析
  ✍️ 写手       - 内容创作
  ✅ 审核员     - 质量把关
  📊 数据分析师 - 数据处理与可视化
  ⚖️ 法律顾问   - 法律合规审查
  💰 财务顾问   - 投资理财建议
  🌐 翻译官     - 多语言翻译
  🎨 创意设计师 - 创意策划与设计

用法：
  python multi_agent_full.py "分析 BTC 投资趋势"
  python multi_agent_full.py "翻译这段英文并分析法律风险"
  python multi_agent_full.py "设计一个营销创意方案"

请输入你的需求：
""")
        request = input("> ").strip()
    
    if not request:
        print("❌ 请输入有效需求")
        return
    
    result = await team.handle_request(request)
    team.print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
