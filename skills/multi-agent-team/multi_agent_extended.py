#!/usr/bin/env python3
"""
多智能体协作系统 - 扩展版（35+ 专业 Agent）
Multi-Agent Team - Extended Version (35+ Specialized Agents)

新增领域：
- 🏥 医疗健康
- 🔬 科学研究  
- 🏠 房地产
- 🛒 电子商务
- 📰 新闻媒体
- ⚽ 体育运动
- 🍽️ 餐饮美食
- ✈️ 旅游出行
- 👗 时尚潮流
- 🎮 游戏娱乐
"""

import asyncio
import json
import os
from datetime import datetime
from enum import Enum
from typing import Dict, List
from dataclasses import dataclass


# ============ Agent 角色定义 ============

class AgentRole(Enum):
    # === 核心管理层 ===
    COMMANDER = "commander"
    PRODUCT_MANAGER = "product"
    PROJECT_MANAGER = "pm"
    
    # === 研究与分析 ===
    RESEARCHER = "researcher"
    DATA_ANALYST = "data_analyst"
    BUSINESS_ANALYST = "business"
    MARKET_ANALYST = "market"
    
    # === 内容与创意 ===
    WRITER = "writer"
    EDITOR = "editor"
    TRANSLATOR = "translator"
    COPYWRITER = "copywriter"
    CREATIVE_DESIGNER = "creative"
    UI_UX_DESIGNER = "ui_ux"
    
    # === 技术专家 ===
    SENIOR_ENGINEER = "engineer"
    DEVOPS专家 = "devops"
    SECURITY专家 = "security"
    
    # === 专业顾问 ===
    FINANCIAL_ADVISOR = "finance"
    LEGAL_ADVISOR = "legal"
    SEO专家 = "seo"
    MARKETING专家 = "marketing"
    HR专家 = "hr"
    
    # === 健康与生活 ===
    HEALTH_ADVISOR = "health"
    PSYCHOLOGIST = "psychologist"
    NUTRITIONIST = "nutritionist"
    FITNESS_COACH = "fitness"
    
    # === 行业专家（新增）===
    MEDICAL_EXPERT = "medical"       # 医疗专家
    SCIENTIST = "scientist"          # 科学家
    REAL_ESTATE_EXPERT = "realestate" # 房地产专家
    ECOMMERCE_EXPERT = "ecommerce"   # 电商专家
    NEWS_ANALYST = "news"            # 新闻分析师
    SPORTS_EXPERT = "sports"         # 体育专家
    FOOD_CRITIC = "food"             # 美食评论家
    TRAVEL_EXPERT = "travel"        # 旅游专家
    FASHION_CONSULTANT = "fashion"   # 时尚顾问
    GAMING_EXPERT = "gaming"        # 游戏专家
    
    # === 教育与职业 ===
    EDUCATION_EXPERT = "education"
    CAREER_COACH = "career"
    
    # === 质量保障 ===
    REVIEWER = "reviewer"
    QA_ENGINEER = "qa"


# ============ 35+ 专业 Agent 定义 ============

@dataclass
class Agent:
    name: str
    role: AgentRole
    specialty: str
    category: str
    color: str
    system_prompt: str


# 完整的 Agent 团队

AGENTS = {
    # ============ 核心管理层 ============
    AgentRole.COMMANDER: Agent(
        name="指挥官",
        role=AgentRole.COMMANDER,
        specialty="任务分解与调度",
        category="管理层",
        color="👑",
        system_prompt="""你是一个项目指挥官，负责领导多智能体团队完成复杂任务。"""
    ),
    AgentRole.PRODUCT_MANAGER: Agent(
        name="产品经理",
        role=AgentRole.PRODUCT_MANAGER,
        specialty="产品规划与需求分析",
        category="管理层",
        color="📦",
        system_prompt="""你是一个专业产品经理，负责产品规划和需求分析。"""
    ),
    AgentRole.PROJECT_MANAGER: Agent(
        name="项目经理",
        role=AgentRole.PROJECT_MANAGER,
        specialty="项目管理与进度控制",
        category="管理层",
        color="📋",
        system_prompt="""你是一个专业项目经理，负责项目管理和进度控制。"""
    ),
    
    # ============ 研究与分析 ============
    AgentRole.RESEARCHER: Agent(
        name="研究员",
        role=AgentRole.RESEARCHER,
        specialty="信息搜索与分析",
        category="研究",
        color="🔍",
        system_prompt="""你是一个专业研究员，负责收集和整理信息。"""
    ),
    AgentRole.DATA_ANALYST: Agent(
        name="数据科学家",
        role=AgentRole.DATA_ANALYST,
        specialty="数据处理与机器学习",
        category="研究",
        color="🧮",
        system_prompt="""你是一个专业数据科学家，负责数据处理和高级分析。"""
    ),
    AgentRole.BUSINESS_ANALYST: Agent(
        name="商业分析师",
        role=AgentRole.BUSINESS_ANALYST,
        specialty="商业模式与竞争分析",
        category="研究",
        color="📊",
        system_prompt="""你是一个专业商业分析师，负责商业模式和竞争分析。"""
    ),
    AgentRole.MARKET_ANALYST: Agent(
        name="市场分析师",
        role=AgentRole.MARKET_ANALYST,
        specialty="市场趋势与用户行为",
        category="研究",
        color="📈",
        system_prompt="""你是一个专业市场分析师，负责市场趋势和用户行为分析。"""
    ),
    
    # ============ 内容与创意 ============
    AgentRole.WRITER: Agent(
        name="专业作家",
        role=AgentRole.WRITER,
        specialty="长内容创作",
        category="内容",
        color="📝",
        system_prompt="""你是一个专业作家，负责长内容创作。"""
    ),
    AgentRole.EDITOR: Agent(
        name="编辑",
        role=AgentRole.EDITOR,
        specialty="内容编辑与润色",
        category="内容",
        color="✏️",
        system_prompt="""你是一个专业编辑，负责内容编辑和润色。"""
    ),
    AgentRole.TRANSLATOR: Agent(
        name="翻译官",
        role=AgentRole.TRANSLATOR,
        specialty="专业翻译与本地化",
        category="内容",
        color="🌐",
        system_prompt="""你是一个专业翻译官，负责多语言翻译和本地化。"""
    ),
    AgentRole.COPYWRITER: Agent(
        name="文案策划",
        role=AgentRole.COPYWRITER,
        specialty="营销文案与广告创意",
        category="内容",
        color="📢",
        system_prompt="""你是一个专业文案策划，负责营销文案和广告创意。"""
    ),
    AgentRole.CREATIVE_DESIGNER: Agent(
        name="创意总监",
        role=AgentRole.CREATIVE_DESIGNER,
        specialty="创意策划与视觉设计",
        category="创意",
        color="🎨",
        system_prompt="""你是一个创意总监，负责创意策划和视觉设计。"""
    ),
    AgentRole.UI_UX_DESIGNER: Agent(
        name="UI/UX 设计师",
        role=AgentRole.UI_UX_DESIGNER,
        specialty="用户体验设计",
        category="创意",
        color="🖥️",
        system_prompt="""你是一个 UI/UX 设计师，负责用户体验设计。"""
    ),
    
    # ============ 技术专家 ============
    AgentRole.SENIOR_ENGINEER: Agent(
        name="高级工程师",
        role=AgentRole.SENIOR_ENGINEER,
        specialty="架构设计与代码审查",
        category="技术",
        color="💻",
        system_prompt="""你是一个高级工程师，负责架构设计和代码审查。"""
    ),
    AgentRole.DEVOPS专家: Agent(
        name="DevOps 工程师",
        role=AgentRole.DEVOPS专家,
        specialty="自动化运维与云架构",
        category="技术",
        color="☁️",
        system_prompt="""你是一个 DevOps 工程师，负责自动化运维和云架构。"""
    ),
    AgentRole.SECURITY专家: Agent(
        name="安全专家",
        role=AgentRole.SECURITY专家,
        specialty="安全审计与渗透测试",
        category="技术",
        color="🔐",
        system_prompt="""你是一个安全专家，负责安全审计和渗透测试。"""
    ),
    
    # ============ 专业顾问 ============
    AgentRole.FINANCIAL_ADVISOR: Agent(
        name="财务顾问",
        role=AgentRole.FINANCIAL_ADVISOR,
        specialty="投资理财与财务规划",
        category="顾问",
        color="💰",
        system_prompt="""你是一个专业财务顾问，负责投资理财分析和建议。"""
    ),
    AgentRole.LEGAL_ADVISOR: Agent(
        name="法律顾问",
        role=AgentRole.LEGAL_ADVISOR,
        specialty="法律合规与合同审查",
        category="顾问",
        color="⚖️",
        system_prompt="""你是一个专业法律顾问，负责法律合规和合同审查。"""
    ),
    AgentRole.SEO专家: Agent(
        name="SEO 专家",
        role=AgentRole.SEO专家,
        specialty="搜索引擎优化",
        category="顾问",
        color="🔎",
        system_prompt="""你是一个 SEO 专家，负责搜索引擎优化。"""
    ),
    AgentRole.MARKETING专家: Agent(
        name="营销专家",
        role=AgentRole.MARKETING专家,
        specialty="数字营销与增长策略",
        category="顾问",
        color="🚀",
        system_prompt="""你是一个营销专家，负责数字营销和增长策略。"""
    ),
    AgentRole.HR专家: Agent(
        name="HR 专家",
        role=AgentRole.HR专家,
        specialty="人才招聘与组织发展",
        category="顾问",
        color="👥",
        system_prompt="""你是一个 HR 专家，负责人才招聘和组织发展。"""
    ),
    
    # ============ 健康与生活 ============
    AgentRole.HEALTH_ADVISOR: Agent(
        name="健康顾问",
        role=AgentRole.HEALTH_ADVISOR,
        specialty="健康管理与生活方式",
        category="健康",
        color="🏃",
        system_prompt="""你是一个健康顾问，负责健康管理和生活方式建议。"""
    ),
    AgentRole.PSYCHOLOGIST: Agent(
        name="心理咨询师",
        role=AgentRole.PSYCHOLOGIST,
        specialty="心理健康与情绪管理",
        category="健康",
        color="🧠",
        system_prompt="""你是一个心理咨询师，负责心理健康和情绪管理。"""
    ),
    AgentRole.NUTRITIONIST: Agent(
        name="营养师",
        role=AgentRole.NUTRITIONIST,
        specialty="营养饮食与膳食规划",
        category="健康",
        color="🥗",
        system_prompt="""你是一个专业营养师，负责营养饮食和膳食规划。"""
    ),
    AgentRole.FITNESS_COACH: Agent(
        name="健身教练",
        role=AgentRole.FITNESS_COACH,
        specialty="运动健身与体能训练",
        category="健康",
        color="💪",
        system_prompt="""你是一个专业健身教练，负责运动健身和体能训练。"""
    ),
    
    # ============ 🔬 医疗健康专家（新增） ============
    AgentRole.MEDICAL_EXPERT: Agent(
        name="医疗专家",
        role=AgentRole.MEDICAL_EXPERT,
        specialty="疾病诊断与治疗方案",
        category="医疗",
        color="🏥",
        system_prompt="""你是一个专业医疗专家，负责疾病知识科普和治疗方案分析。

职责：
1. 常见疾病知识科普
2. 症状分析与就医建议
3. 药物作用与副作用
4. 预防保健方法
5. 体检报告解读

请用通俗易懂的语言回答，非紧急情况建议就医。
请用 JSON 格式回复。
"""
    ),
    
    # ============ 🔬 科学研究专家（新增） ============
    AgentRole.SCIENTIST: Agent(
        name="科学家",
        role=AgentRole.SCIENTIST,
        specialty="前沿科技与学术研究",
        category="科研",
        color="🔬",
        system_prompt="""你是一个专业科学家，负责前沿科技和学术研究分析。

职责：
1. 最新科研进展解读
2. 科学原理科普
3. 研究方法论分析
4. 学术论文解读
5. 科技趋势预测

请用 JSON 格式回复分析结果。
"""
    ),
    
    # ============ 🏠 房地产专家（新增） ============
    AgentRole.REAL_ESTATE_EXPERT: Agent(
        name="房地产专家",
        role=AgentRole.REAL_ESTATE_EXPERT,
        specialty="房产投资与市场分析",
        category="房产",
        color="🏠",
        system_prompt="""你是一个专业房地产专家，负责房产投资和市场分析。

职责：
1. 房产市场趋势分析
2. 投资回报率计算
3. 区位分析建议
4. 购房/租房攻略
5. 政策解读

请用 JSON 格式回复分析结果。
"""
    ),
    
    # ============ 🛒 电商专家（新增） ============
    AgentRole.ECOMMERCE_EXPERT: Agent(
        name="电商专家",
        role=AgentRole.ECOMMERCE_EXPERT,
        specialty="电商运营与选品策略",
        category="电商",
        color="🛒",
        system_prompt="""你是一个专业电商专家，负责电商运营和选品策略。

职责：
1. 电商平台分析
2. 选品策略建议
3. 运营技巧分享
4. 促销策略规划
5. 竞品分析

请用 JSON 格式回复建议。
"""
    ),
    
    # ============ 📰 新闻分析师（新增） ============
    AgentRole.NEWS_ANALYST: Agent(
        name="新闻分析师",
        role=AgentRole.NEWS_ANALYST,
        specialty="新闻解读与舆情分析",
        category="媒体",
        color="📰",
        system_prompt="""你是一个专业新闻分析师，负责新闻解读和舆情分析。

职责：
1. 新闻事件解读
2. 舆情走向分析
3. 信息真实性核实
4. 深度报道策划
5. 公关策略建议

请用 JSON 格式回复分析结果。
"""
    ),
    
    # ============ ⚽ 体育专家（新增） ============
    AgentRole.SPORTS_EXPERT: Agent(
        name="体育专家",
        role=AgentRole.SPORTS_EXPERT,
        specialty="体育赛事与运动分析",
        category="体育",
        color="⚽",
        system_prompt="""你是一个专业体育专家，负责体育赛事和运动分析。

职责：
1. 赛事预测分析
2. 运动技巧指导
3. 运动员/球队分析
4. 体育新闻点评
5. 运动装备推荐

请用 JSON 格式回复分析结果。
"""
    ),
    
    # ============ 🍽️ 美食评论家（新增） ============
    AgentRole.FOOD_CRITIC: Agent(
        name="美食评论家",
        role=AgentRole.FOOD_CRITIC,
        specialty="美食评测与餐厅推荐",
        category="美食",
        color="🍽️",
        system_prompt="""你是一个专业美食评论家，负责美食评测和餐厅推荐。

职责：
1. 菜系特色介绍
2. 餐厅评测推荐
3. 烹饪技巧分享
4. 美食文化科普
5. 食材选购建议

请用 JSON 格式回复推荐结果。
"""
    ),
    
    # ============ ✈️ 旅游专家（新增） ============
    AgentRole.TRAVEL_EXPERT: Agent(
        name="旅游专家",
        role=AgentRole.TRAVEL_EXPERT,
        specialty="旅游攻略与目的地分析",
        category="旅游",
        color="✈️",
        system_prompt="""你是一个专业旅游专家，负责旅游攻略和目的地分析。

职责：
1. 旅游目的地推荐
2. 行程规划建议
3. 省钱技巧分享
4. 当地文化介绍
5. 旅行注意事项

请用 JSON 格式回复建议。
"""
    ),
    
    # ============ 👗 时尚顾问（新增） ============
    AgentRole.FASHION_CONSULTANT: Agent(
        name="时尚顾问",
        role=AgentRole.FASHION_CONSULTANT,
        specialty="时尚趋势与穿搭建议",
        category="时尚",
        color="👗",
        system_prompt="""你是一个专业时尚顾问，负责时尚趋势和穿搭建议。

职责：
1. 流行趋势分析
2. 穿搭风格建议
3. 护肤美妆推荐
4. 品牌档次解读
5. 场合着装指导

请用 JSON 格式回复建议。
"""
    ),
    
    # ============ 🎮 游戏专家（新增） ============
    AgentRole.GAMING_EXPERT: Agent(
        name="游戏专家",
        role=AgentRole.GAMING_EXPERT,
        specialty="游戏评测与攻略",
        category="游戏",
        color="🎮",
        system_prompt="""你是一个专业游戏专家，负责游戏评测和攻略。

职责：
1. 游戏评测分析
2. 通关攻略指导
3. 游戏主机/配置推荐
4. 电竞赛事点评
5. 游戏行业动态

请用 JSON 格式回复建议。
"""
    ),
    
    # ============ 教育与职业 ============
    AgentRole.EDUCATION_EXPERT: Agent(
        name="教育专家",
        role=AgentRole.EDUCATION_EXPERT,
        specialty="学习规划与教育培训",
        category="教育",
        color="📚",
        system_prompt="""你是一个教育专家，负责学习规划和教育培训。"""
    ),
    AgentRole.CAREER_COACH: Agent(
        name="职业教练",
        role=AgentRole.CAREER_COACH,
        specialty="职业规划与发展建议",
        category="职业",
        color="💼",
        system_prompt="""你是一个专业职业教练，负责职业规划和发展建议。"""
    ),
    
    # ============ 质量保障 ============
    AgentRole.REVIEWER: Agent(
        name="质量审核员",
        role=AgentRole.REVIEWER,
        specialty="内容质量把关",
        category="质量",
        color="✅",
        system_prompt="""你是一个质量审核员，负责检查内容质量和准确性。"""
    ),
    AgentRole.QA_ENGINEER: Agent(
        name="QA 工程师",
        role=AgentRole.QA_ENGINEER,
        specialty="测试策略与质量保障",
        category="质量",
        color="🧪",
        system_prompt="""你是一个 QA 工程师，负责测试策略和质量保障。"""
    ),
}


# ============ 智能 Agent 选择器 ============

class AgentSelector:
    """智能 Agent 选择器"""
    
    KEYWORDS = {
        AgentRole.MEDICAL_EXPERT: ["医疗", "疾病", "症状", "治疗", "药物", "健康检查", "体检", "医院", "医生"],
        AgentRole.SCIENTIST: ["科学", "研究", "论文", "实验", "学术", "科技", "物理", "化学", "生物"],
        AgentRole.REAL_ESTATE_EXPERT: ["房产", "买房", "租房", "房价", "楼盘", "房贷", "房地产", "住宅"],
        AgentRole.ECOMMERCE_EXPERT: ["电商", "网店", "淘宝", "京东", "亚马逊", "选品", "运营", "直播带货"],
        AgentRole.NEWS_ANALYST: ["新闻", "舆论", "媒体", "公关", "传播", "报道", "记者"],
        AgentRole.SPORTS_EXPERT: ["体育", "足球", "篮球", "比赛", "运动员", "赛事", "健身", "运动"],
        AgentRole.FOOD_CRITIC: ["美食", "餐厅", "菜系", "烹饪", "食谱", "吃", "美食推荐", "餐厅推荐"],
        AgentRole.TRAVEL_EXPERT: ["旅游", "旅行", "景点", "酒店", "机票", "行程", "度假", "攻略"],
        AgentRole.FASHION_CONSULTANT: ["时尚", "穿搭", "衣服", "护肤", "美妆", "化妆品", "潮流", "品牌"],
        AgentRole.GAMING_EXPERT: ["游戏", "电竞", "手游", "Steam", "Switch", "PS5", "Xbox", "通关"],
        AgentRole.NUTRITIONIST: ["营养", "饮食", "减肥", "增肌", "维生素", "膳食", "卡路里"],
        AgentRole.FITNESS_COACH: ["健身", "运动", "训练", "体能", "锻炼", "瑜伽", "跑步"],
        AgentRole.CAREER_COACH: ["职业", "工作", "简历", "面试", "跳槽", "升职", "职场"],
        AgentRole.PSYCHOLOGIST: ["心理", "情绪", "焦虑", "压力", "抑郁", "心理咨询"],
        AgentRole.HEALTH_ADVISOR: ["健康", "养生", "保健", "体检", "亚健康"],
    }
    
    @classmethod
    def select_agents(cls, request: str) -> List[AgentRole]:
        """根据请求智能选择 Agent"""
        
        request_lower = request.lower()
        selected = [AgentRole.COMMANDER, AgentRole.RESEARCHER, AgentRole.WRITER, AgentRole.REVIEWER]
        
        # 关键词匹配
        for agent_role, keywords in cls.KEYWORDS.items():
            if any(kw in request_lower for kw in keywords):
                if agent_role not in selected:
                    selected.append(agent_role)
        
        # 基础匹配
        if any(kw in request_lower for kw in ["投资", "理财", "BTC", "股票"]):
            if AgentRole.FINANCIAL_ADVISOR not in selected:
                selected.append(AgentRole.FINANCIAL_ADVISOR)
        
        if any(kw in request_lower for kw in ["合同", "法律", "合规"]):
            if AgentRole.LEGAL_ADVISOR not in selected:
                selected.append(AgentRole.LEGAL_ADVISOR)
        
        if any(kw in request_lower for kw in ["创意", "设计", "视觉"]):
            if AgentRole.CREATIVE_DESIGNER not in selected:
                selected.append(AgentRole.CREATIVE_DESIGNER)
        
        return list(set(selected))


# ============ 快速参考 ============

def get_category_count() -> Dict[str, int]:
    """统计各类别 Agent 数量"""
    counts = {}
    for role, agent in AGENTS.items():
        cat = agent.category
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def list_agents_by_category():
    """按类别列出所有 Agent"""
    categories = {}
    for role, agent in AGENTS.items():
        cat = agent.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(agent)
    
    for cat, agents in categories.items():
        print(f"\n📂 {cat}")
        print("-" * 50)
        for agent in agents:
            print(f"   {agent.color} {agent.name:<10} - {agent.specialty}")


# ============ CLI ============

def main():
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   🤖 OpenClaw 多智能体协作系统 - 扩展版               ║
║   35+ 专业领域专家，涵盖生活方方面面                   ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            list_agents_by_category()
        elif sys.argv[1] == "--count":
            counts = get_category_count()
            total = sum(counts.values())
            print(f"\n📊 Agent 统计：")
            for cat, count in counts.items():
                print(f"   {cat}: {count} 位")
            print(f"\n   总计：{total} 位专业 Agent")
        else:
            print("\n用法：")
            print("  python multi_agent_extended.py --list   # 查看所有专家")
            print("  python multi_agent_extended.py --count  # 查看统计")
    else:
        counts = get_category_count()
        total = sum(counts.values())
        
        print(f"📊 已有 {total} 位专业 Agent：\n")
        
        for cat, count in counts.items():
            print(f"   📂 {cat}: {count} 位")
        
        print("\n" + "="*60)
        print("\n使用方式：")
        print("  python multi_agent_extended.py --list   # 查看所有专家")
        print("  python multi_agent_extended.py --count  # 查看统计")
        print("\n示例：")
        print('  python multi_agent.py "推荐北京美食餐厅"')
        print('  python multi_agent.py "分析最新科技趋势"')
        print('  python multi_agent.py "制定健身计划"')


if __name__ == "__main__":
    main()
