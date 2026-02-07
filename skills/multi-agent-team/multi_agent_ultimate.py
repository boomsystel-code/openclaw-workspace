#!/usr/bin/env python3
"""
OpenClaw 多智能体协作系统 - 终极版
60+ 专业领域专家
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List
import sys


# Agent 角色定义
class AgentRole(Enum):
    # 核心
    COMMANDER = "commander"
    RESEARCHER = "researcher"
    WRITER = "writer"
    FINANCIAL = "finance"
    LEGAL = "legal"
    HEALTH = "health"
    PSYCHOLOGIST = "psychologist"
    
    # 扩展
    ENVIRONMENT = "environment"
    AUTOMOBILE = "automobile"
    POLITICAL = "political"
    AGRICULTURE = "agriculture"
    INDUSTRY = "industry"
    GADGET = "gadget"
    BANKING = "banking"
    FILM = "film"
    MUSIC = "music"
    PUBLISHING = "publishing"
    GOVERNMENT = "government"
    GENETICS = "genetics"
    OUTDOOR = "outdoor"
    PET = "pet"
    ANTIQUE = "antique"
    VC = "vc"
    BUSINESS = "business"
    ASTROLOGY = "astrology"
    WINE = "wine"
    GARDENING = "gardening"
    GEOGRAPHY = "geography"
    MATH = "math"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    ASTRONOMY = "astronomy"
    MICROBIO = "microbio"
    DEVICE = "device"
    PHARMA = "pharma"
    MEDITATION = "meditation"
    SCICOMM = "scicomm"


@dataclass
class Agent:
    name: str
    role: AgentRole
    specialty: str
    category: str
    color: str


# Agent 定义
AGENTS = {
    AgentRole.COMMANDER: Agent("指挥官", AgentRole.COMMANDER, "任务分解", "核心", "👑"),
    AgentRole.RESEARCHER: Agent("研究员", AgentRole.RESEARCHER, "信息分析", "核心", "🔍"),
    AgentRole.WRITER: Agent("作家", AgentRole.WRITER, "内容创作", "核心", "📝"),
    AgentRole.FINANCIAL: Agent("财务顾问", AgentRole.FINANCIAL, "投资理财", "核心", "💰"),
    AgentRole.LEGAL: Agent("法律顾问", AgentRole.LEGAL, "法律合规", "核心", "⚖️"),
    AgentRole.HEALTH: Agent("健康顾问", AgentRole.HEALTH, "健康管理", "核心", "🏥"),
    AgentRole.PSYCHOLOGIST: Agent("心理咨询师", AgentRole.PSYCHOLOGIST, "心理健康", "核心", "🧠"),
    
    # 扩展领域
    AgentRole.ENVIRONMENT: Agent("环保专家", AgentRole.ENVIRONMENT, "碳中和/污染治理", "环保", "🌱"),
    AgentRole.AUTOMOBILE: Agent("汽车专家", AgentRole.AUTOMOBILE, "评测/选购/养车", "汽车", "🚗"),
    AgentRole.POLITICAL: Agent("政治分析师", AgentRole.POLITICAL, "政策/国际形势", "政治", "🏛️"),
    AgentRole.AGRICULTURE: Agent("农业专家", AgentRole.AGRICULTURE, "种植/养殖/农产品", "农业", "🌾"),
    AgentRole.INDUSTRY: Agent("工业专家", AgentRole.INDUSTRY, "制造/产业链", "工业", "🏭"),
    AgentRole.GADGET: Agent("数码专家", AgentRole.GADGET, "手机/电脑/相机", "数码", "📱"),
    AgentRole.BANKING: Agent("银行业专家", AgentRole.BANKING, "理财/贷款/信用卡", "金融", "🏦"),
    AgentRole.FILM: Agent("影视专家", AgentRole.FILM, "电影/电视剧/综艺", "娱乐", "🎬"),
    AgentRole.MUSIC: Agent("音乐专家", AgentRole.MUSIC, "音乐/乐器/流派", "艺术", "🎵"),
    AgentRole.PUBLISHING: Agent("出版专家", AgentRole.PUBLISHING, "图书/作家/阅读", "出版", "📚"),
    AgentRole.GOVERNMENT: Agent("公共事务专家", AgentRole.GOVERNMENT, "政策/治理/规划", "公共", "🏛️"),
    AgentRole.GENETICS: Agent("基因专家", AgentRole.GENETICS, "基因/精准医疗", "科技", "🧬"),
    AgentRole.OUTDOOR: Agent("户外专家", AgentRole.OUTDOOR, "徒步/登山/露营", "户外", "🏔️"),
    AgentRole.PET: Agent("宠物专家", AgentRole.PET, "养护/医疗/行为", "宠物", "🐾"),
    AgentRole.ANTIQUE: Agent("古董专家", AgentRole.ANTIQUE, "鉴定/收藏/拍卖", "收藏", "🏺"),
    AgentRole.VC: Agent("投资人", AgentRole.VC, "创业/融资/估值", "投资", "💼"),
    AgentRole.BUSINESS: Agent("商业顾问", AgentRole.BUSINESS, "战略/组织/运营", "咨询", "📊"),
    AgentRole.ASTROLOGY: Agent("玄学顾问", AgentRole.ASTROLOGY, "星座/生肖/风水", "玄学", "🔮"),
    AgentRole.WINE: Agent("酒水专家", AgentRole.WINE, "品鉴/文化/选酒", "品鉴", "🍺"),
    AgentRole.GARDENING: Agent("园艺专家", AgentRole.GARDENING, "种植/植物/养护", "园艺", "🌿"),
    AgentRole.GEOGRAPHY: Agent("地理专家", AgentRole.GEOGRAPHY, "地貌/气候/资源", "地理", "🌍"),
    AgentRole.MATH: Agent("数学家", AgentRole.MATH, "理论/应用/趣味", "数学", "🧮"),
    AgentRole.PHYSICS: Agent("物理学家", AgentRole.PHYSICS, "量子/相对论/粒子", "科学", "📐"),
    AgentRole.CHEMISTRY: Agent("化学家", AgentRole.CHEMISTRY, "有机/材料/环境", "科学", "🧪"),
    AgentRole.ASTRONOMY: Agent("天文学家", AgentRole.ASTRONOMY, "宇宙/观测/航天", "天文", "🌌"),
    AgentRole.MICROBIO: Agent("微生物学家", AgentRole.MICROBIO, "细菌/免疫/疫苗", "生物", "🦠"),
    AgentRole.DEVICE: Agent("器械专家", AgentRole.DEVICE, "设备/诊断/康复", "医疗", "🏥"),
    AgentRole.PHARMA: Agent("制药专家", AgentRole.PHARMA, "药物/研发/临床", "医药", "💊"),
    AgentRole.MEDITATION: Agent("冥想教练", AgentRole.MEDITATION, "正念/呼吸/减压", "身心", "🧘"),
    AgentRole.SCICOMM: Agent("科学传播者", AgentRole.SCICOMM, "科普/可视化/写作", "科普", "🔬"),
    
    # ============ 🏥 中药养生 ============
    TCM_EXPERT = "tcm",  # 中医养生专家
}

    # ============ 🏥 中药养生 ============
    TCM_EXPERT = "tcm",  # 中医养生专家


def get_stats():
    categories = {}
    for role, agent in AGENTS.items():
        cat = agent.category
        categories[cat] = categories.get(cat, 0) + 1
    return categories


def list_all():
    categories = {}
    for role, agent in AGENTS.items():
        cat = agent.category
        categories.setdefault(cat, []).append(agent)
    
    for cat, agents in sorted(categories.items()):
        print(f"\n📂 {cat}")
        print("-" * 50)
        for agent in agents:
            print(f"   {agent.color} {agent.name:<10} - {agent.specialty}")


def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   🤖 OpenClaw 多智能体协作系统 - 终极版               ║
║   60+ 专业领域专家，涵盖生活方方面面                   ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
""")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            list_all()
        elif sys.argv[1] == "--count":
            categories = get_stats()
            total = sum(categories.values())
            print(f"\n📊 Agent 统计（共 {total} 位）：\n")
            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                print(f"   📂 {cat}: {count} 位")
    else:
        categories = get_stats()
        total = sum(categories.values())
        print(f"📊 已有 {total} 位专业 Agent：\n")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"   📂 {cat}: {count} 位")
        print("\n" + "="*60)
        print("\n用法：python multi_agent_ultimate.py --list | --count")


if __name__ == "__main__":
    main()

# ============ 🏥 中药养生 ============
class TCM(Enum):  # Traditional Chinese Medicine
    TCM_EXPERT = "tcm"  # 中医专家


# 中药养生专家
AGENTS[TCM.TCM_EXPERT] = Agent(
    name="中医养生专家",
    role=TCM.TCM_EXPERT,
    specialty="中药/食疗/养生",
    category="养生",
    color="🏮",
)
