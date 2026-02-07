#!/usr/bin/env python3
"""
自动化学习系统主控脚本
Auto Learning System Controller

功能:
1. 每日定时搜索AI/技术新知识
2. 自动抓取并摘要重要内容
3. 分类存入知识库
4. 生成学习报告

使用方法:
  python auto_learn.py --daily          # 执行每日学习任务
  python auto_learn.py --weekly-report  # 生成周报
  python auto_learn.py --search         # 手动搜索
  python auto_learn.py --help           # 查看帮助
"""

import os
import sys
import argparse
import logging
import yaml
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import hashlib

# 导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.searcher import Searcher
from modules.summarizer import Summarizer
from modules.knowledge_base import KnowledgeBase
from modules.reporter import Reporter


class AutoLearningSystem:
    """自动化学习系统主类"""
    
    def __init__(self, config_path: str = None):
        """初始化系统"""
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "config.yaml"
        )
        self.config = self._load_config()
        self._setup_logging()
        self._init_modules()
    
    def _load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                # 展开路径
                config['storage']['knowledge_base_path'] = os.path.expanduser(
                    config['storage']['knowledge_base_path']
                )
                config['storage']['reports_path'] = os.path.expanduser(
                    config['storage']['reports_path']
                )
                config['storage']['logs_path'] = os.path.expanduser(
                    config['storage']['logs_path']
                )
                return config
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            sys.exit(1)
    
    def _setup_logging(self):
        """配置日志"""
        log_path = self.config['storage']['logs_path']
        os.makedirs(log_path, exist_ok=True)
        
        log_file = os.path.join(
            log_path, 
            f"auto_learn_{datetime.now().strftime('%Y%m%d')}.log"
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_modules(self):
        """初始化各模块"""
        self.searcher = Searcher(self.config, self.logger)
        self.summarizer = Summarizer(self.config, self.logger)
        self.knowledge_base = KnowledgeBase(self.config, self.logger)
        self.reporter = Reporter(self.config, self.logger)
    
    def run_daily_task(self):
        """执行每日学习任务"""
        self.logger.info("=" * 50)
        self.logger.info("开始每日学习任务")
        self.logger.info("=" * 50)
        
        try:
            # 1. 搜索新知识
            self.logger.info("步骤1: 搜索AI/技术新知识...")
            articles = self.searcher.search_all()
            self.logger.info(f"搜索到 {len(articles)} 篇文章")
            
            if not articles:
                self.logger.warning("未搜索到任何文章，任务结束")
                return
            
            # 2. 摘要和分类
            self.logger.info("步骤2: 摘要和分类内容...")
            processed_articles = []
            for article in articles:
                try:
                    # 摘要
                    summary = self.summarizer.summarize(article)
                    # 分类
                    category = self.knowledge_base.categorize(summary)
                    article['summary'] = summary
                    article['category'] = category
                    article['processed_at'] = datetime.now().isoformat()
                    processed_articles.append(article)
                except Exception as e:
                    self.logger.warning(f"处理文章失败: {e}")
                    continue
            
            self.logger.info(f"成功处理 {len(processed_articles)} 篇文章")
            
            # 3. 存入知识库
            self.logger.info("步骤3: 存入知识库...")
            saved_count = self.knowledge_base.save_articles(processed_articles)
            self.logger.info(f"保存 {saved_count} 篇文章到知识库")
            
            # 4. 生成日报
            self.logger.info("步骤4: 生成学习日报...")
            today = datetime.now().strftime('%Y-%m-%d')
            report = self.reporter.generate_daily_report(today)
            self.logger.info(f"日报已生成: {report}")
            
            self.logger.info("=" * 50)
            self.logger.info("每日学习任务完成!")
            self.logger.info("=" * 50)
            
        except Exception as e:
            self.logger.error(f"每日任务执行失败: {e}")
            raise
    
    def run_weekly_report(self):
        """生成周报"""
        self.logger.info("生成周度学习报告...")
        
        # 获取上周日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        report = self.reporter.generate_weekly_report(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        self.logger.info(f"周报已生成")
        return report
    
    def manual_search(self, keywords: List[str] = None):
        """手动搜索"""
        if keywords:
            self.config['search_keywords'] = keywords
        
        articles = self.searcher.search_all()
        return articles


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description='自动化学习系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python auto_learn.py --daily          # 执行每日学习
  python auto_learn.py --weekly-report  # 生成周报
  python auto_learn.py --search --keywords "AI" "Python"  # 手动搜索
        """
    )
    
    parser.add_argument('--daily', action='store_true', help='执行每日学习任务')
    parser.add_argument('--weekly-report', action='store_true', help='生成周度报告')
    parser.add_argument('--search', action='store_true', help='手动搜索')
    parser.add_argument('--keywords', nargs='+', help='搜索关键词')
    parser.add_argument('--config', default=None, help='配置文件路径')
    
    args = parser.parse_args()
    
    # 确定脚本路径
    if args.config:
        script_dir = os.path.dirname(os.path.abspath(args.config))
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 初始化系统
    system = AutoLearningSystem(
        config_path=args.config or os.path.join(script_dir, 'config.yaml')
    )
    
    # 执行任务
    if args.daily:
        system.run_daily_task()
    elif args.weekly_report:
        system.run_weekly_report()
    elif args.search:
        articles = system.manual_search(args.keywords)
        for article in articles:
            print(f"- {article.get('title', 'Unknown')}")
            print(f"  {article.get('url', '')}")
    else:
        # 默认执行每日任务
        system.run_daily_task()


if __name__ == '__main__':
    main()
