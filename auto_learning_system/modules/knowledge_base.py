"""
知识库模块
负责文章的存储、分类和管理
"""

import os
import json
import logging
from typing import List, Dict
from datetime import datetime
from pathlib import Path
import hashlib


class KnowledgeBase:
    """知识库管理"""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.base_path = config['storage']['knowledge_base_path']
        self._ensure_directories()
        
        # 分类配置
        self.categories = config.get('categories', [])
    
    def _ensure_directories(self):
        """确保目录存在"""
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(os.path.join(self.base_path, 'articles'), exist_ok=True)
        os.makedirs(os.path.join(self.base_path, 'by_category'), exist_ok=True)
        os.makedirs(os.path.join(self.base_path, 'index'), exist_ok=True)
    
    def categorize(self, article_summary: Dict) -> str:
        """为文章分类"""
        summary_text = article_summary.get('summary', '').lower()
        keywords = article_summary.get('keywords', [])
        content = article_summary.get('content_preview', '').lower()
        
        all_text = f"{summary_text} {' '.join(keywords)} {content}"
        
        best_category = 'general'
        best_score = 0
        
        for category in self.categories:
            category_keywords = category.get('keywords', [])
            score = sum(1 for kw in category_keywords if kw.lower() in all_text)
            
            if score > best_score:
                best_score = score
                best_category = category.get('id', 'general')
        
        return best_category
    
    def save_articles(self, articles: List[Dict]) -> int:
        """保存文章到知识库"""
        saved_count = 0
        
        for article in articles:
            try:
                self._save_article(article)
                saved_count += 1
            except Exception as e:
                self.logger.warning(f"保存文章失败: {e}")
        
        # 更新索引
        self._update_index()
        
        return saved_count
    
    def _save_article(self, article: Dict):
        """保存单篇文章"""
        # 生成唯一ID
        url = article.get('url', '')
        article_id = hashlib.md5(url.encode()).hexdigest()[:12]
        
        # 保存文件
        article_path = os.path.join(
            self.base_path, 
            'articles', 
            f"{article_id}.json"
        )
        
        # 如果已存在，跳过
        if os.path.exists(article_path):
            self.logger.debug(f"文章已存在: {article.get('title', 'Unknown')}")
            return
        
        # 添加元数据
        article['id'] = article_id
        article['saved_at'] = datetime.now().isoformat()
        
        # 保存到文件
        with open(article_path, 'w', encoding='utf-8') as f:
            json.dump(article, f, ensure_ascii=False, indent=2)
        
        # 保存到分类目录
        category = article.get('category', 'general')
        category_path = os.path.join(
            self.base_path, 
            'by_category',
            f"{category}.jsonl"
        )
        
        with open(category_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        self.logger.debug(f"保存文章: {article.get('title', 'Unknown')}")
    
    def _update_index(self):
        """更新索引"""
        index = {
            'last_updated': datetime.now().isoformat(),
            'articles': {}
        }
        
        articles_dir = os.path.join(self.base_path, 'articles')
        
        for filename in os.listdir(articles_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(articles_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    article = json.load(f)
                    index['articles'][article['id']] = {
                        'title': article.get('title', ''),
                        'category': article.get('category', ''),
                        'saved_at': article.get('saved_at', ''),
                        'keywords': article.get('summary', {}).get('keywords', [])
                    }
        
        index_path = os.path.join(self.base_path, 'index', 'articles.json')
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    
    def get_articles(self, category: str = None, limit: int = 100) -> List[Dict]:
        """获取文章列表"""
        articles = []
        
        if category:
            category_path = os.path.join(
                self.base_path, 
                'by_category',
                f"{category}.jsonl"
            )
            if os.path.exists(category_path):
                with open(category_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            articles.append(json.loads(line))
        
        # 如果没有指定分类，获取所有
        if not category:
            articles_dir = os.path.join(self.base_path, 'articles')
            for filename in os.listdir(articles_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(articles_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        articles.append(json.load(f))
        
        return articles[-limit:]
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'total_articles': 0,
            'by_category': {},
            'last_updated': None
        }
        
        articles_dir = os.path.join(self.base_path, 'articles')
        if not os.path.exists(articles_dir):
            return stats
        
        for filename in os.listdir(articles_dir):
            if filename.endswith('.json'):
                stats['total_articles'] += 1
        
        # 按分类统计
        by_category_dir = os.path.join(self.base_path, 'by_category')
        if os.path.exists(by_category_dir):
            for filename in os.listdir(by_category_dir):
                if filename.endswith('.jsonl'):
                    category = filename.replace('.jsonl', '')
                    filepath = os.path.join(by_category_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        stats['by_category'][category] = len(lines)
        
        # 获取最后更新时间
        index_path = os.path.join(self.base_path, 'index', 'articles.json')
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                index = json.load(f)
                stats['last_updated'] = index.get('last_updated')
        
        return stats
    
    def search_articles(self, query: str) -> List[Dict]:
        """搜索文章"""
        results = []
        
        for article in self.get_articles():
            # 检查标题和内容
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            summary = article.get('summary', {}).get('summary', '').lower()
            
            if query.lower() in title or query.lower() in content or query.lower() in summary:
                results.append(article)
        
        return results
