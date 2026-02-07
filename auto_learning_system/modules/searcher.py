"""
搜索模块
负责从网络搜索AI/技术新知识
"""

import logging
from typing import List, Dict
from datetime import datetime
import re


class Searcher:
    """搜索引擎"""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.keywords = config.get('search_keywords', [])
        self.sources = config.get('search_sources', [])
    
    def search_all(self) -> List[Dict]:
        """执行所有搜索"""
        all_articles = []
        
        for keyword in self.keywords:
            try:
                articles = self._search_keyword(keyword)
                all_articles.extend(articles)
                self.logger.info(f"关键词 '{keyword}' 搜索到 {len(articles)} 篇")
            except Exception as e:
                self.logger.warning(f"搜索 '{keyword}' 失败: {e}")
        
        # 去重
        all_articles = self._deduplicate(all_articles)
        
        return all_articles
    
    def _search_keyword(self, keyword: str) -> List[Dict]:
        """搜索单个关键词"""
        # 使用 web_search 工具
        try:
            from tools import web_search
            
            results = web_search(
                query=keyword,
                count=10,
                freshness='pw',  # 过去一周
                search_lang='en'
            )
            
            articles = []
            for item in results.get('results', []):
                article = {
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'description': item.get('description', ''),
                    'source': 'web_search',
                    'keyword': keyword,
                    'found_at': datetime.now().isoformat(),
                    'published_at': item.get('date', '')
                }
                
                # 获取详细内容
                if article['url']:
                    try:
                        from tools import web_fetch
                        content = web_fetch(
                            url=article['url'],
                            extractMode='markdown',
                            maxChars=5000
                        )
                        article['content'] = content
                    except Exception as e:
                        self.logger.debug(f"获取内容失败: {e}")
                        article['content'] = article.get('description', '')
                
                articles.append(article)
            
            return articles
            
        except ImportError:
            self.logger.warning("web_search 工具不可用，使用模拟数据")
            return self._mock_search(keyword)
    
    def _mock_search(self, keyword: str) -> List[Dict]:
        """模拟搜索结果（用于测试）"""
        return [
            {
                'title': f'{keyword} - 最新技术发展',
                'url': f'https://example.com/{keyword.lower().replace(" ", "-")}',
                'description': f'关于 {keyword} 的最新发展和研究进展',
                'content': f'这是关于 {keyword} 的详细内容...',
                'source': 'mock',
                'keyword': keyword,
                'found_at': datetime.now().isoformat()
            }
        ]
    
    def _deduplicate(self, articles: List[Dict]) -> List[Dict]:
        """去重"""
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        return unique_articles
