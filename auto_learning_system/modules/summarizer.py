"""
摘要模块
负责对文章内容进行摘要和关键词提取
"""

import logging
from typing import Dict, List
from collections import Counter
import re


class Summarizer:
    """内容摘要器"""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.max_content_length = config.get('summarization', {}).get(
            'max_content_length', 2000
        )
        self.max_summary_length = config.get('summarization', {}).get(
            'max_summary_length', 300
        )
        self.important_keywords_count = config.get('summarization', {}).get(
            'important_keywords_count', 5
        )
    
    def summarize(self, article: Dict) -> Dict:
        """对文章进行摘要"""
        content = article.get('content', '') or article.get('description', '')
        title = article.get('title', '')
        
        # 截取内容
        content = content[:self.max_content_length]
        
        # 提取关键词
        keywords = self._extract_keywords(content)
        
        # 生成摘要
        summary = self._generate_summary(content, title)
        
        # 提取重要句子
        important_sentences = self._extract_important_sentences(content)
        
        return {
            'summary': summary,
            'keywords': keywords,
            'important_points': important_sentences,
            'word_count': len(content.split()),
            'content_preview': content[:500]
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 停用词
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'it', 'its', 'as', 'in', 'on', 'what', 'which', 'who',
            'how', 'when', 'where', 'why', 'all', 'each', 'every', 'both',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also'
        }
        
        # 清洗文本
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # 过滤停用词和短词
        words = [w for w in words if w not in stopwords and len(w) > 2]
        
        # 词频统计
        word_counts = Counter(words)
        
        # 获取高频词
        top_words = word_counts.most_common(self.important_keywords_count * 2)
        
        # 选择更有意义的词
        keywords = []
        for word, count in top_words:
            if len(keywords) >= self.important_keywords_count:
                break
            # 排除纯数字
            if not word.isdigit():
                keywords.append(word)
        
        return keywords
    
    def _generate_summary(self, content: str, title: str) -> str:
        """生成摘要"""
        if not content:
            return title
        
        # 简单摘要：取前几句
        sentences = content.split('. ')
        summary_parts = []
        
        for sentence in sentences:
            if len(' '.join(summary_parts)) < self.max_summary_length:
                summary_parts.append(sentence.strip())
            else:
                break
        
        summary = '. '.join(summary_parts)
        
        # 如果摘要太短，使用标题
        if len(summary) < 50:
            summary = title
        
        # 截断到最大长度
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length - 3] + '...'
        
        return summary
    
    def _extract_important_sentences(self, content: str) -> List[str]:
        """提取重要句子"""
        sentences = content.split('. ')
        important = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 跳过太短或太长的句子
            if len(sentence) < 20 or len(sentence) > 200:
                continue
            
            # 检查是否包含关键词
            has_keyword = any(
                kw in sentence.lower() 
                for kw in ['ai', 'machine learning', 'neural', 'algorithm', 
                          'model', 'data', 'training', 'inference']
            )
            
            if has_keyword:
                important.append(sentence)
        
        return important[:5]  # 最多返回5个重要句子
