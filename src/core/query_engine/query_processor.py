"""Query Processor for preprocessing user queries.

This module provides query preprocessing functionality including:
- Keyword extraction using rule-based tokenization
- Stopword filtering for Chinese and English
- Filter parsing from query syntax (e.g., "collection:docs")
- Query normalization and cleaning

Design Principles:
- Rule-based first: Use simple, deterministic rules for reliability
- Language-aware: Support both Chinese and English queries
- Extensible: Easy to add synonym expansion or LLM-based processing later
- Configuration-driven: Stopwords and patterns configurable via settings
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Set

from src.core.text.tokenizer import tokenize as shared_tokenize
from src.core.types import ProcessedQuery


# Default stopwords for Chinese
CHINESE_STOPWORDS: Set[str] = {
    # 疑问词
    "如何", "怎么", "怎样", "什么", "哪个", "哪些", "为什么", "为何",
    "谁", "多少", "几", "是否", "能否", "可否",
    # 助词
    "的", "地", "得", "了", "着", "过", "吗", "呢", "吧", "啊", "呀",
    # 介词/连词
    "在", "于", "和", "与", "或", "及", "并", "而", "但", "但是",
    "因为", "所以", "如果", "那么", "虽然", "然而",
    # 代词
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "这", "那",
    "这个", "那个", "这些", "那些", "这里", "那里",
    # 副词
    "很", "非常", "特别", "更", "最", "都", "也", "还", "又", "再",
    "已", "已经", "正在", "将", "会", "能", "可以", "应该", "必须",
    # 动词(通用)
    "是", "有", "做", "进行", "使用", "通过",
    # 量词
    "个", "种", "类",
    # 标点等
    "？", "。", "！", "，", "、",
}

# Default stopwords for English
ENGLISH_STOPWORDS: Set[str] = {
    # Articles
    "a", "an", "the",
    # Prepositions
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "about", "through", "between", "after", "before",
    # Conjunctions
    "and", "or", "but", "if", "then", "because", "while", "although",
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they", "this", "that",
    "these", "those", "what", "which", "who", "whom", "whose",
    # Auxiliary verbs
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can",
    # Common verbs
    "get", "use", "make",
    # Question words
    "how", "why", "when", "where",
    # Others
    "not", "no", "yes", "so", "very", "just", "also", "too",
}

# Combined default stopwords
DEFAULT_STOPWORDS: Set[str] = CHINESE_STOPWORDS | ENGLISH_STOPWORDS

# Pattern for filter syntax: key:value
FILTER_PATTERN: Pattern = re.compile(r'(\w+):([^\s]+)')


@dataclass
class QueryProcessorConfig:
    """Configuration for QueryProcessor.
    
    Attributes:
        stopwords: Set of words to filter out
        min_keyword_length: Minimum length for a keyword to be included
        max_keywords: Maximum number of keywords to extract
        enable_filter_parsing: Whether to parse filter syntax from query
    """
    stopwords: Set[str] = field(default_factory=lambda: DEFAULT_STOPWORDS.copy())
    min_keyword_length: int = 1
    max_keywords: int = 20
    enable_filter_parsing: bool = True


class QueryProcessor:
    """Preprocesses user queries for retrieval.
    
    Extracts keywords, filters stopwords, and parses filter syntax
    to prepare queries for Dense and Sparse retrievers.
    
    Example:
        >>> processor = QueryProcessor()
        >>> result = processor.process("如何配置 Azure OpenAI？")
        >>> print(result.keywords)
        ['配置', 'Azure', 'OpenAI']
    """
    
    def __init__(self, config: Optional[QueryProcessorConfig] = None):
        """Initialize QueryProcessor.
        
        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or QueryProcessorConfig()
    
    def process(self, query: str) -> ProcessedQuery:
        """Process a user query into structured format.
        
        Args:
            query: Raw user query string
            
        Returns:
            ProcessedQuery with extracted keywords and filters
        """
        if not query or not query.strip():
            return ProcessedQuery(
                original_query=query or "",
                keywords=[],
                filters={}
            )
        
        # Normalize query
        normalized = self._normalize(query)
        
        # Extract filters from query (e.g., "collection:docs")
        filters, query_without_filters = self._extract_filters(normalized)
        
        # Tokenize and extract keywords
        tokens = self._tokenize(query_without_filters)
        
        # Filter stopwords and apply constraints
        keywords = self._filter_keywords(tokens)
        
        return ProcessedQuery(
            original_query=query,
            keywords=keywords,
            filters=filters
        )
    
    def _normalize(self, query: str) -> str:
        """Normalize query string.
        
        - Strip whitespace
        - Normalize unicode
        - Convert to consistent format
        
        Args:
            query: Raw query string
            
        Returns:
            Normalized query string
        """
        # Strip and normalize whitespace
        normalized = " ".join(query.split())
        return normalized
    
    def _extract_filters(self, query: str) -> tuple[Dict[str, Any], str]:
        """Extract filter syntax from query.

        只有白名单 key（collection/col、type/doc_type、source/src、tag/tags）
        产生 filter 并从 query 文本删除；未识别的 word:value 当普通查询文本
        原样保留（参与分词），不再走 generic filter 分支（N4 收口）。

        Args:
            query: Normalized query string

        Returns:
            Tuple of (filters dict, query without filter syntax)
        """
        if not self.config.enable_filter_parsing:
            return {}, query

        filters: Dict[str, Any] = {}
        kept: List[str] = []
        last_end = 0

        for m in FILTER_PATTERN.finditer(query):
            key, value = m.group(1), m.group(2)
            key_lower = key.lower()
            # 匹配段之前的文本原样保留
            kept.append(query[last_end:m.start()])

            if key_lower in ("collection", "col"):
                filters["collection"] = value
            elif key_lower in ("type", "doc_type"):
                filters["doc_type"] = value
            elif key_lower in ("source", "src"):
                filters["source_path"] = value
            elif key_lower in ("tag", "tags"):
                filters.setdefault("tags", []).extend(value.split(","))
            else:
                # 未识别 key：N4 收口，不当 filter，原样保留为查询文本
                kept.append(m.group(0))
            last_end = m.end()
        kept.append(query[last_end:])

        query_without_filters = " ".join("".join(kept).split())

        return filters, query_without_filters
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize query text via the shared tokenizer.

        Delegates to ``src.core.text.tokenizer.tokenize`` — the SAME function
        the index-side ``SparseEncoder`` uses — so query terms land on the
        exact stems stored in the BM25 inverted index. English is Porter-
        stemmed; stopwords are removed in ``_filter_keywords`` downstream
        (the shared tokenizer also receives them for defense-in-depth).

        Args:
            text: Query text (filters already stripped by caller)

        Returns:
            List of tokenized terms (stemmed for English, segmented for Chinese).
        """
        return shared_tokenize(
            text,
            lowercase=True,
            min_term_length=1,  # length filter applied in _filter_keywords
            stem_english=True,
            stopwords=frozenset(),  # stopword filtering kept in _filter_keywords
        )
    
    def _filter_keywords(self, tokens: List[str]) -> List[str]:
        """Filter tokens to get meaningful keywords.
        
        - Remove stopwords
        - Apply minimum length constraint
        - Deduplicate while preserving order
        - Apply maximum count limit
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of filtered keywords
        """
        seen: Set[str] = set()
        keywords: List[str] = []
        
        for token in tokens:
            # Normalize for comparison
            token_lower = token.lower()
            
            # Skip if already seen (case-insensitive dedup)
            if token_lower in seen:
                continue
            
            # Skip stopwords (check both original and lowercase)
            if token in self.config.stopwords or token_lower in self.config.stopwords:
                continue
            
            # Skip if too short
            if len(token) < self.config.min_keyword_length:
                continue
            
            # Add keyword (preserve original case)
            seen.add(token_lower)
            keywords.append(token)
            
            # Stop if we have enough
            if len(keywords) >= self.config.max_keywords:
                break
        
        return keywords
    
    def add_stopwords(self, words: Set[str]) -> None:
        """Add words to stopword set.
        
        Args:
            words: Set of words to add
        """
        self.config.stopwords.update(words)
    
    def remove_stopwords(self, words: Set[str]) -> None:
        """Remove words from stopword set.
        
        Args:
            words: Set of words to remove
        """
        self.config.stopwords -= words


def create_query_processor(
    stopwords: Optional[Set[str]] = None,
    min_keyword_length: int = 1,
    max_keywords: int = 20,
    enable_filter_parsing: bool = True
) -> QueryProcessor:
    """Factory function to create QueryProcessor.
    
    Args:
        stopwords: Custom stopwords set. Uses default if None.
        min_keyword_length: Minimum keyword length
        max_keywords: Maximum keywords to extract
        enable_filter_parsing: Whether to parse filter syntax
        
    Returns:
        Configured QueryProcessor instance
    """
    config = QueryProcessorConfig(
        stopwords=stopwords if stopwords is not None else DEFAULT_STOPWORDS.copy(),
        min_keyword_length=min_keyword_length,
        max_keywords=max_keywords,
        enable_filter_parsing=enable_filter_parsing
    )
    return QueryProcessor(config)
