"""Unit tests for QueryProcessor.

Tests cover:
- Basic keyword extraction
- Chinese and English stopword filtering
- Filter syntax parsing (collection:xxx, type:xxx)
- Edge cases (empty query, special characters)
- Configuration options
"""

import pytest
from src.core.query_engine.query_processor import (
    QueryProcessor,
    QueryProcessorConfig,
    create_query_processor,
    DEFAULT_STOPWORDS,
    CHINESE_STOPWORDS,
    ENGLISH_STOPWORDS,
)
from src.core.types import ProcessedQuery


class TestQueryProcessorBasic:
    """Test basic QueryProcessor functionality."""
    
    def test_simple_english_query(self):
        """Test simple English query keyword extraction."""
        processor = QueryProcessor()
        result = processor.process("Azure OpenAI configuration")

        assert result.original_query == "Azure OpenAI configuration"
        # Query-side keywords are now stemmed+lowered (shared tokenizer).
        # Azure→azur, OpenAI→openai, configuration→configur
        assert "azur" in result.keywords
        assert "openai" in result.keywords
        assert "configur" in result.keywords
        assert isinstance(result.filters, dict)
    
    def test_simple_chinese_query(self):
        """Test simple Chinese query keyword extraction."""
        processor = QueryProcessor()
        result = processor.process("配置 Azure OpenAI")

        assert result.original_query == "配置 Azure OpenAI"
        assert "配置" in result.keywords
        # English keywords are stemmed+lowered: Azure→azur, OpenAI→openai
        assert "azur" in result.keywords
        assert "openai" in result.keywords
    
    def test_mixed_language_query(self):
        """Test mixed Chinese-English query."""
        processor = QueryProcessor()
        result = processor.process("如何配置 Azure OpenAI embedding 模型")

        # English keywords are stemmed+lowered
        assert "azur" in result.keywords
        assert "openai" in result.keywords
        assert "embed" in result.keywords
        assert "模型" in result.keywords
        # Keywords should be non-empty (acceptance criteria)
        assert len(result.keywords) > 0


class TestStopwordFiltering:
    """Test stopword filtering."""
    
    def test_chinese_stopwords_filtered(self):
        """Test that Chinese stopwords are filtered."""
        processor = QueryProcessor()
        # Use space-separated Chinese words for proper tokenization
        result = processor.process("如何 在 配置 系统")
        
        # Individual stopwords should be filtered
        assert "如何" not in result.keywords
        assert "在" not in result.keywords
        # Content words should remain
        assert "配置" in result.keywords
        assert "系统" in result.keywords
    
    def test_english_stopwords_filtered(self):
        """Test that English stopwords are filtered."""
        processor = QueryProcessor()
        result = processor.process("how to configure the Azure API")

        # Stopwords should be filtered
        assert "how" not in result.keywords
        assert "to" not in result.keywords
        assert "the" not in result.keywords
        # Content words remain, but stemmed+lowered: configure→configur, Azure→azur, API→api
        assert "configur" in result.keywords
        assert "azur" in result.keywords
        assert "api" in result.keywords
    
    def test_custom_stopwords(self):
        """Test custom stopwords configuration."""
        custom_stopwords = {"custom", "word"}
        config = QueryProcessorConfig(stopwords=custom_stopwords)
        processor = QueryProcessor(config)
        
        result = processor.process("custom word test")
        
        assert "custom" not in result.keywords
        assert "word" not in result.keywords
        assert "test" in result.keywords
    
    def test_add_stopwords(self):
        """Test adding stopwords dynamically."""
        processor = QueryProcessor()
        processor.add_stopwords({"newstop"})

        result = processor.process("newstop important")

        assert "newstop" not in result.keywords
        # important → import (Porter stem); _filter_keywords checks stem form
        assert "import" in result.keywords
    
    def test_remove_stopwords(self):
        """Test removing stopwords dynamically."""
        processor = QueryProcessor()
        processor.remove_stopwords({"如何"})
        
        # Use space-separated input for proper tokenization
        result = processor.process("如何 配置")
        
        assert "如何" in result.keywords
        assert "配置" in result.keywords


class TestFilterParsing:
    """Test filter syntax parsing."""
    
    def test_collection_filter(self):
        """Test collection filter parsing."""
        processor = QueryProcessor()
        result = processor.process("collection:api-docs Azure configuration")

        assert result.filters.get("collection") == "api-docs"
        # English keywords stemmed+lowered: Azure→azur, configuration→configur
        assert "azur" in result.keywords
        assert "configur" in result.keywords
        # Filter syntax should not appear in keywords
        assert "collection" not in result.keywords
        assert "api-docs" not in result.keywords
    
    def test_collection_short_syntax(self):
        """Test collection short syntax (col:)."""
        processor = QueryProcessor()
        result = processor.process("col:docs Azure")

        assert result.filters.get("collection") == "docs"
        # English keywords stemmed+lowered: Azure→azur
        assert "azur" in result.keywords
    
    def test_type_filter(self):
        """Test doc_type filter parsing."""
        processor = QueryProcessor()
        result = processor.process("type:pdf search query")

        assert result.filters.get("doc_type") == "pdf"
        assert "search" in result.keywords
        # query→queri (Porter stem), stemmed+lowered
        assert "queri" in result.keywords
    
    def test_source_filter(self):
        """Test source path filter parsing."""
        processor = QueryProcessor()
        result = processor.process("source:readme.md content")
        
        assert result.filters.get("source_path") == "readme.md"
        assert "content" in result.keywords
    
    def test_tag_filter(self):
        """Test tag filter parsing."""
        processor = QueryProcessor()
        result = processor.process("tag:important,urgent search")
        
        assert "tags" in result.filters
        assert "important" in result.filters["tags"]
        assert "urgent" in result.filters["tags"]
    
    def test_multiple_filters(self):
        """Test multiple filters in one query."""
        processor = QueryProcessor()
        result = processor.process("collection:docs type:pdf Azure configuration")

        assert result.filters.get("collection") == "docs"
        assert result.filters.get("doc_type") == "pdf"
        # English keywords stemmed+lowered: Azure→azur, configuration→configur
        assert "azur" in result.keywords
        assert "configur" in result.keywords
    
    def test_generic_filter(self):
        """N4: 未识别 key:value 不当 filter，当普通查询文本回到 keywords。"""
        processor = QueryProcessor()
        result = processor.process("custom_field:custom_value search")

        # 未识别 key 不再产生 generic filter
        assert "custom_field" not in result.filters
        assert result.filters == {}
        # search 仍是关键词
        assert "search" in result.keywords
    
    def test_disable_filter_parsing(self):
        """Test disabling filter parsing."""
        config = QueryProcessorConfig(enable_filter_parsing=False)
        processor = QueryProcessor(config)

        result = processor.process("collection:docs Azure")

        assert len(result.filters) == 0
        # collection:docs treated as text; stemmed: collection→collect, docs→doc
        assert "collect" in result.keywords or "doc" in result.keywords


class TestFilterAllowlistN4:
    """N4: filter 解析收口——自然语言 word:value 不再被误解析为 filter。"""

    def test_natural_language_colon_not_filter(self):
        """自然语言冒号（时间 / URL / 中文）不当 filter。"""
        processor = QueryProcessor()

        r = processor.process("会议 12:30 下午")
        assert r.filters == {}
        assert "会议" in r.keywords

        r = processor.process("官网 https://example.com 地址")
        assert r.filters == {}
        assert "官网" in r.keywords

        r = processor.process("Azure:服务端 配置")
        assert r.filters == {}
        # Azure→azur after stemming
        assert "azur" in r.keywords or "服务端" in r.keywords

    def test_windows_path_not_filter(self):
        """Windows 路径 c:\\... 不当 filter（c 已不是别名）。"""
        processor = QueryProcessor()
        r = processor.process(r"路径 c:\Users\test 文档")
        assert r.filters == {}
        assert "路径" in r.keywords or "文档" in r.keywords

    def test_single_letter_alias_disabled(self):
        """单字母别名 c/s/t 已删除，不再当 filter。"""
        processor = QueryProcessor()
        for q in ("c:docs", "s:readme", "t:pdf"):
            r = processor.process(q)
            assert r.filters == {}, f"{q} 不应产生 filter"

    def test_multiletter_alias_still_works(self):
        """多字母别名 col/src 仍工作。"""
        processor = QueryProcessor()

        r = processor.process("col:docs Azure")
        assert r.filters.get("collection") == "docs"
        # Azure→azur after stemming
        assert "azur" in r.keywords

        r = processor.process("src:readme.md content")
        assert r.filters.get("source_path") == "readme.md"

    def test_whitelist_filters_still_parsed(self):
        """白名单 4 类回归保护（全名语法）。"""
        processor = QueryProcessor()
        r = processor.process("collection:docs type:pdf source:r.md tag:a,b")
        assert r.filters.get("collection") == "docs"
        assert r.filters.get("doc_type") == "pdf"
        assert r.filters.get("source_path") == "r.md"
        assert "a" in r.filters["tags"]
        assert "b" in r.filters["tags"]


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_query(self):
        """Test empty query handling."""
        processor = QueryProcessor()
        result = processor.process("")
        
        assert result.original_query == ""
        assert result.keywords == []
        assert result.filters == {}
    
    def test_none_query(self):
        """Test None query handling."""
        processor = QueryProcessor()
        result = processor.process(None)
        
        assert result.original_query == ""
        assert result.keywords == []
        assert result.filters == {}
    
    def test_whitespace_only_query(self):
        """Test whitespace-only query."""
        processor = QueryProcessor()
        result = processor.process("   \t\n  ")
        
        assert result.keywords == []
    
    def test_special_characters(self):
        """Test query with special characters."""
        processor = QueryProcessor()
        result = processor.process("Azure-OpenAI API_key configuration")

        # Hyphenated and underscored words split by shared tokenizer;
        # English words are stemmed+lowered.
        assert any("azur" in kw or "openai" in kw for kw in result.keywords)
        assert any("api" in kw or "key" in kw for kw in result.keywords)
        # configuration→configur (stemmed)
        assert "configur" in result.keywords
    
    def test_numbers_in_query(self):
        """Test query with numbers."""
        processor = QueryProcessor()
        result = processor.process("GPT4 text-embedding-3-small")

        # GPT4 is mixed alphanumeric → not stemmed, but lowercased → gpt4
        assert "gpt4" in result.keywords
        # Handle hyphenated model names; embedding→embed (stemmed)
        assert any("embed" in kw for kw in result.keywords)
    
    def test_duplicate_keywords(self):
        """Test duplicate keyword handling."""
        processor = QueryProcessor()
        result = processor.process("Azure Azure azure AZURE")

        # Should deduplicate; shared tokenizer lowercases + deduplicates,
        # so all variants collapse to single "azur" (stemmed).
        azure_count = sum(1 for kw in result.keywords if kw == "azur")
        assert azure_count == 1
    
    def test_very_long_query(self):
        """Test very long query with max_keywords limit."""
        config = QueryProcessorConfig(max_keywords=5)
        processor = QueryProcessor(config)
        
        # Query with many keywords
        query = " ".join([f"keyword{i}" for i in range(20)])
        result = processor.process(query)
        
        assert len(result.keywords) <= 5
    
    def test_min_keyword_length(self):
        """Test minimum keyword length constraint."""
        config = QueryProcessorConfig(min_keyword_length=3)
        processor = QueryProcessor(config)
        
        result = processor.process("a ab abc abcd")
        
        assert "a" not in result.keywords
        assert "ab" not in result.keywords
        assert "abc" in result.keywords
        assert "abcd" in result.keywords


class TestProcessedQueryContract:
    """Test ProcessedQuery data contract."""
    
    def test_processed_query_structure(self):
        """Test ProcessedQuery has expected structure."""
        processor = QueryProcessor()
        result = processor.process("test query")
        
        assert isinstance(result, ProcessedQuery)
        assert hasattr(result, "original_query")
        assert hasattr(result, "keywords")
        assert hasattr(result, "filters")
        assert hasattr(result, "expanded_terms")
    
    def test_processed_query_serialization(self):
        """Test ProcessedQuery can be serialized to dict."""
        processor = QueryProcessor()
        result = processor.process("collection:docs Azure test")

        data = result.to_dict()

        assert isinstance(data, dict)
        assert data["original_query"] == "collection:docs Azure test"
        # Azure→azur (stemmed+lowered)
        assert "azur" in data["keywords"]
        assert "test" in data["keywords"]
        assert data["filters"]["collection"] == "docs"
    
    def test_processed_query_from_dict(self):
        """Test ProcessedQuery can be created from dict."""
        data = {
            "original_query": "test query",
            "keywords": ["test", "query"],
            "filters": {"collection": "docs"},
            "expanded_terms": []
        }
        
        result = ProcessedQuery.from_dict(data)
        
        assert result.original_query == "test query"
        assert result.keywords == ["test", "query"]
        assert result.filters == {"collection": "docs"}


class TestFactoryFunction:
    """Test create_query_processor factory function."""
    
    def test_default_factory(self):
        """Test factory with default settings."""
        processor = create_query_processor()
        result = processor.process("test Azure")

        assert isinstance(processor, QueryProcessor)
        # Azure→azur (stemmed+lowered)
        assert "azur" in result.keywords
    
    def test_factory_with_custom_stopwords(self):
        """Test factory with custom stopwords."""
        processor = create_query_processor(stopwords={"custom"})
        result = processor.process("custom test")
        
        assert "custom" not in result.keywords
        assert "test" in result.keywords
        # Default stopwords should not apply
        assert "how" not in processor.config.stopwords
    
    def test_factory_with_min_length(self):
        """Test factory with custom min_keyword_length."""
        processor = create_query_processor(min_keyword_length=4)
        result = processor.process("a ab abc abcd abcde")
        
        assert "a" not in result.keywords
        assert "abc" not in result.keywords
        assert "abcd" in result.keywords
    
    def test_factory_with_max_keywords(self):
        """Test factory with custom max_keywords."""
        processor = create_query_processor(max_keywords=2)
        result = processor.process("one two three four five")
        
        assert len(result.keywords) <= 2
    
    def test_factory_disable_filters(self):
        """Test factory with filter parsing disabled."""
        processor = create_query_processor(enable_filter_parsing=False)
        result = processor.process("collection:docs test")
        
        assert len(result.filters) == 0


class TestChineseTextProcessing:
    """Test Chinese text processing specifics."""
    
    def test_chinese_only_query(self):
        """Test pure Chinese query."""
        processor = QueryProcessor()
        result = processor.process("向量数据库配置指南")
        
        assert len(result.keywords) > 0
        # Should extract meaningful Chinese words/phrases
        assert any("向量" in kw or "数据库" in kw or "配置" in kw for kw in result.keywords)
    
    def test_chinese_with_punctuation(self):
        """Test Chinese query with punctuation."""
        processor = QueryProcessor()
        # Use space-separated for proper tokenization
        result = processor.process("配置 问题 ？ 帮助 ！")
        
        # Punctuation should not affect extraction, keywords extracted
        assert "配置" in result.keywords
        assert "问题" in result.keywords
        # Keywords should be non-empty
        assert len(result.keywords) > 0


class TestKeywordsNonEmpty:
    """Test that keywords are non-empty for valid queries (acceptance criteria)."""

    def test_keywords_non_empty_english(self):
        """Test keywords non-empty for English query."""
        processor = QueryProcessor()
        result = processor.process("configure Azure API")

        # Per acceptance criteria: keywords should be non-empty
        assert len(result.keywords) > 0

    def test_keywords_non_empty_chinese(self):
        """Test keywords non-empty for Chinese query."""
        processor = QueryProcessor()
        result = processor.process("配置数据库")

        assert len(result.keywords) > 0

    def test_keywords_non_empty_mixed(self):
        """Test keywords non-empty for mixed query."""
        processor = QueryProcessor()
        result = processor.process("Azure 配置")

        assert len(result.keywords) > 0

    def test_filters_is_dict(self):
        """Test filters is always a dict (acceptance criteria)."""
        processor = QueryProcessor()

        # Without filters
        result1 = processor.process("simple query")
        assert isinstance(result1.filters, dict)

        # With filters
        result2 = processor.process("collection:docs query")
        assert isinstance(result2.filters, dict)


class TestCrossLayerTokenization:
    """Cross-layer consistency: query tokens must be a subset of index terms."""

    def test_stem_mutated_stopwords_filtered(self):
        """Stopwords whose Porter stem differs from surface form must still be filtered."""
        processor = QueryProcessor()
        result = processor.process("the optimization because these networks")
        keywords_lower = [k.lower() for k in result.keywords]
        assert "becaus" not in keywords_lower  # because→becaus would leak without pre-stem filter
        assert "thes" not in keywords_lower    # these→thes
        assert "optim" in keywords_lower       # content word kept (stemmed)

    def test_query_and_index_tokenization_match(self):
        """同一文本,query 侧 tokenized 后的关键词 ⊆ 索引侧 term。
        这是 BM25 召回的充要条件(spec 纪律级约束)。"""
        from src.ingestion.embedding.sparse_encoder import SparseEncoder
        from src.core.types import Chunk

        text = "the optimization of neural networks"
        encoder = SparseEncoder()
        stats = encoder.encode([Chunk(id="t", text=text, metadata={"source_path": "dummy"})])[0]
        index_terms = set(stats["term_frequencies"])

        processor = QueryProcessor()
        result = processor.process(text)
        query_terms = set(result.keywords) | {k.lower() for k in result.keywords}

        # 查询侧关键词必须在索引侧 term 集合里
        missing = query_terms - index_terms
        # 允许 query 侧 _filter_keywords 因 max_keywords 截断,但不能出现"索引没有"的词
        assert not missing, f"query keywords not in index terms: {missing}"

    def test_query_side_stems_english(self):
        """query 侧英文也应 stem,否则查 optimization 召不回索引里的 optim 词干。"""
        from src.ingestion.embedding.sparse_encoder import SparseEncoder
        from src.core.types import Chunk

        processor = QueryProcessor()
        result = processor.process("optimization")
        # 经 stem 后的形态应能与索引侧一致(具体词干由 Porter 决定)
        encoder = SparseEncoder()
        idx = set(encoder.encode([Chunk(id="t", text="optimization", metadata={"source_path": "dummy"})])[0]["term_frequencies"])
        assert set(k.lower() for k in result.keywords) & idx, "query stem != index stem"

    def test_no_zero_recall_single_char_chinese(self):
        """V3: query 侧 min_term_length=2 (对齐 index),不再产生 index 从不存储的
        单字 term。'猫吃鱼' jieba 分成三个单字,旧 query(min_term_length=1)会把它们
        当关键词,但 index(min_term_length=2)全丢 → 零召回噪声。"""
        processor = QueryProcessor()
        result = processor.process("猫吃鱼")
        # 单字不应作为关键词(index 侧不会存)
        for kw in result.keywords:
            assert len(kw) >= 2, f"single-char keyword leaked (zero-recall): {kw}"
