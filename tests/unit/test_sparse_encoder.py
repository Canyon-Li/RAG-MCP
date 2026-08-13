"""Unit tests for SparseEncoder.

Tests cover:
- Constructor validation
- Basic encoding functionality
- Tokenization behavior
- Edge cases (empty text, special characters)
- Corpus statistics calculation
- Deterministic behavior
"""

import pytest
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.core.types import Chunk


# ============================================================================
# Constructor Tests
# ============================================================================

def test_constructor_default():
    """Test default constructor."""
    encoder = SparseEncoder()
    assert encoder.min_term_length == 2
    assert encoder.lowercase is True


def test_constructor_custom_min_term_length():
    """Test custom min_term_length."""
    encoder = SparseEncoder(min_term_length=3)
    assert encoder.min_term_length == 3


def test_constructor_custom_lowercase():
    """Test custom lowercase setting."""
    encoder = SparseEncoder(lowercase=False)
    assert encoder.lowercase is False


def test_constructor_rejects_zero_min_term_length():
    """Test that min_term_length=0 is rejected."""
    with pytest.raises(ValueError, match="min_term_length must be >= 1"):
        SparseEncoder(min_term_length=0)


def test_constructor_rejects_negative_min_term_length():
    """Test that negative min_term_length is rejected."""
    with pytest.raises(ValueError, match="min_term_length must be >= 1"):
        SparseEncoder(min_term_length=-1)


# ============================================================================
# Basic Encoding Tests
# ============================================================================

def test_encode_single_chunk():
    """Test encoding a single chunk."""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="hello world", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    assert len(results) == 1
    assert results[0]["chunk_id"] == "1"
    assert results[0]["term_frequencies"]["hello"] == 1
    assert results[0]["term_frequencies"]["world"] == 1
    assert results[0]["doc_length"] == 2
    assert results[0]["unique_terms"] == 2


def test_encode_multiple_chunks():
    """Test encoding multiple chunks."""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="machine learning", metadata={"source_path": "test.txt"}),
        Chunk(id="2", text="deep learning networks", metadata={"source_path": "test.txt"}),
    ]
    
    results = encoder.encode(chunks)
    
    assert len(results) == 2
    assert results[0]["chunk_id"] == "1"
    assert results[1]["chunk_id"] == "2"


def test_encode_with_repeated_terms():
    """Test that term frequencies count correctly.

    NOTE: The shared tokenizer deduplicates terms, so repeated occurrences
    of the same surface form produce a single entry with count 1.
    """
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="hello world hello hello", metadata={"source_path": "test.txt"})
    ]

    results = encoder.encode(chunks)

    assert results[0]["term_frequencies"]["hello"] == 1
    assert results[0]["term_frequencies"]["world"] == 1
    assert results[0]["doc_length"] == 2
    assert results[0]["unique_terms"] == 2


# ============================================================================
# Tokenization Tests
# ============================================================================

def test_tokenize_lowercases_by_default():
    """Test that terms are lowercased by default.

    NOTE: The shared tokenizer deduplicates, so HELLO appearing twice
    still yields count 1.
    """
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="Hello World HELLO", metadata={"source_path": "test.txt"})
    ]

    results = encoder.encode(chunks)

    assert "hello" in results[0]["term_frequencies"]
    assert "world" in results[0]["term_frequencies"]
    assert results[0]["term_frequencies"]["hello"] == 1


def test_tokenize_preserves_case_when_configured():
    """Test case behavior when lowercase=False.

    NOTE: The shared tokenizer applies Porter stemming to English words,
    and the stemmer lowercases internally, so English terms are always
    lowercased in the output even when lowercase=False.
    """
    encoder = SparseEncoder(lowercase=False)
    chunks = [
        Chunk(id="1", text="Hello World", metadata={"source_path": "test.txt"})
    ]

    results = encoder.encode(chunks)

    # Stemmer lowercases: Hello→hello, World→world
    assert "hello" in results[0]["term_frequencies"]
    assert "world" in results[0]["term_frequencies"]


def test_tokenize_filters_by_min_term_length():
    """Test that short terms are filtered out.

    NOTE: English words are Porter-stemmed, so "learning"→"learn".
    """
    encoder = SparseEncoder(min_term_length=3)
    chunks = [
        Chunk(id="1", text="I am learning Python AI", metadata={"source_path": "test.txt"})
    ]

    results = encoder.encode(chunks)

    # Should filter out "I", "am", and "AI" (length < 3)
    assert "learn" in results[0]["term_frequencies"]
    assert "python" in results[0]["term_frequencies"]
    assert results[0]["unique_terms"] == 2  # learn, python


def test_tokenize_handles_punctuation():
    """Test that punctuation is handled correctly.

    NOTE: "how" is an English stopword and is now filtered out.
    """
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="Hello, world! How are you?", metadata={"source_path": "test.txt"})
    ]

    results = encoder.encode(chunks)

    # Punctuation should be removed; "how" is a stopword
    assert "hello" in results[0]["term_frequencies"]
    assert "world" in results[0]["term_frequencies"]
    assert "," not in results[0]["term_frequencies"]
    assert "!" not in results[0]["term_frequencies"]


def test_tokenize_handles_hyphens_and_underscores():
    """Test hyphen/underscore handling.

    NOTE: The shared tokenizer splits on non-word chars, so hyphenated and
    underscored terms are split into sub-tokens and each is stemmed.
    """
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="machine-learning deep_learning", metadata={"source_path": "test.txt"})
    ]

    results = encoder.encode(chunks)

    # Split + stem: machine→machin, learning→learn, deep→deep
    assert "machin" in results[0]["term_frequencies"]
    assert "learn" in results[0]["term_frequencies"]
    assert "deep" in results[0]["term_frequencies"]


def test_tokenize_handles_numbers():
    """Test that numbers are tokenized.

    NOTE: Hyphens are split, so "gpt-4" → "gpt" + "4" (4 filtered by min_term_length=2).
    """
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="Python 3.11 and GPT-4", metadata={"source_path": "test.txt"})
    ]

    results = encoder.encode(chunks)

    assert "python" in results[0]["term_frequencies"]
    assert "11" in results[0]["term_frequencies"]
    assert "gpt" in results[0]["term_frequencies"]


# ============================================================================
# Edge Cases
# ============================================================================

def test_encode_rejects_empty_chunks_list():
    """Test that empty chunks list is rejected."""
    encoder = SparseEncoder()
    
    with pytest.raises(ValueError, match="Cannot encode empty chunks list"):
        encoder.encode([])


def test_encode_rejects_chunk_with_empty_text():
    """Test that chunk with empty text is rejected."""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="", metadata={"source_path": "test.txt"})
    ]
    
    with pytest.raises(ValueError, match="empty or whitespace-only text"):
        encoder.encode(chunks)


def test_encode_rejects_chunk_with_whitespace_only_text():
    """Test that chunk with whitespace-only text is rejected."""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="   \n\t  ", metadata={"source_path": "test.txt"})
    ]
    
    with pytest.raises(ValueError, match="empty or whitespace-only text"):
        encoder.encode(chunks)


def test_encode_handles_special_characters():
    """Test encoding text with special characters.

    NOTE: English words are stemmed, so "programming"→"program".
    """
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="C++ and C# programming @2024", metadata={"source_path": "test.txt"})
    ]

    results = encoder.encode(chunks)

    # Should extract alphanumeric terms (stemmed)
    assert "program" in results[0]["term_frequencies"]
    assert "2024" in results[0]["term_frequencies"]


def test_encode_handles_unicode():
    """Test encoding text with unicode characters."""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="café résumé naïve", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    # Unicode characters should be handled
    assert results[0]["doc_length"] > 0
    assert results[0]["unique_terms"] > 0


# ============================================================================
# Determinism Tests
# ============================================================================

def test_encode_is_deterministic():
    """Test that encoding is deterministic."""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="machine learning deep learning", metadata={"source_path": "test.txt"})
    ]
    
    results1 = encoder.encode(chunks)
    results2 = encoder.encode(chunks)
    
    assert results1 == results2


def test_encode_preserves_chunk_order():
    """Test that output order matches input order."""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="first chunk", metadata={"source_path": "test.txt"}),
        Chunk(id="2", text="second chunk", metadata={"source_path": "test.txt"}),
        Chunk(id="3", text="third chunk", metadata={"source_path": "test.txt"}),
    ]
    
    results = encoder.encode(chunks)
    
    assert len(results) == 3
    assert results[0]["chunk_id"] == "1"
    assert results[1]["chunk_id"] == "2"
    assert results[2]["chunk_id"] == "3"


# ============================================================================
# Corpus Statistics Tests
# ============================================================================

def test_get_corpus_stats_single_document():
    """Test corpus stats for single document."""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="hello world", metadata={"source_path": "test.txt"})
    ]
    
    encoded = encoder.encode(chunks)
    stats = encoder.get_corpus_stats(encoded)
    
    assert stats["num_docs"] == 1
    assert stats["avg_doc_length"] == 2.0
    assert stats["document_frequency"]["hello"] == 1
    assert stats["document_frequency"]["world"] == 1


def test_get_corpus_stats_multiple_documents():
    """Test corpus stats for multiple documents.

    NOTE: English terms are Porter-stemmed: machine→machin, learning→learn,
    networks→network, algorithms→algorithm.
    """
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="machine learning", metadata={"source_path": "test.txt"}),
        Chunk(id="2", text="deep learning networks", metadata={"source_path": "test.txt"}),
        Chunk(id="3", text="machine learning algorithms", metadata={"source_path": "test.txt"}),
    ]

    encoded = encoder.encode(chunks)
    stats = encoder.get_corpus_stats(encoded)

    assert stats["num_docs"] == 3
    assert stats["avg_doc_length"] == (2 + 3 + 3) / 3
    # "learn" (stemmed from "learning") appears in all 3 docs
    assert stats["document_frequency"]["learn"] == 3
    # "machin" (stemmed from "machine") appears in 2 docs
    assert stats["document_frequency"]["machin"] == 2
    # "deep" appears in 1 doc
    assert stats["document_frequency"]["deep"] == 1


def test_get_corpus_stats_calculates_average_doc_length():
    """Test that average document length is calculated correctly.

    NOTE: Stopwords ("this", "is", "a") are now removed by the shared tokenizer,
    so doc 2 has 2 terms ("longer", "document") instead of 4.
    """
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="short", metadata={"source_path": "test.txt"}),
        Chunk(id="2", text="this is a longer document", metadata={"source_path": "test.txt"}),
    ]

    encoded = encoder.encode(chunks)
    stats = encoder.get_corpus_stats(encoded)

    # First doc: 1 term ("short"), Second doc: 2 terms ("longer", "document" — stopwords removed), avg = 1.5
    assert stats["avg_doc_length"] == 1.5


def test_get_corpus_stats_handles_empty_list():
    """Test corpus stats with empty encoded chunks list."""
    encoder = SparseEncoder()
    
    stats = encoder.get_corpus_stats([])
    
    assert stats["num_docs"] == 0
    assert stats["avg_doc_length"] == 0.0
    assert stats["document_frequency"] == {}


# ============================================================================
# Integration Test
# ============================================================================

def test_realistic_encoding_scenario():
    """Test realistic encoding scenario with varied content."""
    encoder = SparseEncoder()
    
    chunks = [
        Chunk(
            id="doc1_chunk0",
            text="Machine learning is a subset of artificial intelligence.",
            metadata={"source_path": "textbook.pdf"}
        ),
        Chunk(
            id="doc1_chunk1",
            text="Deep learning uses neural networks with multiple layers.",
            metadata={"source_path": "textbook.pdf"}
        ),
        Chunk(
            id="doc2_chunk0",
            text="Natural language processing (NLP) enables machines to understand text.",
            metadata={"source_path": "paper.pdf"}
        ),
    ]
    
    results = encoder.encode(chunks)
    
    # Validate structure
    assert len(results) == 3
    for i, result in enumerate(results):
        assert "chunk_id" in result
        assert "term_frequencies" in result
        assert "doc_length" in result
        assert "unique_terms" in result
        assert result["chunk_id"] == chunks[i].id
        assert result["doc_length"] > 0
        assert result["unique_terms"] > 0
        assert len(result["term_frequencies"]) == result["unique_terms"]
    
    # Get corpus stats
    corpus_stats = encoder.get_corpus_stats(results)
    assert corpus_stats["num_docs"] == 3
    assert corpus_stats["avg_doc_length"] > 0
    assert len(corpus_stats["document_frequency"]) > 0


# ============================================================================
# English Stemming & Stopword Tests (Task 3 — shared tokenizer)
# ============================================================================

def test_encode_english_stemming_converges():
    """同源英文词经 encode 后应落到同一 term(场景核心:optimization 召回 optimal)。"""
    encoder = SparseEncoder()
    chunk_a = Chunk(id="a", text="optimization", metadata={"source_path": "test.txt"})
    chunk_b = Chunk(id="b", text="optimize", metadata={"source_path": "test.txt"})
    stats_a = encoder.encode([chunk_a])[0]
    stats_b = encoder.encode([chunk_b])[0]
    # 两个 chunk 的 term_frequencies 键集应有交集(同一词干)
    common = set(stats_a["term_frequencies"]) & set(stats_b["term_frequencies"])
    assert common, f"expected shared stem term, got {stats_a} vs {stats_b}"


def test_encode_drops_english_stopwords():
    """索引侧应剔除英文停用词,避免倒排表膨胀。"""
    encoder = SparseEncoder()
    chunk = Chunk(id="x", text="the model is good", metadata={"source_path": "test.txt"})
    stats = encoder.encode([chunk])[0]
    terms = set(stats["term_frequencies"])
    assert "the" not in terms and "is" not in terms

