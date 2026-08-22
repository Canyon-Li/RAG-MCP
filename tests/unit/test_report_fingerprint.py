"""Unit tests for report fingerprinting (D-031).

Fixture texts are adapted from a real 中信证券 first page (regulatorily
mandated disclosure elements) plus a distribution-platform watermark
variant — the exact scenario L1/L2 must handle.
"""

import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.fingerprint import (
    extract_report_fingerprint,
    hamming_distance,
    is_near_duplicate,
    simhash64,
)

# Adapted from the real 中信证券 润泽科技 first page (2023-06-12).
CITIC_PAGE = """
证券研究报告
请务必阅读正文之后第40 页起的免责条款和声明
园区级IDC 龙头，深度受益AI+数字经济
润泽科技（300442.SZ）投资价值分析报告｜2023.6.12
中信证券研究部
核心观点
黄亚元
通信行业首席
分析师
S1010520040001
公司是国内IDC 龙头。给予公司2023 年43 倍PE，对应目标价46 元，首次覆盖，给予“买入”评级。
评级 买入（首次） 当前价 33.56元 目标价 46.00元
"""

# Same report re-distributed by a platform: identical content, watermark
# footer stamped and re-saved (different file bytes entirely).
CITIC_PAGE_WATERMARKED = CITIC_PAGE + "\n用户691141754于2023-10-06日下载，仅供本人内部使用，不可传播与转载\n"

# A different broker's report on a different company (华泰-style).
HTSC_PAGE = """
华泰证券研究所
宁德时代（300750.SZ）：盈利能力行业领先
2024.11.03
分析师 王某某 S0570523080001
维持“增持”评级
"""

# Same broker + same security, DIFFERENT date → weekly-series sibling, must
# NOT be merged as a version of the 06-12 report.
CITIC_PAGE_NEXT_WEEK = CITIC_PAGE.replace("2023.6.12", "2023.6.19")


def test_extracts_all_elements_from_real_layout():
    fp = extract_report_fingerprint(CITIC_PAGE, filename="report.pdf")
    assert fp.cert_ids == ["S1010520040001"]
    assert fp.security_codes == ["300442.SZ"]
    assert fp.broker == "中信证券"
    assert fp.report_date == "2023-06-12"
    assert fp.confidence == "high"
    assert fp.business_key is not None


def test_watermarked_copy_gets_same_business_key():
    """The core scenario: same report, different distribution bytes."""
    fp_a = extract_report_fingerprint(CITIC_PAGE, filename="原版.pdf")
    fp_b = extract_report_fingerprint(CITIC_PAGE_WATERMARKED, filename="2023-10-06_平台分发版.pdf")
    assert fp_a.business_key == fp_b.business_key


def test_different_report_gets_different_key():
    fp_a = extract_report_fingerprint(CITIC_PAGE)
    fp_b = extract_report_fingerprint(HTSC_PAGE)
    assert fp_a.business_key != fp_b.business_key


def test_weekly_series_same_broker_same_code_not_merged():
    """Same title family, one week apart → distinct documents, not versions."""
    fp_a = extract_report_fingerprint(CITIC_PAGE)
    fp_c = extract_report_fingerprint(CITIC_PAGE_NEXT_WEEK)
    assert fp_a.business_key != fp_c.business_key


def test_filename_date_is_fallback_and_broker_from_filename_segment():
    fp = extract_report_fingerprint(
        "正文第一页，无任何日期与代码要素。",  # body lacking anchors
        filename="2023-06-12_中信证券_润泽科技(300442)投资价值分析报告.pdf",
    )
    assert fp.report_date == "2023-06-12"
    assert fp.broker == "中信证券"


def test_body_date_wins_over_download_renamed_filename():
    """Re-distributed copy renamed after its download date must keep the
    report's own publication date (body text is the authoritative anchor)."""
    fp = extract_report_fingerprint(
        CITIC_PAGE_WATERMARKED, filename="2023-10-06_某平台下载.pdf"
    )
    assert fp.report_date == "2023-06-12"


def test_low_confidence_when_elements_missing():
    fp = extract_report_fingerprint(
        "这是一份没有任何披露要素的普通文档，仅用于测试。", filename="notes.pdf"
    )
    assert fp.business_key is None
    assert fp.confidence == "low"


def test_medium_confidence_without_cert_id():
    """Codes + date present but no cert id → medium (series-level identity)."""
    fp = extract_report_fingerprint(
        "某某公司（600519.SH）点评\n发布日期：2025年3月1日", filename="d.pdf"
    )
    assert fp.confidence == "medium"
    assert fp.business_key is not None


# ── SimHash (L2) ────────────────────────────────────────────────────────────

BODY = (
    "公司专注超大规模园区级数据中心运营，总体规划56栋数据中心，29万个机柜。"
    "2019-2022年，公司收入CAGR超40%，扣非归母净利润约CAGR119%。"
    "终端客户覆盖字节、华为、京东、快手、美团等优质互联网客户。"
) * 20

BODY_WATERMARKED = BODY + "用户691141754于2023-10-06日下载，仅供本人内部使用，不可传播与转载。"

BODY_OTHER = (
    "公司主营业务为新能源汽车动力电池系统研发制造，全球市占率领先，"
    "与主要整车厂商建立长期合作关系，产能持续扩张，单位盈利保持稳定。"
) * 20


def test_simhash_identical_distance_zero():
    assert hamming_distance(simhash64(BODY), simhash64(BODY)) == 0


def test_simhash_watermarked_copy_is_near_duplicate():
    assert is_near_duplicate(simhash64(BODY), simhash64(BODY_WATERMARKED))


def test_simhash_different_document_is_not_near_duplicate():
    assert not is_near_duplicate(simhash64(BODY), simhash64(BODY_OTHER))


def test_simhash_is_deterministic_and_hex():
    h = simhash64(BODY)
    assert len(h) == 16 and int(h, 16) >= 0
    assert h == simhash64(BODY)
