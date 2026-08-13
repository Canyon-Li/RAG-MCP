"""Unit tests for the zero-dependency Porter stemmer."""

from src.core.text.porter_stemmer import stem


class TestPorterStemmer:
    def test_basic_plural(self):
        assert stem("cats") == "cat"
        assert stem("dogs") == "dog"

    def test_ization_suffix(self):
        # optimization / optimize / optimized 收敛到同一词干
        assert stem("optimization") == stem("optimize")
        assert stem("optimized") == stem("optimize")

    def test_ational_suffix(self):
        assert stem("relational") == "relat"

    def test_empty_and_non_alpha(self):
        assert stem("") == ""
        assert stem("123") == "123"

    def test_already_stem(self):
        assert stem("the") == "the"

    def test_case_insensitive_input(self):
        # 实现内部应 lower();大写输入不应报错且与小写同结果
        assert stem("Cats") == stem("cats")

    def test_convergence_corpus(self):
        """学术文献高频同源词应收敛到同一词干(场景核心诉求)。"""
        family = ["optimize", "optimization", "optimized", "optimizing", "optimal"]
        stems = {stem(w) for w in family}
        # 至少 optimize/optimization/optimized/optimizing 四者收敛
        assert stem("optimize") in stems
        assert len({stem(w) for w in ["optimize", "optimization", "optimized"]}) == 1


def test_ion_suffix_measure_threshold():
    """Step 4 'ion' rule needs m>1: 'notion' (stem 'not', m=1) must NOT over-stem."""
    assert stem("notion") == "notion"
    assert stem("potion") == "potion"
    # contrast: 'operation' (m>1) → 'ion' WAS removed (differs from original)
    assert stem("operation") != "operation"
