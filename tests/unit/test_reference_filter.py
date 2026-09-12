"""Unit tests for the references-section detector (T19).

Test fixtures are abridged excerpts of real chunks from the ``evaluation``
corpus (verified against all 619 chunks on 2026-08-29): every "hit" case is a
genuine references tail chunk, every "miss" case is a real content chunk that
shares surface features with references (inline ``[n]`` citations, years,
venue words) — the false-positive guards the ticket demands (零误伤).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.query_engine.reference_filter import detect_reference_section


# =============================================================================
# Hit cases — real references chunks (Springer LNCS unnumbered style)
# =============================================================================

SPRINGER_REFS = """Cid, C., Murphy, S., Robshaw, M.: Small scale variants of the aes. In: International Conference on Fast Software Encryption (2005)

Daemen, J., Rijmen, V.: The Design of Rijndael: AES - The Advanced Encryption Standard. The Design of Rijndael: AES - The Advanced Encryption Standard (2002)

Grassl, M., Langenberg, B., Roetteler, M., Steinwandt, R.: Applying grover's algorithm to aes: quantum resource estimates. In: Springer International Publishing (2015)

Grover, L.K.: A fast quantum mechanical algorithm for database search (1996)"""


# Numbered style (cf17de71 tail): "26. Author A, Author B. Title. Venue (year)"
NUMBERED_REFS = """26. Huang Z, Sun S. Synthesizing quantum circuits of aes with lower t-depth and less qubits. Cryptology ePrint Archive, Paper 2022/620 (2022)

27. Jaques S, Naehrig M, Roetteler M, Virdia F. Implementing grover oracles for quantum key-search on kerneq and prism. In: IACR Transactions on Symmetric Cryptology (2020)

28. Kim P, Han D. Quantum complexity of aes-256. IEEE Transactions on Information Theory 68(4) (2022)

29. Langenberg B, Pham H, Steinwandt R. Reducing the cost of implementing aes as a quantum circuit. IEEE Transactions on Quantum Engineering (2020)"""


# Bracket-numbered style: "[1] Author: Title. Venue, year"
BRACKET_REFS = """[1] Shor, P.W.: Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer. SIAM J. Comput. 26 (1997)

[2] Simon, D.: On the power of quantum computation. In: Proceedings of the 35th Annual Symposium on Foundations of Computer Science (1994)

[3] Daemen, J., Rijmen, V.: Specification for the advanced encryption standard. NIST (2000)"""


@pytest.mark.parametrize(
    "name,text",
    [
        ("springer_unnumbered", SPRINGER_REFS),
        ("numbered_ieee_style", NUMBERED_REFS),
        ("bracket_numbered", BRACKET_REFS),
    ],
)
def test_detects_real_references_chunks(name: str, text: str) -> None:
    assert detect_reference_section(text) is True


# =============================================================================
# Miss cases — real content chunks that must survive (零误伤 guards)
# =============================================================================

# 0fd74b7e_0060: Toffoli depth comparison body text (q22's squeezed-out victim)
CONTENT_TOFFOLI = """One can easily check that both circuits listed in Table 4 perform the same function. However, Circuit 2 costs one more CNOT gate than Circuit 1 (caused by the third operation in Circuit 2). Besides, the Toffoli depth of Circuit 2 is two, while the Toffoli depth of Circuit 1 is one.

Observation 1 Given a quantum circuit with Toffoli gates involved, the Toffoli depth and the CNOT gate consumption of the quantum circuit may be affected by the specific arrangement of CNOT gates.

Example 2 For a quantum circuit denoted by Circuit 3 in Table 5, a is not the operand of the second operation."""

# d28d9bf1_0061: Conclusion — syntactically coherent prose, no entry lines
CONTENT_CONCLUSION = """5 Conclusion

In this work, we focus on minimizing the depth of a subclass of quantum circuits - the CNOT circuits, especially those for the linear building blocks of symmetric-key ciphers. We are motivated by the quantum security analysis of current symmetric-key encryption systems and end up with a framework for constructing low-depth quantum circuits for linear Boolean functions.

Acknowledgements. This work is supported by the National Natural Science Foundation of China (Grant No. 61977060)."""


# Body text with dense inline [n] citations — coherent sentences, not entries
CONTENT_INLINE_CITATIONS = """The resource estimates of Grover's algorithm applied to AES have been studied extensively [12]. Grassl et al. gave the first full quantum implementation of AES [13], and subsequent works reduced the Toffoli depth further [14] [15]. A comparison of these estimates appears in Table 1. As shown in [13], the key schedule dominates the qubit count.

Our approach differs from [14] in that we optimize the linear layer separately from the S-box."""


# Table chunk (d28d9bf1_0060): venue-ish words + [47] footnote citations + years
CONTENT_TABLE_WITH_NOTES = """Table 2. Quantum circuit depth of some invertible matrices with different optimization methods.

1 The size refers to the degree of the corresponding binary matrix.

2 The number of CNOT gates of quantum implementation in [47].

3 Quantum circuit depth with sequence depth.

4 Quantum circuit depth with minimum move-equivalent depth."""


@pytest.mark.parametrize(
    "name,text",
    [
        ("toffoli_body", CONTENT_TOFFOLI),
        ("conclusion", CONTENT_CONCLUSION),
        ("inline_citations", CONTENT_INLINE_CITATIONS),
        ("table_with_notes", CONTENT_TABLE_WITH_NOTES),
    ],
)
def test_does_not_flag_real_content_chunks(name: str, text: str) -> None:
    assert detect_reference_section(text) is False


# =============================================================================
# Degenerate inputs — never crash, never flag
# =============================================================================

@pytest.mark.parametrize(
    "text",
    [
        "",
        "\n\n  \n",
        "Single line of plain text.",
        "Two lines only.\nBoth plain prose, no entries here.",
    ],
)
def test_degenerate_inputs_are_not_references(text: str) -> None:
    assert detect_reference_section(text) is False


def test_none_text_is_not_references() -> None:
    # RetrievalResult.text is Optional — callers may pass None through
    assert detect_reference_section(None) is False  # type: ignore[arg-type]
