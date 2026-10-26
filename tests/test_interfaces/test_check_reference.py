"""The generated check reference lists every core check with its codes."""

import pytest

from dsbf.checks import core_pack
from dsbf.engine import Registry
from dsbf.interfaces.check_reference import finding_codes, render_check_reference

# Checks whose output is a summary of other checks' findings; they raise none.
NO_FINDINGS = {"entropy", "ml_readiness", "quality_dimensions"}


@pytest.fixture(scope="module")
def checks():
    registry = Registry()
    registry.register_pack(core_pack())
    return registry.checks


@pytest.mark.unit
def test_every_core_check_has_a_section_and_its_parameters(checks):
    text = render_check_reference()

    for check in checks:
        assert f"\n## {check.id}\n" in text
        for name in check.spec.params.model_fields:
            assert f"| `{name}` |" in text


@pytest.mark.unit
def test_every_check_that_raises_findings_lists_a_code(checks):
    missing = {c.id for c in checks if not finding_codes(c)}

    assert missing == NO_FINDINGS


@pytest.mark.unit
def test_codes_bound_through_tables_and_loops_are_found(checks):
    by_id = {c.id: c for c in checks}

    assert "timestamp_after_event" in finding_codes(by_id["target_leakage"])
    assert {"seasonal", "trend"} <= set(finding_codes(by_id["seasonal_decomposition"]))
    assert "non_stationary" in finding_codes(by_id["stationarity"])


@pytest.mark.unit
def test_a_summary_is_not_repeated_in_the_body():
    text = render_check_reference()
    summary = "Compute the ACF and PACF of each numeric column on a regular series."

    assert f"{summary}\n\n{summary}" not in text
