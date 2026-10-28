"""A check that raises, so a fixture run has an `error` result with a reason."""

from dsbf.engine import SemanticTypes, check


@check(
    id="always_fails",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes(),
    dimensions=["validity"],
    packs=["fixture"],
)
def always_fails(ctx, params):
    """Raise, to show how a failed check reads on the dashboard."""
    raise RuntimeError("the fixture check always fails")
