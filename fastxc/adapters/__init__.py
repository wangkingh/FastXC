"""Adapters that translate FastXC stage settings into native executable commands."""

from .sac2spec import gen_sac2spec_cmd, sac2spec_deployer
from .stack import gen_pws_sourcepack_cmd, gen_tfpws_sourcepack_cmd, weighted_stack_deployer
from .xc import gen_xc_cmd, xc_deployer

__all__ = [
    "gen_pws_sourcepack_cmd",
    "gen_sac2spec_cmd",
    "gen_tfpws_sourcepack_cmd",
    "gen_xc_cmd",
    "sac2spec_deployer",
    "weighted_stack_deployer",
    "xc_deployer",
]
