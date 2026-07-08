import unittest
from pathlib import Path

from fastxc.operators.stacking.linear_stack import linear_output_path


class LinearStackNamingTests(unittest.TestCase):
    def test_linear_output_path_uses_sac_final_pair_name(self) -> None:
        out = linear_output_path("AA-BB.STA1-STA2.Z-Z.sac", Path("stack"))

        self.assertEqual(
            out.parts[-3:],
            (
                "linearstack",
                "AA-BB.STA1-STA2",
                "AA-BB.STA1-STA2.Z-Z.linearstack.sac",
            ),
        )

    def test_linear_output_path_accepts_unsuffixed_pair_name(self) -> None:
        out = linear_output_path("AA-BB.STA1-STA2.Z-Z", Path("stack"))

        self.assertEqual(out.name, "AA-BB.STA1-STA2.Z-Z.linearstack.sac")


if __name__ == "__main__":
    unittest.main()
