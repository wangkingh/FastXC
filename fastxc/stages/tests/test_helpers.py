import unittest

from fastxc.stages.helpers import unpack_product_name


class TestUnpackProductName(unittest.TestCase):
    def test_single_component_uses_original_label(self):
        self.assertEqual(
            unpack_product_name("linearstack_sourcepack", ["BHZ"]),
            "ncf_linear_BHZ",
        )

    def test_three_component_stack_uses_enz_label(self):
        self.assertEqual(
            unpack_product_name("linearstack_sourcepack", ["BHZ", "BHN", "BHE"]),
            "ncf_linear_ENZ",
        )

    def test_rotated_stack_uses_rtz_label(self):
        self.assertEqual(
            unpack_product_name("rtz_linearstack_sourcepack", ["BHZ", "BHN", "BHE"]),
            "ncf_linear_RTZ",
        )

    def test_legacy_call_without_component_list_keeps_z(self):
        self.assertEqual(unpack_product_name("linearstack_sourcepack"), "ncf_linear_Z")


if __name__ == "__main__":
    unittest.main()
