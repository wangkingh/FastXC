import unittest

from fastxc.cli import _only_stage_modes, _stack_stage_names, build_parser


class CliStageAliasTests(unittest.TestCase):
    def test_stage_alias_commands_parse_config(self):
        parser = build_parser()

        self.assertEqual(parser.parse_args(["sac2spec", "config.ini"]).command, "sac2spec")
        self.assertEqual(parser.parse_args(["xc", "config.ini"]).command, "xc")
        self.assertEqual(parser.parse_args(["stack", "config.ini"]).command, "stack")
        self.assertEqual(parser.parse_args(["rotate", "config.ini"]).command, "rotate")

    def test_xc_can_skip_sourcepack(self):
        args = build_parser().parse_args(["xc", "config.ini", "--no-sourcepack"])

        self.assertTrue(args.no_sourcepack)

    def test_stack_method_names_map_to_stage_names(self):
        self.assertEqual(_stack_stage_names("linear,pws"), ["LinearStack", "PwsStack"])
        self.assertEqual(_stack_stage_names("all"), ["LinearStack", "PwsStack", "TfPwsStack"])
        with self.assertRaisesRegex(ValueError, "unsupported stack method"):
            _stack_stage_names("bad")
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            _stack_stage_names("all,bad")

    def test_only_stage_modes_skip_unselected_compute_stages(self):
        modes = _only_stage_modes(["CrossCorrelation", "SourcePack"])

        self.assertEqual(modes["CrossCorrelation"], "ALL")
        self.assertEqual(modes["SourcePack"], "ALL")
        self.assertEqual(modes["Sac2Spec"], "SKIP")
        self.assertEqual(modes["LinearStack"], "SKIP")


if __name__ == "__main__":
    unittest.main()
