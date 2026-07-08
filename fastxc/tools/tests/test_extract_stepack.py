import unittest
from pathlib import Path

from fastxc.tools.extract_stepack import default_plot_output_path


class ExtractStepackPlotTests(unittest.TestCase):
    def test_default_plot_output_path_replaces_mat_suffix(self):
        self.assertEqual(
            default_plot_output_path(Path("AA.stepack.mat"), quantity="amplitude", db=False).name,
            "AA.stepack.amplitude.png",
        )

    def test_default_plot_output_path_marks_db_amplitude(self):
        self.assertEqual(
            default_plot_output_path(Path("AA.stepack.mat"), quantity="power", db=True).name,
            "AA.stepack.power_db.png",
        )


if __name__ == "__main__":
    unittest.main()
