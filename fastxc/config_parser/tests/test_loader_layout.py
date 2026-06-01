import tempfile
import unittest
from pathlib import Path

from fastxc.config_parser.loader import Config


class TestConfigLoaderLayout(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_public_layout_reads_core_and_advanced_fields(self):
        cfg = self._write_config(
            """
[seisarray1]
sac_dir = {data_dir}
pattern = {{home}}/{{station}}/{{YYYY}}/{{station}}.{{YYYY}}.{{JJJ}}.{{component}}.sac
sta_list = NONE
component_list = E,N,Z

[time_filter]
time_start = 2022-01-01 00:00:00
time_end = 2022-01-02 00:00:00
time_list = NONE

[geometry]
external_geo_tsv = NONE

[executables]
executable_root = NONE
sac2spec = NONE
xc = NONE
pws = NONE
tfpws = NONE

[compute]
sac_len = 86400
win_len = 3600
shift_len = AUTO
delta = 1.0
normalize = AUTO
bands = 0.1/0.2 0.2/0.4
whiten = AFTER
max_lag = 50
stack_flag = 111
workspace_dir = {workspace_dir}

[device]
gpu_list = 0
cpu_workers = 1

[advance.compute]
skip_step = 1,2
phase_only = True
distance_range = 10/20
azimuth_range = 30/40
group_pair_mode = inter
windows_per_xcache = 12
xcache_async_after_sac2spec = False
async_poll_sec = 3
xcache_cleanup_timestamp_spack = False
sourcepack_async_after_xc = False
pre_stack_size = 7
tfpws_band = 0.1/0.3
tfpws_taper_hz = 0.02

[advance.storage]
unpack_enabled = False
unpack_target = STACK
""",
        )

        self.assertEqual(cfg.compute.shift_len, cfg.compute.win_len)
        self.assertTrue(cfg.compute.phase_only)
        self.assertEqual(cfg.compute.skip_step, "1,2")
        self.assertEqual(cfg.xcorr.max_lag, 50)
        self.assertEqual(cfg.xcorr.distance_range, "10/20")
        self.assertEqual(cfg.xcorr.azimuth_range, "30/40")
        self.assertEqual(cfg.xcorr.group_pair_mode, "inter")
        self.assertEqual(cfg.stack.stack_flag, "111")
        self.assertEqual(cfg.stack.pre_stack_size, 7)
        self.assertEqual(cfg.xcache.windows_per_xcache, 12)
        self.assertFalse(cfg.xcache.async_after_sac2spec)
        self.assertEqual(cfg.xcache.async_poll_sec, 3)
        self.assertFalse(cfg.xcache.cleanup_timestamp_spack)
        self.assertTrue(cfg.sourcepack.sort_within_source)
        self.assertFalse(cfg.sourcepack.async_after_xc)
        self.assertEqual(cfg.sourcepack.async_poll_sec, 3)
        self.assertFalse(cfg.unpack.enabled)
        self.assertEqual(cfg.unpack.target, "STACK")
        self.assertEqual(cfg.unpack.output_dir, "result_ncf")
        self.assertEqual(cfg.storage.workspace_dir, (self.root / "workspace").resolve())

    def test_old_preprocess_section_is_rejected(self):
        with self.assertRaisesRegex(Exception, "Retired INI section"):
            self._write_config(
                """
[seisarray1]
sac_dir = {data_dir}
pattern = {{home}}/{{station}}/{{YYYY}}/{{station}}.{{YYYY}}.{{JJJ}}.{{component}}.sac
sta_list = NONE
component_list = E,N,Z

[time_filter]
time_start = 2022-01-01 00:00:00
time_end = 2022-01-02 00:00:00
time_list = NONE

[executables]
executable_root = NONE
sac2spec = NONE
xc = NONE
pws = NONE
tfpws = NONE

[preprocess]
sac_len = 86400
win_len = 3600
shift_len = AUTO
delta = 1.0
normalize = AUTO
bands = 0.1/0.2
whiten = AFTER
max_lag = 50
stack_flag = 100
workspace_dir = {workspace_dir}

[device]
gpu_list = 0
cpu_workers = 1
""",
            )

    def test_old_advance_subsection_is_rejected(self):
        with self.assertRaisesRegex(Exception, "Retired INI section"):
            self._write_config(
                """
[seisarray1]
sac_dir = {data_dir}
pattern = {{home}}/{{station}}/{{YYYY}}/{{station}}.{{YYYY}}.{{JJJ}}.{{component}}.sac
sta_list = NONE
component_list = E,N,Z

[time_filter]
time_start = 2022-01-01 00:00:00
time_end = 2022-01-02 00:00:00
time_list = NONE

[executables]
executable_root = NONE
sac2spec = NONE
xc = NONE
pws = NONE
tfpws = NONE

[compute]
sac_len = 86400
win_len = 3600
shift_len = AUTO
delta = 1.0
normalize = AUTO
bands = 0.1/0.2
whiten = AFTER
max_lag = 50
stack_flag = 100
workspace_dir = {workspace_dir}

[device]
gpu_list = 0
cpu_workers = 1

[advance.xcache]
windows_per_xcache = AUTO
""",
            )

    def test_retired_advance_field_is_rejected(self):
        with self.assertRaisesRegex(Exception, "Retired INI field"):
            self._write_config(
                """
[seisarray1]
sac_dir = {data_dir}
pattern = {{home}}/{{station}}/{{YYYY}}/{{station}}.{{YYYY}}.{{JJJ}}.{{component}}.sac
sta_list = NONE
component_list = E,N,Z

[time_filter]
time_start = 2022-01-01 00:00:00
time_end = 2022-01-02 00:00:00
time_list = NONE

[executables]
executable_root = NONE
sac2spec = NONE
xc = NONE
pws = NONE
tfpws = NONE

[compute]
sac_len = 86400
win_len = 3600
shift_len = AUTO
delta = 1.0
normalize = AUTO
bands = 0.1/0.2
whiten = AFTER
max_lag = 50
stack_flag = 100
workspace_dir = {workspace_dir}

[device]
gpu_list = 0
cpu_workers = 1

[advance.compute]
sourcepack_enabled = True
""",
            )

    def _write_config(self, template: str) -> Config:
        path = self.root / "config.ini"
        text = template.format(
            data_dir=self.data_dir.as_posix(),
            workspace_dir=(self.root / "workspace").as_posix(),
        )
        path.write_text(text.strip() + "\n", encoding="utf-8")
        return Config(path)


if __name__ == "__main__":
    unittest.main()
