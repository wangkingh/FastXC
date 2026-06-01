import unittest
import os
import tempfile
from datetime import datetime
from pathlib import Path
from fastxc.inventory.patterns import (
    FieldRegistry,
    DEFAULT_BASE_FIELDS,
    check_pattern,
    match_path,
    parse_timestamp,
)


class TestPatternUtils(unittest.TestCase):

    def setUp(self):
        """
        run at the beginning of each test method.
        """
        # initialize a FieldRegistry with DEFAULT_BASE_FIELDS
        self.registry = FieldRegistry(DEFAULT_BASE_FIELDS)

        # create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.array_dir = self.temp_dir.name

    def tearDown(self):
        """
        destroy at the end of each test method.
        """
        self.temp_dir.cleanup()

    def test_field_registry_init(self):
        """
        if FieldRegistry is initialized with DEFAULT_BASE_FIELDS
        """
        regex = self.registry.build_regex_pattern("{YYYY}/{station}_{component}.sac")
        self.assertIn("(?P<year>\\d{4})", regex)
        self.assertIn("(?P<station>", regex)
        self.assertIn("(?P<component>", regex)

    def test_field_registry_add_field(self):
        """
        using add_field to add a new field to the registry
        """
        self.registry.add_field("shot", r"\d+")
        regex = self.registry.build_regex_pattern("{YYYY}/{station}_{shot}_{component}.sac")
        self.assertIn("(?P<shot>\\d+)", regex)

    def test_field_registry_add_existing_field_no_overwrite(self):
        """
        if add_field is called with an existing field and overwrite=False
        """
        self.registry.add_field("YYYY", r"\d{8}", overwrite=False)
        regex = self.registry.build_regex_pattern("{YYYY}/{station}_{component}.sac")
        self.assertIn("(?P<year>\\d{4})", regex)

    def test_field_registry_add_existing_field_overwrite(self):
        """
        add_field with an existing field and overwrite=TrueS
        """
        self.registry.add_field("YYYY", r"\d{8}", overwrite=True)
        regex = self.registry.build_regex_pattern("{YYYY}/{station}_{component}.sac")
        self.assertIn("(?P<YYYY>\\d{8})", regex)

    def test_field_registry_add_invalid_regex(self):
        """
        if when add_field is called with an invalid regex
        """
        with self.assertRaises(ValueError):
            self.registry.add_field("invalidRegex", r"[abc")  # missing closing bracket

    def test_field_registry_validate_pattern_fields(self):
        """
        if validate_pattern_fields can check if a pattern contains all fields in the registry
        """
        # format string with all fields
        pattern_ok = "{YYYY}/{station}_{component}.sac"
        # format string with a non-existent field
        pattern_bad = "{YYYY}/{foo}_{component}.sac"

        # for a valid pattern, should not raise any exceptions
        self.registry.validate_pattern_fields(pattern_ok)

        # for a pattern with a non-existent field, should raise a ValueError
        with self.assertRaises(ValueError):
            self.registry.validate_pattern_fields(pattern_bad)

    def test_field_registry_build_regex_pattern(self):
        """
        test build_regex_pattern with a pattern
        """
        pattern = "{YYYY}/{station}.{component}_{?}/{*}"
        reg_str = self.registry.build_regex_pattern(pattern)

        # make sure all fields are replaced
        self.assertIn("(?P<year>\\d{4})", reg_str)
        # make sure '.' '_' '/' are escaped
        self.assertIn(r"\.", reg_str)
        # make sure '{?}' -> '[^. _/]*'
        self.assertIn("[^. _/]*", reg_str)
        # make sure '{*}' -> '.*'
        self.assertIn(".*", reg_str)

    def test_check_pattern_valid(self):
        """
        test check_pattern with a valid pattern
        """
        pattern = "{home}/{YYYY}/{station}_{component}.sac"
        regex_str = check_pattern(self.array_dir, pattern, self.registry)
        self.assertIsInstance(regex_str, str)
        # regex_str should contain the array_dir
        unescaped = (
            regex_str.replace("\\/", "/")  # 把 \/ 还原成 /
            .replace("\\_", "_")  # 把 \_ 还原成 _
            .replace("\\.", ".")  # 把 \. 还原成 .
        )
        self.assertIn(self.array_dir, unescaped)

    def test_match_path_is_root_relative_and_separator_independent(self):
        """
        Pattern matching should depend on the path below {home}, not on the
        platform-specific spelling of the absolute root path.
        """
        data_dir = Path(self.array_dir) / "2023" / "123"
        data_dir.mkdir(parents=True, exist_ok=True)
        sac_file = data_dir / "ABC.BHZ.sac"
        sac_file.write_text("test")

        pattern = r"{home}\{YYYY}\{JJJ}\{station}.{component}.sac"
        compiled = check_pattern(self.array_dir, pattern, self.registry)
        file_text = str(sac_file)
        if os.name == "nt":
            file_text = file_text.replace("/", "\\")
        fields = match_path(compiled, file_text)

        self.assertIsNotNone(fields)
        self.assertEqual(fields["station"], "ABC")
        self.assertEqual(fields["component"], "BHZ")
        self.assertEqual(parse_timestamp(fields), datetime(2023, 5, 3))

    def test_duplicate_field_acts_as_backreference(self):
        """
        Repeated fields are allowed and must match the first captured value.
        """
        data_dir = Path(self.array_dir) / "ABC" / "2023"
        data_dir.mkdir(parents=True, exist_ok=True)
        good_file = data_dir / "ABC.2023.123.BHZ.sac"
        bad_file = data_dir / "XYZ.2023.123.BHZ.sac"
        good_file.write_text("test")
        bad_file.write_text("test")

        pattern = "{home}/{station}/{YYYY}/{station}.{YYYY}.{JJJ}.{component}.sac"
        compiled = check_pattern(self.array_dir, pattern, self.registry)

        self.assertIsNotNone(match_path(compiled, good_file))
        self.assertIsNone(match_path(compiled, bad_file))

    def test_check_pattern_missing_necessary_fields(self):
        """
        if check_pattern is called with a pattern that is missing necessary fields
        """
        pattern = "{YYYY}/{station}.sac"  # lacks "{home}"
        with self.assertRaisesRegex(ValueError, "pattern must contain {home}"):
            check_pattern(self.array_dir, pattern, self.registry)

    def test_check_pattern_no_date_fields(self):
        """
        when check_pattern is called with a pattern that does not contain any date fields
        """
        pattern_no_date = "{home}/{station}_{component}.sac"
        with self.assertRaisesRegex(
            ValueError, "pattern must contain one set of date fields"
        ):
            check_pattern(self.array_dir, pattern_no_date, self.registry)

    def test_check_pattern_not_a_directory(self):
        """
        test check_pattern with a non-directory path
        """
        # create a file in the temp directory
        not_dir_path = os.path.join(self.array_dir, "myfile.txt")
        with open(not_dir_path, "w") as f:
            f.write("test")

        pattern = "{home}/{YYYY}/{station}_{component}.sac"
        # check_pattern should log a warning
        regex_str = check_pattern(not_dir_path, pattern, self.registry)
        self.assertIn("myfile\\.txt", regex_str)


if __name__ == "__main__":
    unittest.main()
