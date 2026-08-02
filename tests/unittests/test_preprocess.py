import os
import tempfile
import shutil
import unittest
import pandas as pd
from unittest.mock import patch

import src.utils.preprocess as pp


# ---------------------------------------------------------------------------
# get_plug_data
# ---------------------------------------------------------------------------

class TestGetPlugData(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_returns_empty_df_when_plug_file_missing(self):
        sm_dir = os.path.join(self.tmp_dir, "sm_01")
        os.makedirs(sm_dir)
        raw_fp = os.path.join(sm_dir, "2012-01-01.csv")
        with open(raw_fp, "w") as f:
            f.write("1.0\n")

        with patch.object(pp, "get_appliance_name", return_value="kettle"):
            result = pp.get_plug_data(raw_fp, "03", "sm_01")

        self.assertTrue(result.empty)
        self.assertEqual(result.columns.tolist(), ["kettle"])
        self.assertEqual(result.dtypes["kettle"], float)

    def test_reads_existing_plug_file(self):
        raw_dir = os.path.join(self.tmp_dir, "sm_01")
        os.makedirs(raw_dir)
        raw_fp = os.path.join(raw_dir, "2012-01-01.csv")
        with open(raw_fp, "w") as f:
            f.write("100\n")

        plug_dir = os.path.join(self.tmp_dir, "plug_01", "01")
        os.makedirs(plug_dir)
        plug_fp = os.path.join(plug_dir, "2012-01-01.csv")
        with open(plug_fp, "w") as f:
            f.write("10.5\n20.5\n30.5\n")

        with patch.object(pp, "get_appliance_name", return_value="fridge"):
            result = pp.get_plug_data(raw_fp, "01", "sm_01")

        self.assertEqual(result.columns.tolist(), ["fridge"])
        self.assertEqual(len(result), 3)
        self.assertEqual(result["fridge"].tolist(), [10.5, 20.5, 30.5])

    def test_column_name_uses_get_appliance_name(self):
        raw_dir = os.path.join(self.tmp_dir, "sm_02")
        os.makedirs(raw_dir)
        raw_fp = os.path.join(raw_dir, "day.csv")
        with open(raw_fp, "w") as f:
            f.write("1\n")

        with patch.object(pp, "get_appliance_name", return_value="tv") as mock_name:
            pp.get_plug_data(raw_fp, "05", "sm_02")

        mock_name.assert_called_once_with("sm_02", "05")

    def test_non_sm_prefixed_dir_unchanged(self):
        other_dir = os.path.join(self.tmp_dir, "notsm_01")
        os.makedirs(other_dir)
        raw_fp = os.path.join(other_dir, "day.csv")
        with open(raw_fp, "w") as f:
            f.write("1\n")

        with patch.object(pp, "get_appliance_name", return_value="misc"):
            result = pp.get_plug_data(raw_fp, "01", "notsm_01")

        self.assertTrue(result.empty)
        self.assertEqual(result.columns.tolist(), ["misc"])


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

class TestPreprocess(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_preprocess_builds_correct_output_path(self):
        with patch.object(pp, "APPLIANCE_MAP", {"sm_01": {"fridge": "01"}}), \
             patch.object(pp.pd, "read_csv", return_value=pd.DataFrame({"power": [1.0, 2.0]})), \
             patch.object(pp, "get_plug_data", return_value=pd.DataFrame({"fridge": [0.1, 0.2]})), \
             patch.object(pp.os, "makedirs") as mock_makedirs, \
             patch.object(pp, "PREPROCESS_DIR", self.tmp_dir), \
             patch.object(pd.DataFrame, "to_csv") as mock_to_csv:

            pp.preprocess("/data/eco/raw/sm_01/2012-01-01.csv")

            expected_dir = os.path.join(self.tmp_dir, "sm_01")
            expected_fp = os.path.join(expected_dir, "2012-01-01.csv")
            mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)
            mock_to_csv.assert_called_once()
            self.assertEqual(mock_to_csv.call_args[0][0], expected_fp)
            self.assertEqual(mock_to_csv.call_args[1], {"index": False})

    def test_preprocess_calls_get_plug_data_for_each_appliance(self):
        with patch.object(pp, "APPLIANCE_MAP", {"sm_01": {"fridge": "01", "kettle": "02"}}), \
             patch.object(pp.pd, "read_csv", return_value=pd.DataFrame({"power": [1.0]})), \
             patch.object(pp, "get_plug_data", return_value=pd.DataFrame({"x": [0.1]})) as mock_get_plug, \
             patch.object(pp.os, "makedirs"), \
             patch.object(pp, "PREPROCESS_DIR", self.tmp_dir), \
             patch.object(pd.DataFrame, "to_csv"):

            pp.preprocess("/data/eco/raw/sm_01/2012-01-01.csv")

            self.assertEqual(mock_get_plug.call_count, 2)
            mock_get_plug.assert_any_call("/data/eco/raw/sm_01/2012-01-01.csv", "01", "sm_01")
            mock_get_plug.assert_any_call("/data/eco/raw/sm_01/2012-01-01.csv", "02", "sm_01")

    def test_preprocess_no_appliances_only_mains(self):
        with patch.object(pp, "APPLIANCE_MAP", {"sm_01": {}}), \
             patch.object(pp.pd, "read_csv", return_value=pd.DataFrame({"power": [1.0, 2.0, 3.0]})), \
             patch.object(pp.os, "makedirs"), \
             patch.object(pp, "PREPROCESS_DIR", self.tmp_dir), \
             patch.object(pd.DataFrame, "to_csv") as mock_to_csv:

            pp.preprocess("/data/eco/raw/sm_01/day.csv")

            mock_to_csv.assert_called_once()

    def test_preprocess_concats_mains_and_appliance_columns(self):
        captured = {}

        def fake_to_csv(self, path, index=False):
            captured["df"] = self.copy()

        with patch.object(pp, "APPLIANCE_MAP", {"sm_01": {"fridge": "01"}}), \
             patch.object(pp.pd, "read_csv", return_value=pd.DataFrame({"power": [1.0, 2.0]})), \
             patch.object(pp, "get_plug_data", return_value=pd.DataFrame({"fridge": [0.5, 0.6]})), \
             patch.object(pp.os, "makedirs"), \
             patch.object(pp, "PREPROCESS_DIR", self.tmp_dir), \
             patch.object(pd.DataFrame, "to_csv", fake_to_csv):

            pp.preprocess("/data/eco/raw/sm_01/day.csv")

            self.assertEqual(list(captured["df"].columns), ["power", "fridge"])
            self.assertEqual(captured["df"]["power"].tolist(), [1.0, 2.0])
            self.assertEqual(captured["df"]["fridge"].tolist(), [0.5, 0.6])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain(unittest.TestCase):

    def test_main_calls_preprocess_for_each_csv(self):
        with patch.object(pp, "preprocess") as mock_preprocess, \
             patch.object(pp.glob, "glob") as mock_glob:

            mock_glob.side_effect = [
                ["/data/eco/raw/sm_01", "/data/eco/raw/sm_02"],
                ["/data/eco/raw/sm_01/a.csv", "/data/eco/raw/sm_01/b.csv"],
                ["/data/eco/raw/sm_02/c.csv"],
            ]

            pp.main()

            self.assertEqual(mock_preprocess.call_count, 3)
            mock_preprocess.assert_any_call("/data/eco/raw/sm_01/a.csv")
            mock_preprocess.assert_any_call("/data/eco/raw/sm_01/b.csv")
            mock_preprocess.assert_any_call("/data/eco/raw/sm_02/c.csv")

    def test_main_no_meter_dirs_does_nothing(self):
        with patch.object(pp, "preprocess") as mock_preprocess, \
             patch.object(pp.glob, "glob", return_value=[]):

            pp.main()

            mock_preprocess.assert_not_called()

    def test_main_no_csvs_in_meter_dir(self):
        with patch.object(pp, "preprocess") as mock_preprocess, \
             patch.object(pp.glob, "glob") as mock_glob:

            mock_glob.side_effect = [["/data/eco/raw/sm_01"], []]

            pp.main()

            mock_preprocess.assert_not_called()


if __name__ == "__main__":
    unittest.main()
