import unittest

from src.utils.appliance_map import (
    ApplianceList,
    APPLIANCE_MAP,
    REVERSE_APPLIANCE_MAP,
    get_appliance_no,
    get_appliance_name,
)


class TestApplianceMapIntegrity(unittest.TestCase):
    def test_all_meters_present(self):
        expected_meters = {'sm_01', 'sm_02', 'sm_03', 'sm_04', 'sm_05', 'sm_06'}
        self.assertEqual(set(APPLIANCE_MAP.keys()), expected_meters)

    def test_appliance_numbers_are_zero_padded_strings(self):
        for meter_id, appliances in APPLIANCE_MAP.items():
            for appliance, no in appliances.items():
                self.assertIsInstance(no, str, f"{meter_id}/{appliance} number is not a string")
                self.assertEqual(len(no), 2, f"{meter_id}/{appliance} number is not zero-padded to 2 digits")
                self.assertTrue(no.isdigit(), f"{meter_id}/{appliance} number is not numeric")

    def test_appliance_numbers_unique_per_meter(self):
        for meter_id, appliances in APPLIANCE_MAP.items():
            nos = list(appliances.values())
            self.assertEqual(len(nos), len(set(nos)), f"Duplicate appliance numbers in {meter_id}")

    def test_appliance_numbers_start_at_01_and_are_contiguous(self):
        for meter_id, appliances in APPLIANCE_MAP.items():
            nos = sorted(int(no) for no in appliances.values())
            expected = list(range(1, len(nos) + 1))
            self.assertEqual(nos, expected, f"{meter_id} appliance numbers are not contiguous from 01")

    def test_all_keys_are_applianceList_members(self):
        for meter_id, appliances in APPLIANCE_MAP.items():
            for appliance in appliances:
                self.assertIsInstance(appliance, ApplianceList, f"{meter_id} has non-enum key {appliance}")


class TestReverseApplianceMap(unittest.TestCase):
    def test_reverse_map_has_same_meters(self):
        self.assertEqual(set(REVERSE_APPLIANCE_MAP.keys()), set(APPLIANCE_MAP.keys()))

    def test_reverse_map_round_trip(self):
        for meter_id, appliances in APPLIANCE_MAP.items():
            for appliance, no in appliances.items():
                self.assertEqual(REVERSE_APPLIANCE_MAP[meter_id][no], appliance)

    def test_reverse_map_size_matches_forward_map(self):
        for meter_id, appliances in APPLIANCE_MAP.items():
            self.assertEqual(len(REVERSE_APPLIANCE_MAP[meter_id]), len(appliances))


class TestGetApplianceNo(unittest.TestCase):
    def test_known_appliance_returns_correct_no(self):
        self.assertEqual(get_appliance_no('sm_01', ApplianceList.FRIDGE), '01')
        self.assertEqual(get_appliance_no('sm_02', ApplianceList.STEREO), '12')
        self.assertEqual(get_appliance_no('sm_06', ApplianceList.KETTLE), '07')

    def test_appliance_not_present_for_meter_returns_none(self):
        # DRYER only exists for sm_01
        self.assertIsNone(get_appliance_no('sm_02', ApplianceList.DRYER))

    def test_unknown_meter_id_raises_keyerror(self):
        with self.assertRaises(KeyError):
            get_appliance_no('sm_99', ApplianceList.FRIDGE)


class TestGetApplianceName(unittest.TestCase):
    def test_known_no_returns_correct_name(self):
        self.assertEqual(get_appliance_name('sm_01', '01'), 'fridge')
        self.assertEqual(get_appliance_name('sm_02', '12'), 'stereo')
        self.assertEqual(get_appliance_name('sm_06', '07'), 'kettle')

    def test_unpadded_no_returns_none(self):
        # get_appliance_name expects zero-padded strings; '1' should not match '01'
        self.assertIsNone(get_appliance_name('sm_01', '1'))

    def test_no_not_present_for_meter_returns_none(self):
        # sm_01 only has appliance numbers '01' through '07'
        self.assertIsNone(get_appliance_name('sm_01', '08'))

    def test_unknown_meter_id_raises_keyerror(self):
        with self.assertRaises(KeyError):
            get_appliance_name('sm_99', '01')

    def test_returns_plain_string_not_enum(self):
        result = get_appliance_name('sm_01', '01')
        self.assertIsInstance(result, str)
        self.assertNotIsInstance(result, ApplianceList)


class TestGetApplianceNoAndNameRoundTrip(unittest.TestCase):
    def test_round_trip_all_meters(self):
        for meter_id, appliances in APPLIANCE_MAP.items():
            for appliance, no in appliances.items():
                looked_up_no = get_appliance_no(meter_id, appliance)
                self.assertEqual(looked_up_no, no)
                looked_up_name = get_appliance_name(meter_id, looked_up_no)
                self.assertEqual(looked_up_name, appliance.value)


if __name__ == '__main__':
    unittest.main()
