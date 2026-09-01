from enum import Enum

class ApplianceList(Enum):
    KETTLE = 'kettle'
    FRIDGE = 'fridge'
    FRIDGE_FREEZER = 'fridgefreezer'
    FREEZER = 'freezer'
    WASHING_MACHINE = 'washingmachine'
    WASHING_MACHINE_TUMBLE_DRIER = 'washingmachinetumbledrier'
    TUMBLE_DRIER = 'tumbledrier'
    DISHWASHER = 'dishwasher'
    MICROWAVE = 'microwave'
    TOASTER = 'toaster'
    VACUUM_CLEANER = 'vacuumcleaner'
    ELECTRIC_HEATER = 'electricheater'
    ELECTRIC_OVEN = 'electricoven'
    DEHUMIDIFIER = 'dehumidifier'
    GRILL = 'grill'
    OTHER = 'other'
    APPLIANCE = 'appliance'


# def get_appliance_no(meter_id: str, appliance: ApplianceList):
#     result = APPLIANCE_MAP[meter_id].get(appliance)
#     if result is None:
#         return None
#     return result

# def get_appliance_name(meter_id: str, appliance_no: str):
#     result = REVERSE_APPLIANCE_MAP[meter_id].get(appliance_no)
#     if result is None:
#         return None
#     return result.value
