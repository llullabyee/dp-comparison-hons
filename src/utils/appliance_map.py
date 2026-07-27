from enum import Enum

class ApplianceList(Enum):
    FRIDGE = 'fridge'
    DRYER = 'dryer'
    COFFEE_MACHINE = 'coffee_machine'
    KETTLE = 'kettle'
    WASHING_MACHINE = 'washing_machine'
    PC = 'pc'
    FREEZER = 'freezer'
    TABLET = 'tablet'
    DISHWASHER = 'dishwasher'
    AIR_EXHAUST = 'air_exhaust'
    ENTERTAINMENT = 'entertainment'
    LAMP = 'lamp'
    LAPTOPS = 'laptops'
    STOVE = 'stove'
    TV = 'tv'
    STEREO = 'stereo'
    MICROWAVE = 'microwave'
    KITCHEN_APPLIANCES = 'kitchen_appliances'
    FOUNTAIN = 'fountain'
    ROUTER = 'router'
    STEREO_LAPTOP = 'stereo_laptop'

APPLIANCE_MAP = {
    'sm_01': {
        ApplianceList.FRIDGE: '01',
        ApplianceList.DRYER: '02',
        ApplianceList.COFFEE_MACHINE: '03',
        ApplianceList.KETTLE: '04',
        ApplianceList.WASHING_MACHINE: '05',
        ApplianceList.PC: '06',
        ApplianceList.FREEZER: '07',
    },
    'sm_02': {
        ApplianceList.TABLET: '01',
        ApplianceList.DISHWASHER: '02',
        ApplianceList.AIR_EXHAUST: '03',
        ApplianceList.FRIDGE: '04',
        ApplianceList.ENTERTAINMENT: '05',
        ApplianceList.FREEZER: '06',
        ApplianceList.KETTLE: '07',
        ApplianceList.LAMP: '08',
        ApplianceList.LAPTOPS: '09',
        ApplianceList.STOVE: '10',
        ApplianceList.TV: '11',
        ApplianceList.STEREO: '12',
    },
    'sm_03': {
        ApplianceList.TABLET: '01',
        ApplianceList.FREEZER: '02',
        ApplianceList.COFFEE_MACHINE: '03',
        ApplianceList.PC: '04',
        ApplianceList.FRIDGE: '05',
        ApplianceList.KETTLE: '06',
        ApplianceList.ENTERTAINMENT: '07',
    },
    'sm_04': {
        ApplianceList.FRIDGE: '01',
        ApplianceList.KITCHEN_APPLIANCES: '02',
        ApplianceList.LAMP: '03',
        ApplianceList.STEREO_LAPTOP: '04',
        ApplianceList.FREEZER: '05',
        ApplianceList.TABLET: '06',
        ApplianceList.ENTERTAINMENT: '07',
        ApplianceList.MICROWAVE: '08',
    },
    'sm_05': {
        ApplianceList.TABLET: '01',
        ApplianceList.COFFEE_MACHINE: '02',
        ApplianceList.FOUNTAIN: '03',
        ApplianceList.MICROWAVE: '04',
        ApplianceList.FRIDGE: '05',
        ApplianceList.ENTERTAINMENT: '06',
        ApplianceList.PC: '07',
        ApplianceList.KETTLE: '08',
    },
    'sm_06': {
        ApplianceList.LAMP: '01',
        ApplianceList.LAPTOPS: '02',
        ApplianceList.ROUTER: '03',
        ApplianceList.COFFEE_MACHINE: '04',
        ApplianceList.ENTERTAINMENT: '05',
        ApplianceList.FRIDGE: '06',
        ApplianceList.KETTLE: '07',
    },
}

REVERSE_APPLIANCE_MAP = {
    meter_id: {no: appliance for appliance, no in appliances.items()}
    for meter_id, appliances in APPLIANCE_MAP.items()
}

def get_appliance_no(meter_id: str, appliance: ApplianceList):
    result = APPLIANCE_MAP[meter_id].get(appliance)
    if result is None:
        return None
    return result

def get_appliance_name(meter_id: str, appliance_no: str):
    result = REVERSE_APPLIANCE_MAP[meter_id].get(appliance_no)
    if result is None:
        return None
    return result.value
