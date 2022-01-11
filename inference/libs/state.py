import enum


class State(enum.Enum):
    RED_LIGHT_ON = "swatting"
    RED_LIGHT_OFF = "not swatting"
    BLINKING = "blinking"
    NORMAL = "normal"


def is_bat_light_on(bbox_dict: dict, color="red"):
    bat_present = len(bbox_dict.get("bat", [])) > 0
    light_present = len(bbox_dict.get(color, [])) > 0

    return bat_present and light_present


def get_state(bbox_dict):
    if is_bat_light_on(bbox_dict, "red"):
        return State.RED_LIGHT_ON
    return State.RED_LIGHT_OFF

