from enroadspy import name_to_id, load_input_specs


def process_config(config: dict) -> dict:
    input_specs = load_input_specs()

    config["context"] = [name_to_id(name, input_specs) for name in config["context"]]
    config["actions"] = [name_to_id(name, input_specs) for name in config["actions"]]

    return config