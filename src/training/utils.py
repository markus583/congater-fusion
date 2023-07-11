import inspect

def map_omega_grid(
    config: dict, seed, adapter_type: str = "pfeiffer"
) -> dict[str, float]:
    config = eval(config)
    if adapter_type == "pfeiffer":
        adapter_path = "st-a"
    elif adapter_type == "congaterV5":
        adapter_path = "C-V5"

    paths = []
    for adapter in config.keys():
        path = "runs/" + adapter_path + "/" + adapter + "/bert-base-uncased/100/" + str(seed)
        paths.append(path)

    # create dict: {path: omega}
    paths = {path: omega for path, omega in zip(paths, config.values())}

    return paths


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


if __name__ == "__main__":
    adapter_type = "pfeiffer"
    seed = 0
    config = '{"mnli": 0.1, "qqp": 0.5}'
    paths = map_omega_grid(config, seed, adapter_type)
    print(paths.items())
