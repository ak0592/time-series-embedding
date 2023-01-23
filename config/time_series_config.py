import yaml

from typing import Dict, Union


class TimeSeriesConfig:

    @staticmethod
    def load_params(file_path: str) -> Dict[str, Union[int, str]]:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file.read())
        except ImportError as e:
            print(f'{type(e)} {file_path} is not found.')


# make data_key from yaml parameters
def param_to_data_key(yaml_key: str, param_dict: Dict[str, Union[int, str]]) -> str:
    if not isinstance(yaml_key, str):
        raise TypeError(f'{yaml_key} type must be string, but it is {type(yaml_key)}.')
    if not isinstance(param_dict, dict):
        raise TypeError(f'{param_dict} type must be dict, but it is {type(param_dict)}.')
    class_key = yaml_key
    for key in param_dict.keys():
        class_key += f'_{key}_{param_dict[key]}'

    return class_key
