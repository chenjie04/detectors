"""提供解析配置文件相关功能的模块。

此模块在本配置系统中主要用于:
1. 解析配置文件
2. 提供点符号访问功能
"""

import os
import sys
import importlib.util


class Config(dict):
    """
    A dictionary that allows access to its items using dot notation
    (attributes) in addition to the standard dictionary key access.

    This class is designed to load configuration from Python files and
    provide a convenient way to access nested configuration parameters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = Config(v)
            elif isinstance(v, list):
                self[k] = [
                    Config(item) if isinstance(item, dict) else item for item in v
                ]

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from exc

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            super().__setitem__(key, Config(value))
        elif isinstance(value, list):
            super().__setitem__(
                key,
                [Config(item) if isinstance(item, dict) else item for item in value],
            )
        else:
            super().__setitem__(key, value)

    @classmethod
    def fromfile(cls, filename):
        """
        Loads configuration from a Python file.

        Args:
            filename (str): The path to the configuration file.

        Returns:
            Config: A Config object containing the loaded configuration.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Config file not found: {filename}")

        spec = importlib.util.spec_from_file_location("config_module", filename)
        if spec is None:
            raise ImportError(f"Could not load spec for {filename}")
        config_module = importlib.util.module_from_spec(spec)
        sys.modules["config_module"] = config_module
        spec.loader.exec_module(config_module)

        cfg_dict = {
            k: v for k, v in config_module.__dict__.items() if not k.startswith("__")
        }
        return cls(cfg_dict)

    def __repr__(self):
        return f"Config({super().__repr__()})"

    def __str__(self):
        return str(dict(self))

    def to_dict(self):
        """
        Converts the Config object and its nested Config objects to a regular dictionary.
        """
        result = {}
        for k, v in self.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result

    def copy(self):
        """
        Returns a shallow copy of the Config object.
        """
        return Config(super().copy())

    def update(self, other=None, **kwargs):
        """
        Updates the Config object with key-value pairs from another dictionary or keyword arguments.
        """
        if other:
            for k, v in other.items():
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def pretty_text(self, indent=0):
        s = ''
        for k, v in self.items():
            if isinstance(v, Config):
                s += ' ' * indent + str(k) + ':\n'
                s += v.pretty_text(indent + 4)
            elif isinstance(v, list):
                s += ' ' * indent + str(k) + ':\n'
                for item in v:
                    if isinstance(item, Config):
                        s += item.pretty_text(indent + 4)
                    else:
                        s += ' ' * (indent + 4) + str(item) + '\n'
            else:
                s += ' ' * indent + str(k) + ': ' + str(v) + '\n'
        return s
