"""测试配置文件解析功能的模块

本模块用于测试配置系统的各项功能，包括:
- 配置文件加载
- 配置访问方法
- 递归解析测试
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import unittest
from engine.config.config import Config


class TestConfig(unittest.TestCase):
    """配置系统测试类
    
    本测试类用于验证配置系统的核心功能，包括：
    - 配置文件的加载和解析
    - 配置项的访问和遍历
    - 配置的格式化输出
    - 递归解析配置项
    """
    def setUp(self):
        self.config_file_path = (
            "/home/chenjie04/workstation/detectors/configs/sod_detr/sod_detr_n_500e_coco.py"
        )
        self.cfg = Config.fromfile(self.config_file_path)

    def test_config_loading_and_keys(self):
        """测试配置文件加载和键值访问功能
        
        本测试用例验证:
        - 配置文件是否成功加载为Config对象
        - 配置对象是否包含有效的键值
        """
        print(f"\nConfiguration loaded: {self.cfg}")
        print(f"Keys of cfg: {self.cfg.keys()}")
        self.assertIsInstance(self.cfg, Config)
        self.assertGreater(len(self.cfg.keys()), 0)

    def test_config_parsing(self, config_obj=None, path=""):
        """测试配置项的递归解析功能
        
        本测试用例通过递归遍历配置对象的所有层级，验证:
        - 所有配置项是否都可以正确访问
        - 嵌套的字典和列表是否能被正确解析
        - 配置项的路径构建是否正确
        
        Args:
            config_obj: 待解析的配置对象，默认为None时使用self.cfg
            path: 当前配置项的路径字符串，用于构建完整的访问路径
        """
        if config_obj is None:
            config_obj = self.cfg

        for key, value in config_obj.items():
            full_key = f"{path}.{key}" if path else key
            try:
                _ = getattr(config_obj, key)
                print(f"Successfully accessed: {full_key}")
                if isinstance(value, dict):
                    self.test_config_parsing(value, full_key)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            self.test_config_parsing(item, f"{full_key}[{i}]")
            except (AttributeError, KeyError) as e:
                self.fail(f"Error accessing {full_key}: {e}")

    def test_pretty_text_output(self):
        """测试配置的格式化文本输出功能
        
        本测试用例验证:
        - 配置对象是否能正确转换为格式化文本
        - 输出的文本是否为有效的字符串
        - 输出文本是否包含实际内容
        """
        print("\nTesting pretty_text output:")
        pretty_output = self.cfg.pretty_text()
        print(pretty_output)
        # Add assertions here to validate the format of pretty_output
        # For example, check for expected substrings or line counts
        self.assertIsInstance(pretty_output, str)
        self.assertGreater(len(pretty_output), 0)


if __name__ == '__main__':
    unittest.main(exit=False)
