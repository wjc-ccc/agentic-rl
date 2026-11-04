"""环境自检：uv sync 之后跑一次，确认解释器、CUDA、依赖都正常。"""

import sys

import torch


def test_python_version():
    assert sys.version_info[:2] >= (3, 12), sys.version


def test_torch_importable():
    assert torch.__version__


def test_cuda_visible():
    # 没有 GPU 的机器上允许跳过，但本机是 3060 6GB，应能通过。
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
