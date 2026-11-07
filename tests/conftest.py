"""pytest 启动自举：保证被测包可导入，不依赖 .venv 内 .pth 的状态。

背景：仓库位于 iCloud 同步目录，`.venv` 内的 `.pth` 可能被系统加上 UF_HIDDEN
标志（CPython `site.addpackage` 会跳过 hidden .pth）。conftest 在收集测试
模块之前执行，因此这里直接把 ``src/`` 放进 ``sys.path``，一次生效、幂等。
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
