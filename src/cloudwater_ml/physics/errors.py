"""物理约束模块的异常类型。

沿用 ``cloudwater_ml.utils.exceptions.PhysicsConstraintError``（物理约束定义、
单位口径或组合方式非法）作为基类，另加两个本模块特有的错误：

- :class:`UnitMismatchError` —— 单位口径不符。**报错而不是默默算错**：
  物理约束一旦拿到量纲不对的输入（例如把 ``kg/m2`` 的 CWP 当成 ``kg/kg``
  的云水含量），必须立刻失败，而不是算出一个「看起来正常」的数。
- :class:`PhysicsNotReadyError` —— 物理检查尚不具备启用条件（单位口径未定
  或前置输入缺失）。它继承 ``NotImplementedError``，使「未实现」在**类型
  层面**即可被断言捕获（``pytest.raises(NotImplementedError)``），从代码结构
  上防止把预留约束误当成已实现。
- :class:`ReservedConstraintError` —— 预留约束被尝试实例化（构造即抛）。
"""
from __future__ import annotations

from cloudwater_ml.utils.exceptions import PhysicsConstraintError

__all__ = [
    "UnitMismatchError",
    "PhysicsNotReadyError",
    "ReservedConstraintError",
]


class UnitMismatchError(PhysicsConstraintError):
    """单位口径不符（量纲不一致 / 变量与单位不匹配）。

    触发场景：约束声明了期望单位，而输入的单位与之不符；或量纲不同的两个
    单位被要求互相换算（例如 ``kg/kg`` ↔ ``kg/m3``）。
    """

    exit_code = 2


class PhysicsNotReadyError(PhysicsConstraintError, NotImplementedError):
    """物理检查尚不具备启用条件（单位口径未定 / 前置输入缺失）。

    继承 ``NotImplementedError``：未实现的检查在类型层面即可被捕获，
    避免被误当作「已实现但这次没跑」。
    """

    exit_code = 2


class ReservedConstraintError(PhysicsNotReadyError):
    """预留约束被实例化（构造即抛）。

    对应既有 ``training/losses.py::CWPConsistencyPenalty`` 的做法：
    单位口径与前置输入未确认前，**不允许**把未验证的物理假设写进损失。
    """
