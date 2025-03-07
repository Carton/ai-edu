# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""
Version 2.0
what's new:
- add InitialMethod
"""

from enum import Enum

class NetType(Enum):
    Fitting = 1,
    BinaryClassifier = 2,
    MultipleClassifier = 3

class InitialMethod(Enum):
    Zero = 0,  # 全零初始化
    Normal = 1,  # 正态分布初始化
    Xavier = 2,  # Xavier初始化
    MSRA = 3  # MSRA初始化

