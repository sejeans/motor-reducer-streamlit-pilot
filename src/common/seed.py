"""전역 시드 고정 모듈.

NFR-03 재현성 요구사항: 동일 seed로 set_global_seed를 두 번 호출하면
이후 np.random.rand() 결과가 완전히 동일해야 한다.
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """random / numpy / PYTHONHASHSEED 를 일괄 고정한다.

    PYTHONHASHSEED 는 이미 인터프리터가 떠 있는 경우 set 자체로는 효과가 없지만,
    이후 spawn 되는 자식 프로세스(예: joblib n_jobs=-1)에는 전파된다.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
