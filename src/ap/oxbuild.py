from typing import List
import re

def calc_oxbuild_ap(searched_results: List[str], input_image: str):
    PATTERN = re.compile(r'_\d+')
    slices = re.split(PATTERN, input_image)
    cls = slices[0]
    p = 0
    cnt = 0
    for i, it, in enumerate(searched_results):
        if it.startswith(cls):
            cnt += 1
            p += cnt / (i + 1)
    if cnt == 0:
        return 0
    p /= cnt
    return p

