from typing import List

def calc_holiday_ap(search_results: List[str], input_image: str):
    cls = input_image[:4]
    p = 0
    cnt = 0
    for i, it in enumerate(search_results):
        if it.startswith(cls):
            cnt += 1
            p += cnt / (i + 1)

    if cnt == 0:
        return 0
    p /= cnt
    return p

