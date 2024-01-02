from src.map.map_clip import calc_clip_map
from src.map.map_sift import calc_sift_map
from src.ap.holiday import calc_holiday_ap
from src.ap.oxbuild import calc_oxbuild_ap

calc_clip_map('oxbuild', 5, calc_oxbuild_ap)
calc_clip_map('holiday', 5, calc_holiday_ap)

calc_sift_map('oxbuild', 5, calc_oxbuild_ap)
calc_sift_map('holiday', 5, calc_holiday_ap)

