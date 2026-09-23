from datetime import datetime

from satfin.data.download import parse_start_time


def test_parse_start_time():
    f = "OR_ABI-L1b-RadM1-M6C13_G19_s20251521800279_e20251521800349_c20251521800390.nc"
    assert parse_start_time(f) == datetime(2025, 6, 1, 18, 0, 27)
