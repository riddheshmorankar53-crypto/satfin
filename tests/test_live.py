from datetime import datetime

from satfin.live import complete_scans


def test_complete_scans_waits_for_all_segments():
    fl = "AHI-L1b-FLDK/2026/09/27/0600/HS_H09_20260927_0600_B13_FLDK_R20_S{:02d}10.DAT.bz2"
    tg = "AHI-L1b-Target/2026/09/27/0600/HS_H09_20260927_0600_B13_R30{}_R20_S0101.DAT.bz2"
    partial = [fl.format(s) for s in range(1, 10)]  # segment 10 not uploaded yet
    assert complete_scans(partial, hima=True) == []
    assert len(complete_scans(partial + [fl.format(10)], hima=True)[0][1]) == 10
    scans = complete_scans([tg.format(2), tg.format(1)], hima=True)
    assert [t for t, _ in scans] == [datetime(2026, 9, 27, 6, 0), datetime(2026, 9, 27, 6, 2, 30)]
