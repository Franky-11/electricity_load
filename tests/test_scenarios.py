import pandas as pd

from power_forecast.scenarios import event_days, mult_holiday_weekend, shift_load, temp_adjust


def test_shift_load_preserves_daily_energy():
    idx = pd.date_range("2026-01-02 00:00", periods=24, freq="h", tz="Europe/Berlin")
    y = pd.Series(100.0, index=idx)

    shifted = shift_load(y, frac=0.1, src_hours=[18, 19], dst_hours=[2, 3])

    assert shifted.sum() == y.sum()
    assert shifted.loc["2026-01-02 18:00"] == 90.0
    assert shifted.loc["2026-01-02 02:00"] == 110.0


def test_holiday_weekend_multiplier_applies_holiday_factor():
    idx = pd.date_range("2026-01-01 00:00", periods=24, freq="h", tz="Europe/Berlin")
    y = pd.Series(100.0, index=idx)

    adjusted = mult_holiday_weekend(y, holidays={pd.Timestamp("2026-01-01").date()}, hol_mult=0.9, weekend_mult=0.95)

    assert adjusted.eq(90.0).all()


def test_event_days_scale_only_configured_dates():
    idx = pd.date_range("2026-01-01 00:00", periods=48, freq="h", tz="Europe/Berlin")
    y = pd.Series(100.0, index=idx)

    adjusted = event_days(y, ["2026-01-02"], mult=1.2)

    assert adjusted.loc["2026-01-01"].eq(100.0).all()
    assert adjusted.loc["2026-01-02"].eq(120.0).all()


def test_temperature_adjustment_is_multiplicative():
    idx = pd.date_range("2026-01-01 00:00", periods=2, freq="h", tz="Europe/Berlin")
    y = pd.Series([100.0, 200.0], index=idx)

    adjusted = temp_adjust(y, delta_c=2.0, k_perc_per_c=0.01)

    assert list(adjusted.values) == [102.0, 204.0]
