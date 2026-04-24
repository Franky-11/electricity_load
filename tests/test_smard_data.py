import pandas as pd

from power_forecast.smard_data import load_smard_api


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def test_load_smard_api_parses_hourly_series(monkeypatch):
    block_start = int(pd.Timestamp("2025-12-31 00:00", tz="UTC").timestamp() * 1000)
    points = [
        [int(pd.Timestamp("2025-12-31 23:00", tz="UTC").timestamp() * 1000), 10.0],
        [int(pd.Timestamp("2026-01-01 00:00", tz="UTC").timestamp() * 1000), 20.0],
        [int(pd.Timestamp("2026-01-01 01:00", tz="UTC").timestamp() * 1000), 30.0],
    ]

    def fake_get(url, timeout):
        assert timeout == 30
        if url.endswith("index_hour.json"):
            return FakeResponse({"timestamps": [block_start]})
        return FakeResponse({"series": points})

    monkeypatch.setattr("power_forecast.smard_data.requests.get", fake_get)

    s = load_smard_api(start="2026-01-01 00:00", end="2026-01-01 02:00", tz="Europe/Berlin")

    assert s.name == "Netzlast_MW"
    assert str(s.index.tz) == "Europe/Berlin"
    assert list(s.values) == [10.0, 20.0, 30.0]
    assert list(s.index.hour) == [0, 1, 2]
