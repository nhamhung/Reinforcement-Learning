from datetime import datetime
from functools import total_ordering
import pandas as pd

wt_map = {
    "L": 0,
    "M": 1,
    "H": 2,
    "X": 3,
    "U": 4
}


@total_ordering
class Container:
    def __init__(self, df_row, n_days_from_snap_dt=3):
        self.n_days_from_snap_dt = n_days_from_snap_dt
        self.name = df_row['CNTR_N']
        self.row = df_row['row']
        self.level = df_row['level']
        self.container_name = df_row['CNTR_N']
        self.vessel = df_row['vsl2']
        self.snap_dt = self.convert_datetime(df_row['SNAP_DT'])
        self.etb = self.convert_datetime(df_row['etb2'])
        self.etu = self.convert_datetime(df_row['etu2'])
        self.port_mark = ''
        self.size = 0
        self.category = ''
        self.weight = 0
        self._process_pscw(df_row['pscw'])
        self.is_freeze = False
        self.container_short_name = self.name.split(" ")[0]

    def _process_pscw(self, pscw):
        ptmk, sz, cat, wt = pscw.split("_")
        self.port_mark, self.size, self.category, self.weight = ptmk, int(
            sz), cat, wt_map[wt]

    def get_days_from_snap_dt(self):
        return int((self.etb - self.snap_dt).total_seconds() // 3600)

    def is_near_snap_dt(self):
        return self.get_days_from_snap_dt() < self.n_days_from_snap_dt

    def get_etb_hours_from_snap_dt(self):
        time_delta = self.etb - self.snap_dt
        total_seconds = time_delta.total_seconds()
        return int(total_seconds // 3600)

    def get_etu_hours_from_snap_dt(self):
        time_delta = self.etu - self.snap_dt
        total_seconds = time_delta.total_seconds()
        return int(total_seconds // 3600)

    def convert_datetime(self, dt):
        if isinstance(dt, pd.Timestamp):
            return dt.to_pydatetime()
        elif isinstance(dt, str):
            return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")

    def equal_without_weight(self, other):
        return self == other

    def equal_with_weight(self, other):
        return self == other and self.weight == other.weight

    def __eq__(self, other):
        return self.vessel == other.vessel and self.port_mark == other.port_mark and self.size == other.size and self.category == other.category

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return self.etb > other.etb

    def __str__(self):
        return f"{self.vessel}_{self.get_etb_hours_from_snap_dt()}:{self.get_etu_hours_from_snap_dt()}_{self.port_mark}_{self.size}_{self.category}_{self.weight}"

    def __repr__(self):
        return f"{self.vessel}_{self.get_etb_hours_from_snap_dt()}:{self.get_etu_hours_from_snap_dt()}_{self.port_mark}_{self.size}_{self.category}_{self.weight}"

    def short_name(self):
        # return f"Container {self.name} - weight {list(wt_map.keys())[list(wt_map.values()).index(self.weight)]}"
        return f"{self.vessel}_{self.get_etb_hours_from_snap_dt()}:{self.get_etu_hours_from_snap_dt()}_{self.port_mark}_{self.size}_{self.category}"

    def ui_repr(self):
        is_freeze_text = "_(F)" if self.is_freeze else ""
        return f"{self.get_etb_hours_from_snap_dt()}:{self.get_etu_hours_from_snap_dt()}_{self.port_mark}_{self.size}_{self.category}_{list(wt_map.keys())[list(wt_map.values()).index(self.weight)]}{is_freeze_text}"
