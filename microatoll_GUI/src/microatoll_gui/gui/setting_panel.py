# src/gui/setting_panel.py
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox, QGridLayout, QLabel, QSizePolicy, QWidget, QFormLayout, QDoubleSpinBox, QSpinBox,
    QHBoxLayout, QPushButton, QVBoxLayout,
)

# SimParams はシミュレーション側の定義を参照
from microatoll_gui.simulator.simulator import SimParams

RES_SPACING_M = {"Very Fine": 0.001, "Fine": 0.002, "Medium": 0.005, "Coarse": 0.01, "Very Coarse": 0.02}


class SettingsPanel(QWidget):
    """
    下段のパラメータセッティングパネル（4列グリッド：Label|Input|Label|Input）
      - Polyline resolution（単独行）
      - Growth rate | Base Height (BH)
      - T0 (initial time) | T1 (end time)
      - Δt | Record every
      - Initialize / Apply / Run ボタン
    値の確定（編集完了 or Apply）で parametersChanged(SimParams) を送出します。
    """

    parametersChanged = Signal(SimParams)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # --- Widgets ---
        # 解像度（頂点間隔）
        self.resolution = QComboBox()
        self.resolution.addItems(["Very Fine", "Fine", "Medium", "Coarse", "Very Coarse"])
        self.resolution.setCurrentText("Medium")

        self.growth = QDoubleSpinBox()
        self.growth.setRange(0.0, 50.0)
        self.growth.setDecimals(2)
        self.growth.setValue(10.0)
        self.growth.setSuffix(" mm/yr")

        self.base_height = QDoubleSpinBox()
        self.base_height.setRange(-1e3, 1e3)
        self.base_height.setDecimals(3)
        self.base_height.setValue(0.0)
        self.base_height.setSingleStep(0.01)
        self.base_height.setSuffix(" m")

        self.t0 = QDoubleSpinBox()
        self.t0.setRange(-1e9, 1e9)
        self.t0.setDecimals(2)
        self.t0.setValue(0.0)
        self.t0.setSuffix(" yr")

        self.t1 = QDoubleSpinBox()
        self.t1.setRange(-1e9, 1e9)
        self.t1.setDecimals(2)
        self.t1.setValue(25.0)
        self.t1.setSuffix(" yr")

        self.dt = QDoubleSpinBox()
        self.dt.setRange(0.01, 10.0)
        self.dt.setDecimals(2)
        self.dt.setValue(0.1)
        self.dt.setSuffix(" yr")

        self.record_every = QDoubleSpinBox()
        self.record_every.setRange(0.1, 1e6)
        self.record_every.setDecimals(2)
        self.record_every.setValue(1.0)
        self.record_every.setSuffix(" yr")

        self.initial_size = QDoubleSpinBox()
        self.initial_size.setRange(0.01, 1000.0)
        self.initial_size.setDecimals(3)
        self.initial_size.setValue(0.2)      # 既定値：0.2 m
        self.initial_size.setSingleStep(0.01)
        self.initial_size.setSuffix(" m")

        #self.apply_btn = QPushButton("Apply")
        self.run_btn = QPushButton("Run")
        self.init_btn = QPushButton("Initialize")

        # 入力欄は横に広がるように
        for w in (self.resolution, self.growth, self.base_height, self.t0, self.t1, self.dt, self.record_every, self.initial_size):
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # --- Layout: 4-column grid (L1 | W1 | L2 | W2) ---
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        def L(text: str) -> QLabel:
            lab = QLabel(text)
            lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # ラベルは右寄せで列頭を揃える
            return lab

        row = 0
        # 1行目：Resolution は入力を3列ぶち抜き（見栄え優先）
        grid.addWidget(L("Polyline resolution:"), row, 0)
        grid.addWidget(self.resolution, row, 1, 1, 3)  # (row, col, rowspan, colspan)
        row += 1

        grid.addWidget(L("Initial size (radius):"), row, 0)
        grid.addWidget(self.initial_size, row, 1, 1, 3)
        row += 1

        # 2行目：Growth | BH
        grid.addWidget(L("Growth speed:"),     row, 0)
        grid.addWidget(self.growth,            row, 1)
        grid.addWidget(L("Base Height:"), row, 2)
        grid.addWidget(self.base_height,       row, 3)
        row += 1

        # 3行目：T0 | T1
        grid.addWidget(L("T0 (initial time):"), row, 0)
        grid.addWidget(self.t0,                 row, 1)
        grid.addWidget(L("T1 (end time):"),     row, 2)
        grid.addWidget(self.t1,                 row, 3)
        row += 1

        # 4行目：Δt | Record every
        grid.addWidget(L("Δt:"),               row, 0)
        grid.addWidget(self.dt,                row, 1)
        grid.addWidget(L("Record every:"),     row, 2)
        grid.addWidget(self.record_every,      row, 3)
        row += 1

        # 列ストレッチ：入力列(1,3)を広げ、ラベル列(0,2)は内容幅に合わせる
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)
        grid.setColumnStretch(3, 1)

        # ボタン行
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.init_btn)
        #buttons.addWidget(self.apply_btn)
        buttons.addWidget(self.run_btn)

        root = QVBoxLayout(self)
        root.addLayout(grid)
        root.addLayout(buttons)

        # --- Signals ---
        #self.apply_btn.clicked.connect(self._emit_params)
        for w in (self.growth, self.base_height, self.t0, self.t1, self.dt, self.record_every, self.initial_size):
            w.editingFinished.connect(self._emit_params)
        self.resolution.currentIndexChanged.connect(self._emit_params)

    def set_time_window(self, t0: int, t1: int) -> None:
        """Programmatically set T0/T1 (years) without firing multiple signals."""
        # Temporarily block signals from widgets to avoid duplicate emits
        self.t0.blockSignals(True)
        self.t1.blockSignals(True)
        try:
            self.t0.setValue(int(t0))
            self.t1.setValue(int(t1))
        finally:
            self.t0.blockSignals(False)
            self.t1.blockSignals(False)
        # Emit once with updated params so upper layers stay in sync
        self._emit_params()

    # ---- Public API ----
    def current_params(self) -> SimParams:
        """現在のUI値から SimParams を組み立てて返す。"""
        spacing = RES_SPACING_M.get(self.resolution.currentText(), 0.05)
        return SimParams(
            growth_rate_mm_yr=float(self.growth.value()),
            base_height=float(self.base_height.value()),
            vertex_spacing_m=float(spacing),
            t0_years=float(self.t0.value()),
            t1_years=float(self.t1.value()),
            dt_years=float(self.dt.value()),
            record_every_years=float(self.record_every.value()),
            initial_size_m=float(self.initial_size.value()), 
        )

    def get_bh(self) -> float:
        """Base Height (BH) のショートカット取得。"""
        return float(self.base_height.value())

    # ---- Internal ----
    @Slot()
    def _emit_params(self) -> None:
        self.parametersChanged.emit(self.current_params())
