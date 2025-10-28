# src/gui/setting_panel.py
from __future__ import annotations

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QComboBox, QWidget, QFormLayout, QDoubleSpinBox, QSpinBox,
    QHBoxLayout, QPushButton, QVBoxLayout,
)

# SimParams はシミュレーション側の定義を参照
from simulator.simulator import SimParams

RES_SPACING_M = {"High": 0.01, "Medium": 0.05, "Low": 0.1}

class SettingsPanel(QWidget):
    """
    下段のパラメータセッティングパネル。
    - Growth rate, Tidal range, Max elevation, Base Height (BH), Δt, Steps
    - Apply/Run ボタン
    値の確定（編集完了 or Apply）で parametersChanged(SimParams) を送出します。
    """

    parametersChanged = Signal(SimParams)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Resolution (vertex spacing)
        self.resolution = QComboBox()
        self.resolution.addItems(["High", "Medium", "Low"])
        self.resolution.setCurrentText("Low")

        # --- Widgets ---
        self.growth = QDoubleSpinBox()
        self.growth.setRange(0.0, 50.0)
        self.growth.setDecimals(2)
        self.growth.setValue(10.0)
        self.growth.setSuffix(" mm/yr")

        self.tidal = QDoubleSpinBox()
        self.tidal.setRange(0.0, 10.0)
        self.tidal.setDecimals(2)
        self.tidal.setValue(2.7)
        self.tidal.setSuffix(" m")

        self.max_elev = QDoubleSpinBox()
        self.max_elev.setRange(0.0, 10.0)
        self.max_elev.setDecimals(3)
        self.max_elev.setValue(1.5)
        self.max_elev.setSuffix(" m")

        # NEW: Base Height (BH)
        self.base_height = QDoubleSpinBox()
        self.base_height.setRange(-1e3, 1e3)
        self.base_height.setDecimals(3)
        self.base_height.setValue(0.0)
        self.base_height.setSingleStep(0.01)
        self.base_height.setSuffix(" m")

        self.dt = QDoubleSpinBox()
        self.dt.setRange(0.01, 10.0)
        self.dt.setDecimals(2)
        self.dt.setValue(1.0)
        self.dt.setSuffix(" yr")

        self.steps = QSpinBox()
        self.steps.setRange(1, 10_000)
        self.steps.setValue(50)

        self.apply_btn = QPushButton("Apply")
        self.run_btn = QPushButton("Run")
        self.init_btn = QPushButton("Initialize")

        # --- Layout ---
        form = QFormLayout()
        form.addRow("Polyline resolution:", self.resolution)
        form.addRow("Growth rate:", self.growth)
        form.addRow("Tidal range:", self.tidal)
        form.addRow("Max elevation:", self.max_elev)
        form.addRow("Base Height (BH):", self.base_height)
        form.addRow("Δt:", self.dt)
        form.addRow("Steps:", self.steps)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.apply_btn)
        buttons.addWidget(self.run_btn)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addLayout(buttons)

        # --- Signals ---
        self.apply_btn.clicked.connect(self._emit_params)
        for w in (self.growth, self.tidal, self.max_elev, self.base_height, self.dt, self.steps):
            w.editingFinished.connect(self._emit_params)
        self.resolution.currentIndexChanged.connect(self._emit_params)

    # ---- Public API ----
    def current_params(self) -> SimParams:
        """現在のUI値から SimParams を組み立てて返す。"""
        spacing = RES_SPACING_M.get(self.resolution.currentText(), 0.05)
        return SimParams(
            growth_rate_mm_yr=float(self.growth.value()),
            tidal_range_m=float(self.tidal.value()),
            max_elevation_m=float(self.max_elev.value()),
            base_height=float(self.base_height.value()),
            vertex_spacing_m=float(spacing),
            dt_years=float(self.dt.value()),
            n_steps=int(self.steps.value()),
        )

    def get_bh(self) -> float:
        """Base Height (BH) のショートカット取得。"""
        return float(self.base_height.value())

    # ---- Internal ----
    @Slot()
    def _emit_params(self) -> None:
        self.parametersChanged.emit(self.current_params())
