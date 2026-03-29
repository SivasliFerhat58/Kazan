import sys
import os
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QStatusBar, QGroupBox, QSlider,
    QCheckBox, QComboBox
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QTimer, QRect

from preprocessing import load_and_preprocess
from terrain_analysis import compute_slope_map, compute_obstacle_map
from terrain_roughness import compute_roughness_map
from crater_module import detect_craters
from risk_map import build_risk_map
from path_planner import plan_path as planner_algo
from path_smoother import smooth_path
from energy_model import compute_energy

GRID_SIZE = (200, 200)
W_SLOPE = 0.40
W_OBSTACLE = 0.25
W_CRATER = 0.20
W_ROUGH = 0.15
RISK_WEIGHT = 10.0
OBS_DARK_THRESH = 0.15
OBS_BRIGHT_THRESH = 0.90
SMOOTHING_FACTOR = 1.5
SMOOTH_OVERSAMPLE = 4
DISPLAY_SIZE = 600  

class HazardDetectionWorker(QThread):
    """
    QThread to detect hazards in the background.
    Prevents the UI from freezing during YOLO + MobileSAM inference.
    """
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, image_path, image_norm, image_uint8):
        super().__init__()
        self.image_path = image_path
        self.image_norm = image_norm
        self.image_uint8 = image_uint8

    def run(self):
        try:

            slope_map = compute_slope_map(self.image_norm)

            obstacle_map = compute_obstacle_map(
                self.image_norm,
                dark_threshold=OBS_DARK_THRESH,
                bright_threshold=OBS_BRIGHT_THRESH
            )

            roughness_map = compute_roughness_map(self.image_norm)

            crater_map = detect_craters(self.image_uint8, image_path=self.image_path)

            risk_map = build_risk_map(
                slope_map, obstacle_map, crater_map, roughness_map,
                w_slope=W_SLOPE, w_obs=W_OBSTACLE, w_crat=W_CRATER, w_rough=W_ROUGH
            )

            results = {
                "slope_map": slope_map,
                "obstacle_map": obstacle_map,
                "roughness_map": roughness_map,
                "crater_map": crater_map,
                "risk_map": risk_map
            }
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))

class MapCanvas(QLabel):
    """
    Custom QLabel to display the map, capture mouse clicks for Start/Goal,
    and draw overlays (obstacles, path).
    """
    click_event = pyqtSignal(int, int)  
    map_modified = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(DISPLAY_SIZE, DISPLAY_SIZE)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #0D0D2B; border: 2px solid #333355;")

        self.base_pixmap = None      
        self.overlay_pixmap = None   

        self.start_pos = None
        self.goal_pos = None
        self.path = None
        self.smooth_path_points = None

        self.crater_map = None
        self.obstacle_map = None
        self.risk_map = None
        self.risk_pixmap = None
        self.show_risk_overlay = False
        self.slope_map = None
        self.manual_hazard_map = np.zeros(GRID_SIZE, dtype=np.float32)
        self.is_painting_hazard = False

        self.anim_max_index = 0
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.on_anim_step)

    def mousePressEvent(self, event):
        if self.base_pixmap is None:
            return

        if event.button() == Qt.LeftButton:
            x = event.pos().x()
            y = event.pos().y()

            logical_x = int((x / DISPLAY_SIZE) * GRID_SIZE[0])
            logical_y = int((y / DISPLAY_SIZE) * GRID_SIZE[1])

            self.click_event.emit(logical_x, logical_y)

    def mouseMoveEvent(self, event):
        if self.is_painting_hazard and event.buttons() & Qt.LeftButton:
            x = event.pos().x()
            y = event.pos().y()
            if 0 <= x < DISPLAY_SIZE and 0 <= y < DISPLAY_SIZE:
                lx = int((x / DISPLAY_SIZE) * GRID_SIZE[0])
                ly = int((y / DISPLAY_SIZE) * GRID_SIZE[1])

                self.manual_hazard_map[max(0, ly-1):min(GRID_SIZE[1], ly+2), max(0, lx-1):min(GRID_SIZE[0], lx+2)] = 1.0
                self.redraw()
                self.map_modified.emit()

    def update_base_image(self, bgr_image):
        """Converts an OpenCV BGR image (200x200) to QPixmap and scales it."""
        h, w, c = bgr_image.shape
        bytes_per_line = c * w
        qImg = QImage(bgr_image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        raw_pixmap = QPixmap.fromImage(qImg)
        self.base_pixmap = raw_pixmap.scaled(DISPLAY_SIZE, DISPLAY_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.redraw()

    def set_overlays(self, crater_map, obstacle_map, risk_map=None, slope_map=None):
        self.crater_map = crater_map
        self.obstacle_map = obstacle_map
        self.slope_map = slope_map

        if risk_map is not None:
            self.risk_map = risk_map

            self.risk_pixmap = self.generate_risk_pixmap(risk_map)

        self.redraw()

    def generate_risk_pixmap(self, risk_map):
        h, w = risk_map.shape
        risk_img = QImage(w, h, QImage.Format_ARGB32)
        for y in range(h):
            for x in range(w):
                r_val = risk_map[y, x]

                red = int(255 * r_val)
                green = int(255 * (1 - r_val))
                risk_img.setPixelColor(x, y, QColor(red, green, 0, 140))

        return QPixmap.fromImage(risk_img).scaled(DISPLAY_SIZE, DISPLAY_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def set_show_risk_overlay(self, show):
        self.show_risk_overlay = show
        self.redraw()

    def display_elements(self, start, goal, path, smooth_path_points):
        self.start_pos = start
        self.goal_pos = goal
        self.path = path
        self.smooth_path_points = smooth_path_points
        self.anim_max_index = 0
        if self.smooth_path_points:
            self.anim_timer.start(25)  
        else:
            self.anim_timer.stop()
            self.redraw()

    def on_anim_step(self):
        if self.smooth_path_points and self.anim_max_index < len(self.smooth_path_points):
            self.anim_max_index += 1

            head = self.smooth_path_points[self.anim_max_index - 1]
            hx, hy = int(head[0]), int(head[1])
            local_slope = float(self.slope_map[hy, hx]) if self.slope_map is not None else 0.0

            base_interval = 35
            speed_penalty = int(local_slope * 150) 
            new_interval = max(15, min(200, base_interval + speed_penalty))
            self.anim_timer.setInterval(new_interval)

            self.redraw()
        else:
            self.anim_timer.stop()

    def redraw(self):
        if self.base_pixmap is None:
            return

        combined = self.base_pixmap.copy()
        painter = QPainter(combined)
        painter.setRenderHint(QPainter.Antialiasing)

        scale_x = DISPLAY_SIZE / GRID_SIZE[0]
        scale_y = DISPLAY_SIZE / GRID_SIZE[1]

        if self.obstacle_map is not None and self.crater_map is not None:

            crater_y, crater_x = np.where(self.crater_map > 0.5)
            obs_y, obs_x = np.where(self.obstacle_map > 0.5)

            painter.setPen(Qt.NoPen)

            painter.setBrush(QColor(255, 0, 0, 100))
            for x, y in zip(crater_x, crater_y):
                 painter.drawRect(int(x * scale_x), int(y * scale_y), int(scale_x), int(scale_y))

            painter.setBrush(QColor(255, 255, 0, 100))
            for x, y in zip(obs_x, obs_y):
                 painter.drawRect(int(x * scale_x), int(y * scale_y), int(scale_x), int(scale_y))

        if self.manual_hazard_map is not None:
            hz_y, hz_x = np.where(self.manual_hazard_map > 0.5)
            painter.setBrush(QColor(128, 0, 128, 140)) 
            for x, y in zip(hz_x, hz_y):
                painter.drawRect(int(x * scale_x), int(y * scale_y), int(scale_x), int(scale_y))

        if self.smooth_path_points:
            pen = QPen(QColor(0, 255, 255), 3)  
            painter.setPen(pen)

            draw_len = min(self.anim_max_index, len(self.smooth_path_points))

            for i in range(draw_len - 1):
                p1 = self.smooth_path_points[i]
                p2 = self.smooth_path_points[i+1]
                painter.drawLine(
                    int(p1[0] * scale_x), int(p1[1] * scale_y),
                    int(p2[0] * scale_x), int(p2[1] * scale_y)
                )

            if draw_len > 0:
                head = self.smooth_path_points[draw_len - 1]
                painter.setBrush(QColor(255, 165, 0)) 
                painter.setPen(QPen(Qt.black, 1))
                painter.drawEllipse(QPoint(int(head[0] * scale_x), int(head[1] * scale_y)), 6, 6)

        if self.show_risk_overlay and self.risk_pixmap is not None:
             painter.drawPixmap(0, 0, self.risk_pixmap)

        if self.start_pos:
            painter.setBrush(QColor(0, 255, 136)) 
            painter.setPen(QPen(Qt.black, 2))
            cx, cy = int(self.start_pos[0] * scale_x), int(self.start_pos[1] * scale_y)
            painter.drawEllipse(QPoint(cx, cy), 8, 8)

            painter.setPen(QPen(Qt.white, 2))
            painter.setFont(QFont("Arial", 12, QFont.Bold))
            painter.drawText(cx + 10, cy + 10, "S")

        if self.goal_pos:
            painter.setBrush(QColor(255, 51, 51)) 
            painter.setPen(QPen(Qt.black, 2))
            cx, cy = int(self.goal_pos[0] * scale_x), int(self.goal_pos[1] * scale_y)
            painter.drawEllipse(QPoint(cx, cy), 8, 8)

            painter.setPen(QPen(Qt.white, 2))
            painter.setFont(QFont("Arial", 12, QFont.Bold))
            painter.drawText(cx + 10, cy + 10, "G")

        painter.end()
        self.setPixmap(combined)

class LunarApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌑 Lunar Rover Navigation - Path Planner")
        self.resize(800, 750)
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a2e; }
            QLabel { color: #ffffff; font-weight: bold; }
            QPushButton {
                background-color: #003366;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #0055aa; }
            QPushButton:disabled { background-color: #444; color: #888; }
            QGroupBox { color: #00ccff; font-weight: bold; border: 1px solid #333355; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QSlider::groove:horizontal { border: 1px solid #999; height: 8px; background: #333; margin: 2px 0; border-radius: 4px; }
            QSlider::handle:horizontal { background: #00ccff; border: 1px solid #00ccff; width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; }
            QCheckBox { color: #88ccff; font-weight: bold; }
            QComboBox { background-color: #0d0d2b; color: #00ccff; border: 1px solid #333355; padding: 5px; font-weight: bold; }
        """)

        self.image_path = None
        self.image_norm = None
        self.image_uint8 = None
        self.hazard_results = None

        self.click_mode = "START" 
        self.start_pos = None
        self.goal_pos = None
        self.path_results = None
        self.is_hazard_mode = False

        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        self.canvas = MapCanvas()
        self.canvas.click_event.connect(self.on_canvas_click)
        self.canvas.map_modified.connect(self.on_map_modified)
        main_layout.addWidget(self.canvas)

        side_panel = QWidget()
        side_panel.setFixedWidth(300)
        control_panel = QVBoxLayout(side_panel)
        main_layout.addWidget(side_panel)
        control_panel.setAlignment(Qt.AlignTop)

        btn_load = QPushButton("📂 Load Image")
        btn_load.clicked.connect(self.load_image)
        control_panel.addWidget(btn_load)

        self.btn_detect = QPushButton("⚠️ Detect Hazards")
        self.btn_detect.setEnabled(False)
        self.btn_detect.clicked.connect(self.detect_hazards)
        control_panel.addWidget(self.btn_detect)

        self.chk_risk_overlay = QCheckBox("Show Risk Heatmap")
        self.chk_risk_overlay.toggled.connect(self.canvas.set_show_risk_overlay)
        self.chk_risk_overlay.setEnabled(False)
        control_panel.addWidget(self.chk_risk_overlay)

        self.hz_group = QGroupBox("🚧 Hazard Injection Tool")
        hz_layout = QVBoxLayout()
        self.btn_paint_mode = QPushButton("🖌️ Paint Hazards")
        self.btn_paint_mode.setCheckable(True)
        self.btn_paint_mode.clicked.connect(self.toggle_hazard_mode)
        self.btn_clear_manual = QPushButton("🧹 Clear Manual Hazards")
        self.btn_clear_manual.clicked.connect(self.clear_manual_hazards)
        hz_layout.addWidget(self.btn_paint_mode)
        hz_layout.addWidget(self.btn_clear_manual)
        self.hz_group.setLayout(hz_layout)
        self.hz_group.setEnabled(False)
        control_panel.addWidget(self.hz_group)

        sel_group = QGroupBox("Target Selection")
        sel_layout = QVBoxLayout()

        self.btn_set_start = QPushButton("🏁 Set Start (S)")
        self.btn_set_start.setStyleSheet("background-color: #008855; color: white; border-radius: 5px; padding: 10px; font-weight: bold;")
        self.btn_set_start.clicked.connect(lambda: self.set_click_mode("START"))

        self.btn_set_goal = QPushButton("🚩 Set Goal (G)")
        self.btn_set_goal.setStyleSheet("background-color: #AA2222; color: white; border-radius: 5px; padding: 10px; font-weight: bold;")
        self.btn_set_goal.clicked.connect(lambda: self.set_click_mode("GOAL"))

        self.lbl_start = QLabel("Start: None")
        self.lbl_goal = QLabel("Goal: None")

        sel_layout.addWidget(self.btn_set_start)
        sel_layout.addWidget(self.btn_set_goal)
        sel_layout.addWidget(self.lbl_start)
        sel_layout.addWidget(self.lbl_goal)
        sel_group.setLayout(sel_layout)
        sel_group.setEnabled(False)
        self.sel_group = sel_group
        control_panel.addWidget(sel_group)

        self.pref_group = QGroupBox("Pathfinding Strategy")
        pref_layout = QVBoxLayout()

        self.combo_strategy = QComboBox()
        self.combo_strategy.addItems(["Balanced (Ideal)", "Maximum Safety (Avoid Craters)", "Fastest (Energy Saving)"])
        self.combo_strategy.currentIndexChanged.connect(self.on_strategy_changed)
        pref_layout.addWidget(self.combo_strategy)

        self.lbl_slope = QLabel("Slope Avoidance: 40%")
        self.slider_slope = QSlider(Qt.Horizontal)
        self.slider_slope.setRange(0, 100)
        self.slider_slope.setValue(40)
        self.slider_slope.valueChanged.connect(self.on_param_changed)

        self.lbl_crater = QLabel("Kraterden Kaçınma (Crater): 20%")
        self.slider_crater = QSlider(Qt.Horizontal)
        self.slider_crater.setRange(0, 100)
        self.slider_crater.setValue(20)
        self.slider_crater.valueChanged.connect(self.on_param_changed)

        pref_layout.addWidget(self.lbl_slope)
        pref_layout.addWidget(self.slider_slope)
        pref_layout.addWidget(self.lbl_crater)
        pref_layout.addWidget(self.slider_crater)
        self.pref_group.setLayout(pref_layout)
        self.pref_group.setEnabled(False)
        control_panel.addWidget(self.pref_group)

        self.stats_group = QGroupBox("Mission Statistics")
        stats_layout = QVBoxLayout()
        self.lbl_stat_energy = QLabel("Energy: ---")
        self.lbl_stat_dist = QLabel("Distance: ---")
        self.lbl_stat_slope = QLabel("Uphill/Downhill: ---")
        stats_layout.addWidget(self.lbl_stat_energy)
        stats_layout.addWidget(self.lbl_stat_dist)
        stats_layout.addWidget(self.lbl_stat_slope)
        self.stats_group.setLayout(stats_layout)
        self.stats_group.setEnabled(False)
        control_panel.addWidget(self.stats_group)

        self.stats_group.setEnabled(False)
        control_panel.addWidget(self.stats_group)

        self.btn_plan = QPushButton("🚀 Find A* Path")
        self.btn_plan.setStyleSheet("background-color: #9900CC; border-radius: 5px; padding: 15px; font-weight: bold;")
        self.btn_plan.setEnabled(False)
        self.btn_plan.clicked.connect(self.plan_path)
        control_panel.addWidget(self.btn_plan)

        self.btn_clear = QPushButton("🗑️ Clear Path")
        self.btn_clear.setEnabled(False)
        self.btn_clear.clicked.connect(self.clear_mission)
        control_panel.addWidget(self.btn_clear)

        control_panel.addStretch(1)

        legend_group = QGroupBox("Legend")
        legend_layout = QVBoxLayout()
        lbl_crater = QLabel("🟥 Red: Crater Hazard")
        lbl_crater.setStyleSheet("color: #ff3333;")
        lbl_rocks = QLabel("🟨 Yellow: Obstacle / Hill")
        lbl_rocks.setStyleSheet("color: #ffff33;")
        lbl_path = QLabel("🟦 Cyan Line: Optimal Path")
        lbl_path.setStyleSheet("color: #00ffff;")
        legend_layout.addWidget(lbl_crater)
        legend_layout.addWidget(lbl_rocks)
        legend_layout.addWidget(lbl_path)
        legend_group.setLayout(legend_layout)
        control_panel.addWidget(legend_group)

        main_layout.addLayout(control_panel)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Welcome. Please load a lunar image.")

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load Lunar Image", "", "Images (*.png *.jpg *.tif *.tiff);;All Files (*)", options=options
        )
        if file_name:
            self.image_path = file_name
            try:
                self.statusBar.showMessage(f"Loading {file_name}...")

                self.image_norm = load_and_preprocess(self.image_path, target_size=GRID_SIZE)
                self.image_uint8 = (self.image_norm * 255).astype(np.uint8)

                display_bgr = cv2.cvtColor(self.image_uint8, cv2.COLOR_GRAY2BGR)

                self.hazard_results = None
                self.start_pos = None
                self.goal_pos = None
                self.lbl_start.setText("Start: None")
                self.lbl_goal.setText("Goal: None")
                self.canvas.display_elements(None, None, None, None)
                self.canvas.set_overlays(None, None)

                self.canvas.update_base_image(display_bgr)
                self.btn_detect.setEnabled(True)
                self.sel_group.setEnabled(False)
                self.pref_group.setEnabled(False)
                self.stats_group.setEnabled(False)
                self.hz_group.setEnabled(False)
                self.btn_plan.setEnabled(False)
                self.btn_clear.setEnabled(False)
                self.chk_risk_overlay.setEnabled(False)

                self.statusBar.showMessage("Image loaded. You can now detect hazards.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")
                self.statusBar.showMessage("Error loading image.")

    def detect_hazards(self):
        if self.image_norm is None:
            return

        self.statusBar.showMessage("Running YOLO + MobileSAM for hazard detection... Please wait.")
        self.btn_detect.setEnabled(False)

        self.worker = HazardDetectionWorker(self.image_path, self.image_norm, self.image_uint8)
        self.worker.finished.connect(self.on_hazards_detected)
        self.worker.error.connect(self.on_hazard_error)
        self.worker.start()

    def on_hazards_detected(self, results):
        self.hazard_results = results

        self.update_risk_map()

        self.sel_group.setEnabled(True)
        self.pref_group.setEnabled(True)
        self.btn_detect.setEnabled(True)
        self.chk_risk_overlay.setEnabled(True)
        self.hz_group.setEnabled(True)
        self.statusBar.showMessage("Hazards detected! Please set Start and Goal points.")

    def on_hazard_error(self, err_msg):
        QMessageBox.critical(self, "Detection Error", f"An error occurred during hazard detection:\n{err_msg}")
        self.btn_detect.setEnabled(True)
        self.statusBar.showMessage("Hazard detection failed.")

    def set_click_mode(self, mode):
        self.click_mode = mode
        if mode != "ADD_HAZARD":
            self.btn_paint_mode.setChecked(False)
            self.canvas.is_painting_hazard = False
        self.statusBar.showMessage(f"Click on the map to set the {mode.title()} point.")

    def toggle_hazard_mode(self, checked):
        if checked:
            self.click_mode = "ADD_HAZARD"
            self.canvas.is_painting_hazard = True
            self.statusBar.showMessage("🎨 Paint Mode: Click and Drag on map to add hazards!")
        else:
            self.click_mode = "START"
            self.canvas.is_painting_hazard = False
            self.statusBar.showMessage("Point Selection Mode.")

    def on_map_modified(self):

        if self.start_pos and self.goal_pos:
            self.plan_path()

    def clear_manual_hazards(self):
        self.canvas.manual_hazard_map.fill(0)
        self.canvas.redraw()
        self.on_map_modified()

    def on_canvas_click(self, x, y):

        if self.click_mode == "ADD_HAZARD":
            return 
        if self.hazard_results is not None:
            if self.hazard_results["obstacle_map"][y, x] > 0.5 or self.hazard_results["crater_map"][y, x] > 0.5:
                QMessageBox.warning(self, "Invalid Point", "You clicked on an obstacle or crater! Please click a free space.")
                return

        if self.click_mode == "START":
            self.start_pos = (x, y)
            self.lbl_start.setText(f"Start: ({x}, {y})")
        else:
            self.goal_pos = (x, y)
            self.lbl_goal.setText(f"Goal: ({x}, {y})")

        if self.click_mode == "START" and self.goal_pos is None:
            self.set_click_mode("GOAL")

        self.canvas.display_elements(self.start_pos, self.goal_pos, None, None)

        if self.start_pos and self.goal_pos:
            self.btn_plan.setEnabled(True)
            self.statusBar.showMessage("Start and Goal set. Ready to plan path.")

    def plan_path(self):
        if self.start_pos is None or self.goal_pos is None or self.hazard_results is None:
            return

        self.statusBar.showMessage("Calculating optimal A* path...")
        QApplication.processEvents() 

        try:

            w_slope = self.slider_slope.value() / 100.0
            w_crat = self.slider_crater.value() / 100.0

            risk_map = build_risk_map(
                self.hazard_results["slope_map"],
                self.hazard_results["obstacle_map"],
                self.hazard_results["crater_map"],
                self.hazard_results["roughness_map"],
                w_slope=w_slope, w_obs=W_OBSTACLE, w_crat=w_crat, w_rough=W_ROUGH
            )

            risk_map = np.maximum(risk_map, self.canvas.manual_hazard_map)
            obstacle_map = np.maximum(self.hazard_results["obstacle_map"], self.canvas.manual_hazard_map)

            path = planner_algo(
                risk_map, obstacle_map, 
                start=self.start_pos, goal=self.goal_pos, 
                risk_weight=RISK_WEIGHT
            )

            if path is None:
                QMessageBox.information(self, "No Path", "No valid path found between Start and Goal.")
                self.statusBar.showMessage("Path planning failed: No valid path.")
                return

            path_smooth = smooth_path(
                path, obstacle_map,
                smoothing=SMOOTHING_FACTOR, oversample=SMOOTH_OVERSAMPLE
            )

            self.canvas.display_elements(self.start_pos, self.goal_pos, path, path_smooth)

            energy_data = compute_energy(path, self.image_norm, self.hazard_results["slope_map"], self.hazard_results["roughness_map"])
            self.update_stats(energy_data, len(path))

            self.statusBar.showMessage(f"Path found! Nodes: {len(path)}. Strategy: {self.combo_strategy.currentText()}")
            self.btn_clear.setEnabled(True)
            self.stats_group.setEnabled(True)

        except ValueError as e:
            QMessageBox.critical(self, "Planning Error", str(e))
            self.statusBar.showMessage("Path planning error.")

    def update_stats(self, energy_data, path_len):
        self.lbl_stat_energy.setText(f"Energy Consumed: {energy_data['total']:.2f}")
        self.lbl_stat_dist.setText(f"Path Distance: {path_len}m (equiv)")
        self.lbl_stat_slope.setText(f"Uphill: {energy_data['n_uphill']} / Downhill: {energy_data['n_downhill']}")

    def on_strategy_changed(self, index):
        if index == 0: 
            self.slider_slope.setValue(40)
            self.slider_crater.setValue(20)
        elif index == 1: 
            self.slider_slope.setValue(70)
            self.slider_crater.setValue(80)
        else: 
            self.slider_slope.setValue(15)
            self.slider_crater.setValue(5)
        self.update_risk_map()

    def clear_mission(self):
        self.start_pos = None
        self.goal_pos = None
        self.lbl_start.setText("Start: None")
        self.lbl_goal.setText("Goal: None")
        self.canvas.display_elements(None, None, None, None)
        self.stats_group.setEnabled(False)
        self.btn_plan.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.statusBar.showMessage("Mission cleared. Please reset Start and Goal.")

    def on_param_changed(self, value):

        self.lbl_slope.setText(f"Eğimden Kaçınma (Slope): {self.slider_slope.value()}%")
        self.lbl_crater.setText(f"Kraterden Kaçınma (Crater): {self.slider_crater.value()}%")
        self.update_risk_map()

    def update_risk_map(self):
        if self.hazard_results is None:
            return

        w_slope = self.slider_slope.value() / 100.0
        w_crat = self.slider_crater.value() / 100.0

        risk_map = build_risk_map(
            self.hazard_results["slope_map"],
            self.hazard_results["obstacle_map"],
            self.hazard_results["crater_map"],
            self.hazard_results["roughness_map"],
            w_slope=w_slope, w_obs=W_OBSTACLE, w_crat=w_crat, w_rough=W_ROUGH
        )

        risk_map = np.maximum(risk_map, self.canvas.manual_hazard_map)

        self.canvas.set_overlays(
            self.hazard_results["crater_map"],
            self.hazard_results["obstacle_map"],
            risk_map,
            self.hazard_results["slope_map"]
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LunarApp()
    window.show()
    sys.exit(app.exec_())
