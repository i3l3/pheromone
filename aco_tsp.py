"""Qt realtime Ant Colony Optimization visualization for TSP.

Run:
    python .\aco_test\aco_tsp_realtime.py
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from typing import Sequence
import qdarktheme

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


@dataclass
class City:
    x: float
    y: float


@dataclass(frozen=True)
class AcoConfig:
    cities: int = 30
    ants: int = 45
    iterations: int = 200
    alpha: float = 1.0
    beta: float = 5.0
    evaporation: float = 0.45
    q: float = 100.0
    seed: int = 7
    delay_ms: int = 60


class AcoSolver:
    def __init__(self, config: AcoConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.cities = [
            City(self.rng.uniform(0, 100), self.rng.uniform(0, 100))
            for _ in range(config.cities)
        ]
        self.distances = self._build_distance_matrix()
        self.pheromones = [
            [1.0 for _ in range(config.cities)] for _ in range(config.cities)
        ]
        self.best_tour: list[int] = []
        self.best_length = math.inf
        self.iteration = 0

    def reset_search(self) -> None:
        city_count = len(self.cities)
        self.distances = self._build_distance_matrix()
        self.pheromones = [
            [1.0 for _ in range(city_count)] for _ in range(city_count)
        ]
        self.best_tour = []
        self.best_length = math.inf
        self.iteration = 0

    def randomize_cities(self) -> None:
        self.cities = [
            City(self.rng.uniform(0, 100), self.rng.uniform(0, 100))
            for _ in range(self.config.cities)
        ]
        self.reset_search()

    def move_city(self, index: int, x: float, y: float) -> None:
        self.cities[index].x = min(max(x, 0.0), 100.0)
        self.cities[index].y = min(max(y, 0.0), 100.0)
        self.reset_search()

    def step(self) -> None:
        tours = [self._build_ant_tour() for _ in range(self.config.ants)]

        for tour in tours:
            length = self._tour_length(tour)
            if length < self.best_length:
                self.best_tour = list(tour)
                self.best_length = length

        self._update_pheromones(tours)
        self.iteration += 1

    def _build_distance_matrix(self) -> list[list[float]]:
        matrix: list[list[float]] = []
        for city_a in self.cities:
            row = []
            for city_b in self.cities:
                row.append(math.hypot(city_a.x - city_b.x, city_a.y - city_b.y))
            matrix.append(row)
        return matrix

    def _tour_length(self, tour: Sequence[int]) -> float:
        return sum(
            self.distances[tour[index]][tour[(index + 1) % len(tour)]]
            for index in range(len(tour))
        )

    def _choose_next_city(self, current: int, unvisited: set[int]) -> int:
        weights = []
        total = 0.0

        for city in unvisited:
            pheromone = self.pheromones[current][city] ** self.config.alpha
            visibility = (
                1.0 / max(self.distances[current][city], 1e-12)
            ) ** self.config.beta
            weight = pheromone * visibility
            weights.append((city, weight))
            total += weight

        if total <= 0.0:
            return self.rng.choice(tuple(unvisited))

        pick = self.rng.random() * total
        cumulative = 0.0
        for city, weight in weights:
            cumulative += weight
            if cumulative >= pick:
                return city

        return weights[-1][0]

    def _build_ant_tour(self) -> list[int]:
        city_count = len(self.cities)
        start = self.rng.randrange(city_count)
        tour = [start]
        unvisited = set(range(city_count))
        unvisited.remove(start)

        while unvisited:
            next_city = self._choose_next_city(tour[-1], unvisited)
            tour.append(next_city)
            unvisited.remove(next_city)

        return tour

    def _update_pheromones(self, tours: Sequence[Sequence[int]]) -> None:
        for row in self.pheromones:
            for index, value in enumerate(row):
                row[index] = max(value * (1.0 - self.config.evaporation), 1e-12)

        for tour in tours:
            deposit = self.config.q / self._tour_length(tour)
            for index, city_a in enumerate(tour):
                city_b = tour[(index + 1) % len(tour)]
                self.pheromones[city_a][city_b] += deposit
                self.pheromones[city_b][city_a] += deposit

    def remove_city(self, index: int) -> None:
        self.cities.pop(index)
        self.refresh_config()
        self.reset_search()

    def refresh_config(self) -> None:
        self.config = AcoConfig(
            cities=len(self.cities),
            ants=self.config.ants,
            iterations=self.config.iterations,
            alpha=self.config.alpha,
            beta=self.config.beta,
            evaporation=self.config.evaporation,
            q=self.config.q,
            seed=self.config.seed,
            delay_ms=self.config.delay_ms,
        )


class TspCanvas(QWidget):
    def __init__(self, solver: AcoSolver, status_changed) -> None:
        super().__init__()
        self.solver = solver
        self.status_changed = status_changed
        self.padding = 56
        self.dragging_city: int | None = None
        self.setMinimumSize(640, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        plot_rect = self._plot_rect()
        painter.setPen(QPen(QColor("#d8cfc0"), 1))
        painter.drawRect(plot_rect)

        if self.solver.best_tour:
            painter.setPen(QPen(QColor("#2457a6"), 3))
            points = [
                self._screen_point(self.solver.cities[index])
                for index in self.solver.best_tour + [self.solver.best_tour[0]]
            ]
            for start, end in zip(points, points[1:]):
                painter.drawLine(start, end)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        for index, city in enumerate(self.solver.cities): # 노드 그리기
            point = self._screen_point(city)
            fill = QColor("#f2b134") if index == self.dragging_city else QColor("#d1495b")
            painter.setBrush(fill)
            painter.setPen(QPen(QColor("#7c1f2d"), 1))
            painter.drawEllipse(point, 6, 6)
            painter.setPen(QPen(QColor("#1a88ef"), 1))
            painter.drawText(
                QRectF(point.x() - 16, point.y() - 28, 32, 14),
                Qt.AlignmentFlag.AlignCenter,
                str(index),
            )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging_city = self._nearest_city(event.position())
            if self.dragging_city is not None:
                self.status_changed()
                self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self.dragging_city = self._nearest_city(event.position())
            if self.dragging_city is not None and len(self.solver.cities) > 2:
                self.solver.remove_city(self.dragging_city)
                self.dragging_city = None
                self.status_changed()
                self.update()
        else:
            return

    def mouseMoveEvent(self, event) -> None:
        if self.dragging_city is None:
            return

        city_x, city_y = self._city_point(event.position())
        self.solver.move_city(self.dragging_city, city_x, city_y)
        self.status_changed()
        self.update()

    def mouseReleaseEvent(self, _event) -> None:
        self.dragging_city = None
        self.status_changed()
        self.update()

    def _plot_rect(self) -> QRectF:
        return QRectF(
            self.padding,
            self.padding,
            max(self.width() - self.padding * 2, 1),
            max(self.height() - self.padding * 2, 1),
        )

    def _screen_point(self, city: City) -> QPointF:
        plot_rect = self._plot_rect()
        return QPointF(
            plot_rect.left() + (city.x / 100.0) * plot_rect.width(),
            plot_rect.top() + (city.y / 100.0) * plot_rect.height(),
        )

    def _city_point(self, point: QPointF) -> tuple[float, float]:
        plot_rect = self._plot_rect()
        x = ((point.x() - plot_rect.left()) / plot_rect.width()) * 100.0
        y = ((point.y() - plot_rect.top()) / plot_rect.height()) * 100.0
        return min(max(x, 0.0), 100.0), min(max(y, 0.0), 100.0)

    def _nearest_city(self, point: QPointF) -> int | None:
        nearest_index = None
        nearest_distance = math.inf

        for index, city in enumerate(self.solver.cities):
            city_point = self._screen_point(city)
            distance = math.hypot(city_point.x() - point.x(), city_point.y() - point.y())
            if distance < nearest_distance:
                nearest_index = index
                nearest_distance = distance

        return nearest_index if nearest_distance <= 16 else None


class MainWindow(QMainWindow):
    def __init__(self, solver: AcoSolver) -> None:
        super().__init__()
        self.solver = solver
        self.running = False
        self.setWindowTitle("ACO TSP Qt Visualization")
        self.resize(900, 680)
        self.setMinimumSize(700, 520)

        self.timer = QTimer(self)
        self.timer.setInterval(self.solver.config.delay_ms)
        self.timer.timeout.connect(self._tick)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QWidget()
        toolbar.setObjectName("toolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 8, 10, 8)
        toolbar_layout.setSpacing(6)

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.step_button = QPushButton("Step")
        self.reset_button = QPushButton("Reset")
        self.random_button = QPushButton("Randomize")
        self.add_button = QPushButton("Add city")

        for button in (
            self.start_button,
            self.stop_button,
            self.step_button,
            self.reset_button,
            self.random_button,
            self.add_button
        ):
            button.setFixedWidth(96)
            toolbar_layout.addWidget(button)

        toolbar_layout.addStretch(1)
        layout.addWidget(toolbar)

        self.canvas = TspCanvas(self.solver, self._city_moved)
        layout.addWidget(self.canvas, stretch=1)

        self.status_label = QLabel()
        self.status_label.setObjectName("statusBar")
        layout.addWidget(self.status_label)
        self.setCentralWidget(root)

        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.step_button.clicked.connect(self.step_once)
        self.reset_button.clicked.connect(self.reset_search)
        self.random_button.clicked.connect(self.randomize)
        self.add_button.clicked.connect(self.add_city)
        self._update_status()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            self.stop() if self.running else self.start()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def start(self) -> None:
        self.running = True
        self.timer.start()
        self._update_status()

    def stop(self) -> None:
        self.running = False
        self.timer.stop()
        self._update_status()

    def step_once(self) -> None:
        self.stop()
        if self.solver.iteration < self.solver.config.iterations:
            self.solver.step()
        self.canvas.update()
        self._update_status()

    def reset_search(self) -> None:
        self.stop()
        self.solver.reset_search()
        self.canvas.update()
        self._update_status()

    def randomize(self) -> None:
        self.stop()
        self.solver.randomize_cities()
        self.canvas.update()
        self._update_status()

    def add_city(self) -> None:
        self.stop()
        self.solver.cities.append(City(self.solver.rng.uniform(0, 100), self.solver.rng.uniform(0, 100)))
        self.solver.refresh_config()
        self.reset_search()
        self._update_status()

    def _city_moved(self) -> None:
        self.stop()
        self._update_status()

    def _tick(self) -> None:
        if self.solver.iteration >= self.solver.config.iterations:
            self.stop()
            return

        self.solver.step()
        self.canvas.update()
        self._update_status()

    def _update_status(self) -> None:
        state = "running" if self.running else "paused"
        length = (
            f"{self.solver.best_length:.2f}"
            if math.isfinite(self.solver.best_length)
            else "-"
        )
        self.status_label.setText(
            f"{state} | iteration {self.solver.iteration}/"
            f"{self.solver.config.iterations} | number of cities {len(self.solver.cities)} | best distance {length} | "
            "drag cities to move | Space: start/stop | Esc: close"
        )


def parse_args() -> AcoConfig:
    parser = argparse.ArgumentParser(description="Qt realtime ACO visualization for TSP.")
    parser.add_argument("--cities", type=int, default=AcoConfig.cities)
    parser.add_argument("--ants", type=int, default=AcoConfig.ants)
    parser.add_argument("--iterations", type=int, default=AcoConfig.iterations)
    parser.add_argument("--alpha", type=float, default=AcoConfig.alpha)
    parser.add_argument("--beta", type=float, default=AcoConfig.beta)
    parser.add_argument("--evaporation", type=float, default=AcoConfig.evaporation)
    parser.add_argument("--q", type=float, default=AcoConfig.q)
    parser.add_argument("--seed", type=int, default=AcoConfig.seed)
    parser.add_argument("--delay-ms", type=int, default=AcoConfig.delay_ms)
    args = parser.parse_args()

    if args.cities < 3:
        parser.error("--cities must be at least 3")
    if args.ants < 1:
        parser.error("--ants must be at least 1")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if not 0.0 <= args.evaporation < 1.0:
        parser.error("--evaporation must be in the range [0.0, 1.0)")
    if args.delay_ms < 1:
        parser.error("--delay-ms must be at least 1")

    return AcoConfig(
        cities=args.cities,
        ants=args.ants,
        iterations=args.iterations,
        alpha=args.alpha,
        beta=args.beta,
        evaporation=args.evaporation,
        q=args.q,
        seed=args.seed,
        delay_ms=args.delay_ms,
    )


def main() -> None:
    config = parse_args()
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("auto")
    window = MainWindow(AcoSolver(config))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
