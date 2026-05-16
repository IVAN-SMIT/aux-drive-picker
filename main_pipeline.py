#!/usr/bin/env python3
"""
main_pipeline.py

СКВОЗНОЙ ПАЙПЛАЙН: изображение → геометрия → кинематика → биомеханика → привод.

ПОЛНЫЙ ФУНКЦИОНАЛ (ЭТАПЫ 1–6):
    Этап 1: Компьютерное зрение + геометрия (HandTracker + GeometryCalculator)
    Этап 2: Кинематика (KinematicsCalculator — диапазоны, прямая задача)
    Этап 3: Биомеханика (BiomechanicsCalculator — M(θ), Mmax)
    Этап 4: Подбор привода (ActuatorDatabase — оптимизация по массе/цене)
    Этап 5: Интеграция (этот файл — сквозной прогон)
    Этап 6: Валидация + визуализация (MuJoCo + графики + CSV)

АРХИТЕКТУРА:
    Видео/Камера
    → HandTracker (3D-координаты)
    → GeometryCalculator (длины, углы)
    → KinematicsCalculator (валидация, координаты)
    → BiomechanicsCalculator (моменты, Mmax)
    → ActuatorDatabase (подбор привода+редуктор)
    → MuJoCoArmSim (визуализация, опционально)
    → CSV + Графики

ЗАПУСК:
    python main_pipeline.py                                    # видео по умолчанию
    python main_pipeline.py --video other.mp4                  # другое видео
    python main_pipeline.py --camera 0                         # веб-камера
    python main_pipeline.py --load 5.0 --side right            # груз 5 кг, правая рука
    python main_pipeline.py --no-mujoco --calc-only            # только расчёты
    python main_pipeline.py --output report                    # папка для результатов

    Авторы: Иван Смирнов, Леонид Нагорный

    Тг: https://t.me/ivansmittt
    Cайт: auxexo.ru
"""

import sys
import os
import time
import csv
import json
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.parameters import (
    DEFAULT_SHOULDER_WIDTH,
    AVG_UPPER_ARM,
    AVG_FOREARM
)
from vision.hand_tracker import HandTracker
from vision.geometry_calculator import GeometryCalculator
from kinematic.kinematics import KinematicsCalculator
from biomechanic.biomechanics import BiomechanicsCalculator
from actuators.actuator_database import ActuatorDatabase

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

if HAS_MUJOCO:
    from biomechanic.mujoco_sim import MuJoCoArmSim

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ============================================================================
# КОНСТАНТЫ
# ============================================================================

DEFAULT_VIDEO_PATH = "vision/video.mp4"
LOAD_SCENARIOS = {
    "no_load": 0.0,
    "light": 2.0,
    "heavy": 5.0,
}


# ============================================================================
# СТРУКТУРА РЕЗУЛЬТАТА
# ============================================================================

@dataclass
class PipelineResult:
    """Полный результат обработки одного кадра"""
    frame: int
    time_sec: float
    
    # Геометрия
    upper_arm_m: float
    forearm_m: float
    shoulder_angle: float
    elbow_angle: float
    brachial_index: float
    
    # Кинематика
    wrist_x: float
    wrist_y: float
    movement_range_pct: float
    kinematics_valid: bool
    
    # Биомеханика
    shoulder_torque: float
    elbow_torque: float
    
    # MuJoCo (опционально)
    mujoco_shoulder_torque: Optional[float] = None
    mujoco_elbow_torque: Optional[float] = None
    
    # Привод (заполняется в конце)
    selected_actuator_elbow: Optional[str] = None
    selected_actuator_shoulder: Optional[str] = None


# ============================================================================
# ПАЙПЛАЙН
# ============================================================================

class ArmPipeline:
    """
    Сквозной пайплайн: изображение → привод.
    """
    
    def __init__(
        self,
        side: str = "left",
        camera_id: int = 0,
        video_path: str = None,
        use_camera: bool = False,
        external_load_kg: float = 0.0,
        use_mujoco: bool = True,
        output_dir: str = "outputs",
        show_plots: bool = True,
        save_plots: bool = True
    ):
        self.side = side
        self.camera_id = camera_id
        self.use_camera = use_camera
        self.video_path = video_path or DEFAULT_VIDEO_PATH
        self.external_load_kg = external_load_kg
        self.use_mujoco = use_mujoco and HAS_MUJOCO
        self.output_dir = Path(output_dir)
        self.show_plots = show_plots and HAS_PLT
        self.save_plots = save_plots and HAS_PLT
        
        # Компоненты
        self.tracker: HandTracker = None
        self.geometry_calc: GeometryCalculator = None
        self.kinematics: KinematicsCalculator = None
        self.biomechanics: BiomechanicsCalculator = None
        self.actuator_db: ActuatorDatabase = None
        self.mujoco_sim: MuJoCoArmSim = None
        
        self.cap = None
        self.frame_count = 0
        self.start_time = None
        
        # Результаты
        self.results: List[PipelineResult] = []
        self.actuator_selections: Dict = {}
        
        # Статистика
        self.kinematics_violations = 0
        self.total_frames = 0
    
    # ==================================================================
    # ИНИЦИАЛИЗАЦИЯ
    # ==================================================================
    
    def initialize(self) -> bool:
        """Инициализация всех компонентов"""
        print(f"\n{'='*60}")
        print(f"  🦾 СКВОЗНОЙ ПАЙПЛАЙН: ИЗОБРАЖЕНИЕ → ПРИВОД")
        print(f"{'='*60}")
        
        # Этап 1: Компьютерное зрение
        print("\n[Этап 1] Инициализация трекера и геометрии...")
        self.tracker = HandTracker(side=self.side)
        self.geometry_calc = GeometryCalculator()
        print("  ✅ HandTracker + GeometryCalculator")
        
        # Этап 2: Кинематика
        print("\n[Этап 2] Инициализация кинематики...")
        self.kinematics = KinematicsCalculator()
        ws = self.kinematics.get_workspace_info()
        print(f"  ✅ Рабочая зона: {ws['max_reach_m']} м")
        
        # Этап 3: Биомеханика
        print("\n[Этап 3] Инициализация биомеханики...")
        self.biomechanics = BiomechanicsCalculator()
        req = self.biomechanics.get_actuator_requirements(self.external_load_kg)
        print(f"  ✅ Mmax плечо={req['max_shoulder_raw']:.3f} Н·м, "
              f"локоть={req['max_elbow_raw']:.3f} Н·м")
        
        # Этап 4: База приводов
        print("\n[Этап 4] Загрузка базы приводов...")
        self.actuator_db = ActuatorDatabase()
        stats = self.actuator_db.get_statistics()
        print(f"  ✅ Загружено {stats['total_models']} приводов "
              f"({stats['russian_models']} РФ, {stats['chinese_models']} КНР)")
        
        # MuJoCo (опционально)
        if self.use_mujoco:
            print("\n[MuJoCo] Запуск симуляции...")
            try:
                self.mujoco_sim = MuJoCoArmSim(external_load_kg=self.external_load_kg)
                self.mujoco_sim.stabilize(50)
                print("  ✅ MuJoCo запущен")
            except Exception as e:
                print(f"  ⚠️ MuJoCo не запущен: {e}")
                self.use_mujoco = False
        
        # Источник видео
        if self.use_camera:
            self.cap = cv2.VideoCapture(self.camera_id)
            source = f"камера {self.camera_id}"
            self._need_flip = True
        else:
            if not os.path.exists(self.video_path):
                print(f"\n❌ Видео не найдено: {self.video_path}")
                return False
            self.cap = cv2.VideoCapture(self.video_path)
            source = f"видео: {self.video_path}"
            self._need_flip = False
        
        if not self.cap.isOpened():
            print(f"❌ Не удалось открыть {source}")
            return False
        
        cv2.namedWindow("Arm Pipeline", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Arm Pipeline", 960, 720)
        
        self.start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"  ЗАПУСК ПАЙПЛАЙНА")
        print(f"{'='*60}")
        print(f"  Источник: {source}")
        print(f"  Рука: {self.side}")
        print(f"  Нагрузка: {self.external_load_kg} кг")
        print(f"  MuJoCo: {'да' if self.use_mujoco else 'нет'}")
        print(f"  Графики: {'да' if self.show_plots else 'нет'}")
        print(f"\n  q — выход, p — пауза")
        print(f"{'='*60}\n")
        
        return True
    
    # ==================================================================
    # ОСНОВНОЙ ЦИКЛ
    # ==================================================================
    
    def run(self):
        """Основной цикл обработки"""
        if not self.initialize():
            return
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("📭 Конец видео")
                    break
                
                if self._need_flip:
                    frame = cv2.flip(frame, 1)
                self.frame_count += 1
                self.total_frames += 1
                
                # === ЭТАП 1: Компьютерное зрение + геометрия ===
                raw_points = self.tracker.process_frame(frame)
                
                if raw_points is not None and self.tracker.are_points_visible(raw_points):
                    geometry = self.geometry_calc.process(raw_points)
                    
                    if geometry is not None:
                        sh_angle = geometry.shoulder_angle
                        el_angle = geometry.elbow_angle
                        
                        # === ЭТАП 2: Кинематика ===
                        kin = self.kinematics.forward_kinematics(sh_angle, el_angle)
                        range_pct = self.kinematics.movement_range_percent(el_angle)
                        
                        if not kin.is_valid:
                            self.kinematics_violations += 1
                        
                        # === ЭТАП 3: Биомеханика ===
                        moments = self.biomechanics.compute_all(
                            sh_angle, el_angle, self.external_load_kg
                        )
                        
                        # === ЭТАП 5: MuJoCo ===
                        mujoco_sh_torque = None
                        mujoco_el_torque = None
                        
                        if self.use_mujoco and self.mujoco_sim:
                            self.mujoco_sim.set_angles(sh_angle, el_angle)
                            for _ in range(2):
                                joint_state = self.mujoco_sim.step()
                            mujoco_sh_torque = joint_state.shoulder_torque
                            mujoco_el_torque = joint_state.elbow_torque
                        
                        # Сохраняем результат
                        result = PipelineResult(
                            frame=self.frame_count,
                            time_sec=round(time.time() - self.start_time, 1),
                            upper_arm_m=geometry.upper_arm_m,
                            forearm_m=geometry.forearm_m,
                            shoulder_angle=sh_angle,
                            elbow_angle=el_angle,
                            brachial_index=geometry.brachial_index,
                            wrist_x=kin.wrist_pos.x,
                            wrist_y=kin.wrist_pos.y,
                            movement_range_pct=round(range_pct, 1),
                            kinematics_valid=kin.is_valid,
                            shoulder_torque=moments.shoulder_torque,
                            elbow_torque=moments.elbow_torque,
                            mujoco_shoulder_torque=mujoco_sh_torque,
                            mujoco_elbow_torque=mujoco_el_torque
                        )
                        self.results.append(result)
                
                # === ВИЗУАЛИЗАЦИЯ ===
                frame = self.tracker.draw_debug(frame)
                frame = self._draw_info(frame)
                
                cv2.imshow("Arm Pipeline", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("⏸️  ПАУЗА" if paused else "▶️  ПРОДОЛЖЕНИЕ")
        
        self._finalize()
    
    # ==================================================================
    # ОТРИСОВКА
    # ==================================================================
    
    def _draw_info(self, frame: np.ndarray) -> np.ndarray:
        """Отображение информации на кадре"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Полупрозрачный фон
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), (30, 30, 30), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        y = 22
        
        if self.results:
            r = self.results[-1]
            
            cv2.putText(frame, f"Frame: {r.frame}", (10, y), font, 0.6, (255, 255, 255), 2)
            y += 28
            
            cv2.putText(frame, f"Angles: sh={r.shoulder_angle:+.1f}, el={r.elbow_angle:.1f} deg", 
                       (10, y), font, 0.5, (0, 255, 255), 1)
            y += 22
            
            cv2.putText(frame, f"Torque: sh={r.shoulder_torque:.3f}, el={r.elbow_torque:.3f} Nm", 
                       (10, y), font, 0.5, (0, 255, 255), 1)
            y += 22
            
            if r.mujoco_shoulder_torque is not None:
                cv2.putText(frame, f"MuJoCo: sh={r.mujoco_shoulder_torque:.3f}, "
                           f"el={r.mujoco_elbow_torque:.3f} Nm",
                           (10, y), font, 0.5, (0, 200, 255), 1)
                y += 22
            
            if r.selected_actuator_elbow:
                cv2.putText(frame, f"Actuator: {r.selected_actuator_elbow[:30]}", 
                           (10, y), font, 0.45, (0, 255, 0), 1)
                y += 22
            
            cv2.putText(frame, f"Valid: {'OK' if r.kinematics_valid else 'VIOLATION'}", 
                       (10, y), font, 0.5, 
                       (0, 255, 0) if r.kinematics_valid else (0, 0, 255), 1)
        
        # Статус
        status = "TRACKING" if self.tracker.is_tracking() else "LOST"
        color = (0, 255, 0) if self.tracker.is_tracking() else (0, 0, 255)
        cv2.putText(frame, status, (w - 150, 25), font, 0.6, color, 2)
        
        cv2.putText(frame, f"Samples: {len(self.results)}", (w - 200, h - 15), 
                   font, 0.5, (150, 150, 150), 1)
        
        return frame
    
    # ==================================================================
    # ЗАВЕРШЕНИЕ
    # ==================================================================
    
    def _finalize(self):
        """Подбор привода, графики, отчёт"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n{'='*60}")
        print(f"  ОБРАБОТКА ЗАВЕРШЕНА")
        print(f"{'='*60}")
        print(f"  Кадров: {self.frame_count}")
        print(f"  Измерений: {len(self.results)}")
        print(f"  Нарушений кинематики: {self.kinematics_violations}")
        
        if not self.results:
            self.shutdown()
            return
        
        # === ЭТАП 4: Подбор привода ===
        self._select_actuators()
        
        # === Графики ===
        if self.show_plots or self.save_plots:
            self._generate_plots()
        
        # === Сохранение ===
        self._save_results()
        self._save_report()
        
        # Вывод итогов
        self._print_summary()
        
        self.shutdown()
    
    def _select_actuators(self):
        """Подбор оптимальных приводов для всех сценариев"""
        print(f"\n{'='*60}")
        print(f"  ЭТАП 4: ПОДБОР ПРИВОДОВ")
        print(f"{'='*60}")
        
        self.actuator_selections = {}
        
        for scenario_name, load in LOAD_SCENARIOS.items():
            req = self.biomechanics.get_actuator_requirements(load)
            
            print(f"\n  Сценарий: {scenario_name} ({load} кг)")
            print(f"  Требуемый момент плеча: {req['M_shoulder_required']:.3f} Н·м")
            print(f"  Требуемый момент локтя:  {req['M_elbow_required']:.3f} Н·м")
            
            # Подбор для локтя
            elbow_actuator = self.actuator_db.find_optimal(
                required_torque=req['M_elbow_required'],
                joint="elbow",
                max_weight=0.5,
                prefer_russian=True
            )
            
            # Подбор для плеча
            shoulder_actuator = self.actuator_db.find_optimal(
                required_torque=req['M_shoulder_required'],
                joint="shoulder",
                max_weight=1.0,
                prefer_russian=True
            )
            
            if elbow_actuator:
                print(f"  ✅ Локоть:  {elbow_actuator.name} "
                      f"({elbow_actuator.nominal_torque} Н·м, {elbow_actuator.weight} кг)")
            else:
                print(f"  ❌ Локоть:  подходящий привод не найден")
            
            if shoulder_actuator:
                print(f"  ✅ Плечо:   {shoulder_actuator.name} "
                      f"({shoulder_actuator.nominal_torque} Н·м, {shoulder_actuator.weight} кг)")
            else:
                print(f"  ❌ Плечо:   подходящий привод не найден")
            
            self.actuator_selections[scenario_name] = {
                'load_kg': load,
                'required_shoulder': req['M_shoulder_required'],
                'required_elbow': req['M_elbow_required'],
                'elbow_actuator': elbow_actuator.name if elbow_actuator else "—",
                'shoulder_actuator': shoulder_actuator.name if shoulder_actuator else "—",
                'elbow_actuator_obj': elbow_actuator,
                'shoulder_actuator_obj': shoulder_actuator,
            }
        
        # Присваиваем выбранный привод результатам
        if self.actuator_selections:
            default_elbow = self.actuator_selections.get(
                'no_load', next(iter(self.actuator_selections.values()))
            ).get('elbow_actuator', "—")
            
            for r in self.results:
                r.selected_actuator_elbow = default_elbow
    
    def _generate_plots(self):
        """Генерация графиков"""
        if not HAS_PLT:
            return
        
        print(f"\n{'='*60}")
        print(f"  ГЕНЕРАЦИЯ ГРАФИКОВ")
        print(f"{'='*60}")
        
        frames = [r.frame for r in self.results]
        times = [r.time_sec for r in self.results]
        sh_angles = [r.shoulder_angle for r in self.results]
        el_angles = [r.elbow_angle for r in self.results]
        sh_torques = [r.shoulder_torque for r in self.results]
        el_torques = [r.elbow_torque for r in self.results]
        upper_lengths = [r.upper_arm_m for r in self.results]
        forearm_lengths = [r.forearm_m for r in self.results]
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle('Arm Pipeline Results', fontsize=14)
        
        # Углы
        axes[0, 0].plot(times, sh_angles, 'r-', label='Shoulder', alpha=0.7)
        axes[0, 0].plot(times, el_angles, 'b-', label='Elbow', alpha=0.7)
        axes[0, 0].set_ylabel('Angle (deg)')
        axes[0, 0].set_title('Joint Angles')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Моменты
        axes[0, 1].plot(times, sh_torques, 'r-', label='Shoulder', alpha=0.7)
        axes[0, 1].plot(times, el_torques, 'b-', label='Elbow', alpha=0.7)
        axes[0, 1].set_ylabel('Torque (N·m)')
        axes[0, 1].set_title('Joint Torques (Analytical)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Длины звеньев
        axes[1, 0].plot(times, upper_lengths, 'r-', label='Upper arm', alpha=0.7)
        axes[1, 0].plot(times, forearm_lengths, 'b-', label='Forearm', alpha=0.7)
        axes[1, 0].set_ylabel('Length (m)')
        axes[1, 0].set_title('Segment Lengths')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Траектория запястья
        wrist_x = [r.wrist_x for r in self.results]
        wrist_y = [r.wrist_y for r in self.results]
        axes[1, 1].plot(wrist_x, wrist_y, 'g-', alpha=0.5)
        axes[1, 1].scatter(wrist_x[0], wrist_y[0], c='green', s=100, label='Start')
        axes[1, 1].scatter(wrist_x[-1], wrist_y[-1], c='red', s=100, label='End')
        axes[1, 1].set_xlabel('X (m)')
        axes[1, 1].set_ylabel('Y (m)')
        axes[1, 1].set_title('Wrist Trajectory')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axis('equal')
        
        # M(θ) — момент от угла локтя
        axes[2, 0].scatter(el_angles, el_torques, c='blue', alpha=0.3, s=10)
        axes[2, 0].set_xlabel('Elbow Angle (deg)')
        axes[2, 0].set_ylabel('Elbow Torque (N·m)')
        axes[2, 0].set_title('M(θ) — Elbow')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Гистограмма моментов
        axes[2, 1].hist(el_torques, bins=20, alpha=0.5, color='blue', label='Elbow')
        axes[2, 1].hist(sh_torques, bins=20, alpha=0.5, color='red', label='Shoulder')
        axes[2, 1].set_xlabel('Torque (N·m)')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].set_title('Torque Distribution')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            self.output_dir.mkdir(exist_ok=True)
            plot_path = self.output_dir / f"pipeline_plots_{time.strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  📊 Графики сохранены: {plot_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def _save_results(self):
        """Сохранение CSV с результатами"""
        self.output_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"pipeline_results_{timestamp}.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'frame', 'time_sec', 'upper_arm_m', 'forearm_m',
                'shoulder_angle', 'elbow_angle', 'brachial_index',
                'wrist_x', 'wrist_y', 'movement_range_pct', 'kinematics_valid',
                'shoulder_torque', 'elbow_torque',
                'mujoco_shoulder_torque', 'mujoco_elbow_torque',
                'selected_actuator_elbow'
            ])
            writer.writeheader()
            for r in self.results:
                writer.writerow({
                    'frame': r.frame,
                    'time_sec': r.time_sec,
                    'upper_arm_m': r.upper_arm_m,
                    'forearm_m': r.forearm_m,
                    'shoulder_angle': r.shoulder_angle,
                    'elbow_angle': r.elbow_angle,
                    'brachial_index': r.brachial_index,
                    'wrist_x': r.wrist_x,
                    'wrist_y': r.wrist_y,
                    'movement_range_pct': r.movement_range_pct,
                    'kinematics_valid': r.kinematics_valid,
                    'shoulder_torque': r.shoulder_torque,
                    'elbow_torque': r.elbow_torque,
                    'mujoco_shoulder_torque': r.mujoco_shoulder_torque,
                    'mujoco_elbow_torque': r.mujoco_elbow_torque,
                    'selected_actuator_elbow': r.selected_actuator_elbow
                })
        
        print(f"  📁 Результаты: {csv_path}")
    
    def _save_report(self):
        """Сохранение JSON-отчёта"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"pipeline_report_{timestamp}.json"
        
        # Собираем статистику
        if self.results:
            sh_torques = [r.shoulder_torque for r in self.results]
            el_torques = [r.elbow_torque for r in self.results]
            sh_angles = [r.shoulder_angle for r in self.results]
            el_angles = [r.elbow_angle for r in self.results]
            
            report = {
                'timestamp': timestamp,
                'source': 'camera' if self.use_camera else self.video_path,
                'side': self.side,
                'external_load_kg': self.external_load_kg,
                'total_frames': self.total_frames,
                'measurements': len(self.results),
                'kinematics_violations': self.kinematics_violations,
                'statistics': {
                    'shoulder_angle': {
                        'min': min(sh_angles), 'max': max(sh_angles),
                        'mean': round(np.mean(sh_angles), 1)
                    },
                    'elbow_angle': {
                        'min': min(el_angles), 'max': max(el_angles),
                        'mean': round(np.mean(el_angles), 1)
                    },
                    'shoulder_torque': {
                        'min': round(min(sh_torques), 3),
                        'max': round(max(sh_torques), 3),
                        'mean': round(np.mean(sh_torques), 3)
                    },
                    'elbow_torque': {
                        'min': round(min(el_torques), 3),
                        'max': round(max(el_torques), 3),
                        'mean': round(np.mean(el_torques), 3)
                    }
                },
                'actuator_selections': {}
            }
            
            for name, sel in self.actuator_selections.items():
                report['actuator_selections'][name] = {
                    'load_kg': sel['load_kg'],
                    'required_shoulder_torque': round(sel['required_shoulder'], 3),
                    'required_elbow_torque': round(sel['required_elbow'], 3),
                    'selected_shoulder': sel['shoulder_actuator'],
                    'selected_elbow': sel['elbow_actuator']
                }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"  📋 Отчёт: {report_path}")
    
    def _print_summary(self):
        """Итоговый вывод"""
        print(f"\n{'='*60}")
        print(f"  ИТОГОВЫЙ ОТЧЁТ")
        print(f"{'='*60}")
        
        if self.results:
            sh_torques = [r.shoulder_torque for r in self.results]
            el_torques = [r.elbow_torque for r in self.results]
            
            print(f"\n  📊 Статистика моментов:")
            print(f"  Плечо:  макс={max(sh_torques):.3f} Н·м, "
                  f"сред={np.mean(sh_torques):.3f} Н·м")
            print(f"  Локоть: макс={max(el_torques):.3f} Н·м, "
                  f"сред={np.mean(el_torques):.3f} Н·м")
        
        if self.actuator_selections:
            print(f"\n  🔧 РЕКОМЕНДОВАННЫЕ ПРИВОДЫ:")
            for name, sel in self.actuator_selections.items():
                print(f"\n  Сценарий: {name} ({sel['load_kg']} кг)")
                print(f"    Плечо:  {sel['shoulder_actuator']}")
                print(f"    Локоть: {sel['elbow_actuator']}")
        
        print(f"\n{'='*60}")
    
    def shutdown(self):
        """Корректное завершение"""
        if self.mujoco_sim:
            self.mujoco_sim.close()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        if self.tracker:
            self.tracker.close()
        print("👋 Пайплайн завершён")


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Сквозной пайплайн: изображение → привод",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python main_pipeline.py                                    # видео по умолчанию
  python main_pipeline.py --video other.mp4                  # другое видео
  python main_pipeline.py --camera 0                         # веб-камера
  python main_pipeline.py --load 5.0 --side right            # груз 5 кг
  python main_pipeline.py --no-mujoco                        # без MuJoCo
  python main_pipeline.py --no-plots                         # без графиков
  python main_pipeline.py --output my_results                # своя папка
        """
    )
    parser.add_argument("--side", default="left", choices=["left", "right"])
    parser.add_argument("--camera", type=int, default=-1)
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--load", type=float, default=0.0,
                       help="Внешняя нагрузка, кг")
    parser.add_argument("--no-mujoco", action="store_true",
                       help="Отключить MuJoCo")
    parser.add_argument("--no-plots", action="store_true",
                       help="Не показывать графики")
    parser.add_argument("--output", type=str, default="outputs",
                       help="Папка для результатов")
    
    args = parser.parse_args()
    
    use_camera = (args.camera >= 0)
    
    if not use_camera and not os.path.exists(args.video):
        print(f"\n❌ Видео не найдено: {args.video}")
        sys.exit(1)
    
    pipeline = ArmPipeline(
        side=args.side,
        camera_id=args.camera if use_camera else 0,
        video_path=args.video,
        use_camera=use_camera,
        external_load_kg=args.load,
        use_mujoco=not args.no_mujoco,
        output_dir=args.output,
        show_plots=not args.no_plots,
        save_plots=not args.no_plots
    )
    
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n⚠️ Прервано пользователем")
        pipeline._finalize()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        pipeline.shutdown()