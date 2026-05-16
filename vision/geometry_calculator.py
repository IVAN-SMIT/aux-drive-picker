"""
vision/geometry_calculator.py

Калькулятор геометрии руки на основе 3D World Landmarks.

ОТВЕТСТВЕННОСТЬ МОДУЛЯ:
- Принимает 3D-координаты из HandTracker (уже в метрах)
- Вычисляет длины звеньев как евклидово расстояние в 3D
- Вычисляет углы плеча и локтя
- Сглаживает углы и длины
- Вычисляет проценты от средних анатомических значений
- Вычисляет брахиальный индекс

КЛЮЧЕВОЕ ОТЛИЧИЕ ОТ ПРЕДЫДУЩЕЙ ВЕРСИИ:
Никакой нормализации глубины! MediaPipe уже даёт координаты в метрах.
Длины = sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)


Авторы: Иван Смирнов, Леонид Нагорный

Тг: https://t.me/ivansmittt
Cайт: auxexo.ru
"""

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass

from config.parameters import AVG_UPPER_ARM, AVG_FOREARM


@dataclass
class NormalizedGeometry:
    """Геометрические параметры руки (все в метрах и градусах)"""
    upper_arm_m: float
    forearm_m: float
    shoulder_angle: float
    elbow_angle: float
    scale_factor: float           # всегда 1.0 (для совместимости)
    shoulder_width: float         # реальная ширина плеч из 3D
    upper_arm_percent: float
    forearm_percent: float
    brachial_index: float
    tracking_confidence: float
    scale_frozen: bool = False


class GeometryCalculator:
    """
    Калькулятор геометрии на основе 3D World Landmarks.
    
    Фильтрация:
    - Сглаживание длин (скользящее среднее, окно 8 кадров)
    - Сглаживание углов (скользящее среднее, окно 8 кадров)
    - Анатомические ограничения (min/max для длин)
    """
    
    SMOOTHING_WINDOW = 8          # окно сглаживания
    MIN_UPPER_ARM_M = 0.12        # мин. длина плеча (м)
    MAX_UPPER_ARM_M = 0.50        # макс. длина плеча (м)
    MIN_FOREARM_M = 0.10          # мин. длина предплечья (м)
    MAX_FOREARM_M = 0.45          # макс. длина предплечья (м)
    
    def __init__(self, user_scale: float = 1.0):
        """
        Args:
            user_scale: калибровочный коэффициент (1.0 = без коррекции)
        """
        self.user_scale = user_scale
        
        # Буферы для сглаживания
        self._shoulder_angles_buffer = []
        self._elbow_angles_buffer = []
        self._upper_buffer = []
        self._forearm_buffer = []
        
        self._frame_count = 0
    
    # ==================================================================
    # ПУБЛИЧНЫЙ ИНТЕРФЕЙС
    # ==================================================================
    
    def process(self, raw_points: Dict) -> Optional[NormalizedGeometry]:
        """Обработка одного кадра"""
        if raw_points is None:
            return None
        
        self._frame_count += 1
        
        # Длины — евклидово расстояние в 3D
        raw_upper = float(np.linalg.norm(
            raw_points['shoulder'] - raw_points['elbow']
        ))
        raw_forearm = float(np.linalg.norm(
            raw_points['elbow'] - raw_points['wrist']
        ))
        raw_shoulder_width = float(np.linalg.norm(
            raw_points['shoulder'] - raw_points['other_shoulder']
        ))
        
        # Анатомическая проверка + сглаживание
        upper_arm = np.clip(raw_upper, self.MIN_UPPER_ARM_M, self.MAX_UPPER_ARM_M)
        forearm = np.clip(raw_forearm, self.MIN_FOREARM_M, self.MAX_FOREARM_M)
        
        upper_arm = self._smooth(self._upper_buffer, upper_arm)
        forearm = self._smooth(self._forearm_buffer, forearm)
        
        # Углы
        shoulder_angle = self._compute_shoulder_angle(raw_points)
        elbow_angle = self._compute_elbow_angle(raw_points)
        
        shoulder_angle = self._smooth(self._shoulder_angles_buffer, shoulder_angle)
        elbow_angle = self._smooth(self._elbow_angles_buffer, elbow_angle)
        
        # Статистика
        upper_pct = (upper_arm / (AVG_UPPER_ARM * self.user_scale)) * 100
        forearm_pct = (forearm / (AVG_FOREARM * self.user_scale)) * 100
        brachial = forearm / upper_arm if upper_arm > 0 else 0.0
        
        # Уверенность трекинга
        confidence = float(np.mean([
            raw_points.get('vis_shoulder', 0),
            raw_points.get('vis_elbow', 0),
            raw_points.get('vis_wrist', 0),
            raw_points.get('vis_other_shoulder', 0)
        ]))
        
        return NormalizedGeometry(
            upper_arm_m=round(upper_arm, 4),
            forearm_m=round(forearm, 4),
            shoulder_angle=round(shoulder_angle, 1),
            elbow_angle=round(elbow_angle, 1),
            scale_factor=1.0,
            shoulder_width=round(raw_shoulder_width, 4),
            upper_arm_percent=round(upper_pct, 1),
            forearm_percent=round(forearm_pct, 1),
            brachial_index=round(brachial, 3),
            tracking_confidence=round(confidence, 4)
        )
    
    def calibrate(self, measured_upper_arm_m: float) -> float:
        """
        Калибровка под конкретного пользователя.
        
        Args:
            measured_upper_arm_m: реальная длина плеча в метрах
        
        Returns:
            Новый user_scale
        """
        self.user_scale = measured_upper_arm_m / AVG_UPPER_ARM
        return self.user_scale
    
    def reset(self):
        """Сброс состояния (для нового видео/человека)"""
        self._shoulder_angles_buffer.clear()
        self._elbow_angles_buffer.clear()
        self._upper_buffer.clear()
        self._forearm_buffer.clear()
        self._frame_count = 0
    
    def get_stats(self) -> Dict:
        """Статистика для отладки"""
        return {
            'frame_count': self._frame_count,
            'frozen_count': 0,
            'frozen_percent': 0.0
        }
    
    # ==================================================================
    # ВЫЧИСЛЕНИЕ УГЛОВ
    # ==================================================================
    
    def _compute_shoulder_angle(self, points: Dict) -> float:
        """
        Угол плеча в сагиттальной плоскости (YZ world-координат).
        
        Система координат World Landmarks:
        - X: вправо
        - Y: вверх
        - Z: вперёд (от камеры)
        
        0° — рука висит вертикально вниз
        +90° — рука поднята вперёд
        -90° — рука отведена назад
        """
        shoulder = points['shoulder']
        elbow = points['elbow']
        
        # Вектор плечо → локоть
        vec = shoulder - elbow  # направлен вверх при висящей руке
        
        # Проекция на сагиттальную плоскость YZ
        yz_vec = np.array([vec[1], vec[2]])  # Y (вверх), Z (вперёд)
        vertical = np.array([1.0, 0.0])       # чисто вверх
        
        norm = np.linalg.norm(yz_vec)
        if norm < 0.001:
            return 0.0
        
        # Косинус угла между вектором и вертикалью
        cos_angle = np.dot(vertical, yz_vec) / norm
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        # Знак: Z-компонента вектора
        # vec[2] = Z_shoulder - Z_elbow
        # Если Z_elbow > Z_shoulder (локоть дальше от камеры), то vec[2] < 0
        # Это значит рука направлена вперёд → положительный угол
        if vec[2] < 0:
            angle = -angle
        
        return float(np.clip(angle, -90.0, 90.0))
    
    def _compute_elbow_angle(self, points: Dict) -> float:
        """
        Угол в локтевом суставе (3D).
        
        0° — полностью разогнут
        130° — максимально согнут
        """
        upper = points['shoulder'] - points['elbow']
        lower = points['wrist'] - points['elbow']
        
        norm_u = np.linalg.norm(upper)
        norm_l = np.linalg.norm(lower)
        
        if norm_u < 0.001 or norm_l < 0.001:
            return 0.0
        
        cos_angle = np.dot(upper, lower) / (norm_u * norm_l)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return float(np.clip(angle, 0.0, 130.0))
    
    # ==================================================================
    # СГЛАЖИВАНИЕ
    # ==================================================================
    
    def _smooth(self, buffer: list, new_value: float) -> float:
        """
        Скользящее среднее с защитой от выбросов.
        
        Если новое значение отличается от среднего в буфере более чем на 30%,
        оно считается выбросом и заменяется на среднее.
        """
        buffer.append(new_value)
        if len(buffer) > self.SMOOTHING_WINDOW:
            buffer.pop(0)
        
        if len(buffer) >= 3:
            mean_val = np.mean(buffer)
            # Проверка на выброс
            if abs(new_value - mean_val) / mean_val > 0.30:
                buffer[-1] = mean_val  # заменяем последнее значение на среднее
        
        return float(np.mean(buffer))


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    """
    Быстрый тест на синтетических данных.
    """
    print("\n" + "=" * 55)
    print("  ТЕСТ GeometryCalculator (3D World Landmarks)")
    print("=" * 55)
    
    calc = GeometryCalculator()
    
    # Симуляция: рука висит вертикально
    points = {
        'shoulder':       np.array([0.15, 0.4, -0.05]),   # правое плечо
        'elbow':          np.array([0.15, 0.1, -0.05]),   # локоть под плечом
        'wrist':          np.array([0.15, -0.15, -0.05]), # запястье ниже
        'other_shoulder': np.array([-0.15, 0.4, -0.05]),  # левое плечо
        'vis_shoulder':       1.0,
        'vis_elbow':          1.0,
        'vis_wrist':          1.0,
        'vis_other_shoulder': 1.0,
        'frame_width':   1920,
        'frame_height':  1080,
    }
    
    geo = calc.process(points)
    
    if geo:
        print(f"\n  Плечо: {geo.upper_arm_m:.3f} м (ожидалось ~0.30 м)")
        print(f"  Предплечье: {geo.forearm_m:.3f} м (ожидалось ~0.25 м)")
        print(f"  Ширина плеч: {geo.shoulder_width:.3f} м (ожидалось ~0.30 м)")
        print(f"  Угол плеча: {geo.shoulder_angle:.1f}° (ожидалось ~0°)")
        print(f"  Угол локтя: {geo.elbow_angle:.1f}° (ожидалось ~0°)")
        print(f"  Брахиальный индекс: {geo.brachial_index}")
        print(f"  Проценты: плечо {geo.upper_arm_percent:.1f}%, "
              f"предплечье {geo.forearm_percent:.1f}%")
    
    print("=" * 55)