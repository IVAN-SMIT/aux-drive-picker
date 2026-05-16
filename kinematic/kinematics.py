"""
kinematics/kinematics.py

МОДУЛЬ КИНЕМАТИКИ РУКИ.

ОТВЕТСТВЕННОСТЬ:
- Аналитический расчёт углов и положений звеньев
- Проверка диапазонов движения (0°–120° для локтя)
- Зависимость момента в плече от угла
- Расчёт положения конечной точки (запястья) в 2D
- Кинематические ограничения

ВХОД:  углы из GeometryCalculator (градусы), длины звеньев (м)
ВЫХОД: проверенные углы, координаты суставов, кинематические параметры


Авторы: Иван Смирнов, Леонид Нагорный

Тг: https://t.me/ivansmittt
Cайт: auxexo.ru
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from config.parameters import AVG_UPPER_ARM, AVG_FOREARM


# ============================================================================
# СТРУКТУРЫ ДАННЫХ
# ============================================================================

@dataclass
class JointPosition:
    """Положение сустава в 2D (сагиттальная плоскость)"""
    x: float  # горизонталь (вперёд+)
    y: float  # вертикаль (вверх+)
    name: str

@dataclass
class ArmKinematics:
    """Полное кинематическое состояние руки"""
    shoulder_angle: float      # градусы
    elbow_angle: float         # градусы
    shoulder_pos: JointPosition
    elbow_pos: JointPosition
    wrist_pos: JointPosition
    upper_arm_length: float    # метры
    forearm_length: float      # метры
    is_valid: bool             # углы в допустимых пределах
    violation_message: str = ""


# ============================================================================
# КАЛЬКУЛЯТОР КИНЕМАТИКИ
# ============================================================================

class KinematicsCalculator:
    """
    Калькулятор кинематики двухзвенной руки.
    
    Система координат (сагиттальная плоскость XY):
    - X: вперёд (от тела)
    - Y: вверх
    - Начало координат: плечевой сустав (0, 0)
    
    Кинематическая модель:
    - Звено 1 (плечо): длина L1, угол θ₁ от вертикали
    - Звено 2 (предплечье): длина L2, угол θ₂ от продолжения звена 1
    
    Прямая задача: по углам → координаты суставов
    """
    
    # Диапазоны движения (градусы)
    SHOULDER_RANGE = (-90.0, 90.0)    # отведение назад / подъём вперёд
    ELBOW_RANGE = (0.0, 130.0)        # разогнут / согнут
    
    def __init__(
        self,
        upper_arm_length: float = AVG_UPPER_ARM,
        forearm_length: float = AVG_FOREARM
    ):
        """
        Args:
            upper_arm_length: длина плеча (м)
            forearm_length: длина предплечья (м)
        """
        self.L1 = upper_arm_length   # плечо
        self.L2 = forearm_length     # предплечье
    
    # ==================================================================
    # ПРОВЕРКА ДИАПАЗОНОВ
    # ==================================================================
    
    def validate_shoulder(self, angle_deg: float) -> Tuple[float, bool, str]:
        """
        Проверка и ограничение угла плеча.
        
        Returns:
            (ограниченный_угол, валиден, сообщение)
        """
        min_a, max_a = self.SHOULDER_RANGE
        
        if min_a <= angle_deg <= max_a:
            return (angle_deg, True, "")
        
        limited = np.clip(angle_deg, min_a, max_a)
        return (limited, False, f"Плечо {angle_deg:.1f}° вне диапазона [{min_a}°, {max_a}°]")
    
    def validate_elbow(self, angle_deg: float) -> Tuple[float, bool, str]:
        """
        Проверка и ограничение угла локтя.
        
        Returns:
            (ограниченный_угол, валиден, сообщение)
        """
        min_a, max_a = self.ELBOW_RANGE
        
        if min_a <= angle_deg <= max_a:
            return (angle_deg, True, "")
        
        limited = np.clip(angle_deg, min_a, max_a)
        return (limited, False, f"Локоть {angle_deg:.1f}° вне диапазона [{min_a}°, {max_a}°]")
    
    # ==================================================================
    # ПРЯМАЯ КИНЕМАТИКА
    # ==================================================================
    
    def forward_kinematics(
        self,
        shoulder_angle_deg: float,
        elbow_angle_deg: float
    ) -> ArmKinematics:
        """
        Прямая кинематика: углы → координаты суставов.
        
        Args:
            shoulder_angle_deg: угол плеча от вертикали (0° = рука вниз)
            elbow_angle_deg: угол локтя (0° = разогнут, 130° = макс. сгибание)
        
        Returns:
            ArmKinematics с координатами и статусом
        """
        # Проверка диапазонов
        sh_angle, sh_valid, sh_msg = self.validate_shoulder(shoulder_angle_deg)
        el_angle, el_valid, el_msg = self.validate_elbow(elbow_angle_deg)
        
        is_valid = sh_valid and el_valid
        messages = []
        if sh_msg:
            messages.append(sh_msg)
        if el_msg:
            messages.append(el_msg)
        
        # Плечевой сустав — начало координат
        shoulder_pos = JointPosition(x=0.0, y=0.0, name="shoulder")
        
        # Локоть
        sh_rad = np.radians(sh_angle)
        elbow_x = self.L1 * np.sin(sh_rad)
        elbow_y = -self.L1 * np.cos(sh_rad)
        elbow_pos = JointPosition(
            x=round(elbow_x, 4),
            y=round(elbow_y, 4),
            name="elbow"
        )
        
        # Запястье: продолжение от локтя с учётом угла локтя
        el_rad = np.radians(el_angle)
        # Абсолютный угол предплечья = угол плеча + угол локтя
        forearm_abs_angle = sh_rad + el_rad
        
        wrist_x = elbow_x + self.L2 * np.sin(forearm_abs_angle)
        wrist_y = elbow_y - self.L2 * np.cos(forearm_abs_angle)
        wrist_pos = JointPosition(
            x=round(wrist_x, 4),
            y=round(wrist_y, 4),
            name="wrist"
        )
        
        return ArmKinematics(
            shoulder_angle=sh_angle,
            elbow_angle=el_angle,
            shoulder_pos=shoulder_pos,
            elbow_pos=elbow_pos,
            wrist_pos=wrist_pos,
            upper_arm_length=self.L1,
            forearm_length=self.L2,
            is_valid=is_valid,
            violation_message="; ".join(messages) if messages else ""
        )
    
    def get_wrist_position(
        self,
        shoulder_angle_deg: float,
        elbow_angle_deg: float
    ) -> Tuple[float, float]:
        """
        Быстрое получение координат запястья без полной кинематики.
        
        Returns:
            (x, y) в метрах
        """
        kin = self.forward_kinematics(shoulder_angle_deg, elbow_angle_deg)
        return (kin.wrist_pos.x, kin.wrist_pos.y)
    
    # ==================================================================
    # ОБРАТНАЯ КИНЕМАТИКА
    # ==================================================================
    
    def inverse_kinematics(
        self,
        target_x: float,
        target_y: float,
        elbow_up: bool = True
    ) -> Optional[Tuple[float, float]]:
        """
        Обратная кинематика: координаты запястья → углы.
        
        Args:
            target_x, target_y: желаемое положение запястья (м)
            elbow_up: True = локоть вверх, False = локоть вниз
        
        Returns:
            (shoulder_angle, elbow_angle) в градусах или None
        """
        # Расстояние от плеча до цели
        r = np.sqrt(target_x**2 + target_y**2)
        
        # Проверка достижимости
        if r > self.L1 + self.L2 or r < abs(self.L1 - self.L2):
            return None
        
        # Угол цели от вертикали
        target_angle = np.arctan2(target_x, -target_y)
        
        # Косинус угла локтя (закон косинусов)
        cos_elbow = (r**2 - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
        cos_elbow = np.clip(cos_elbow, -1.0, 1.0)
        
        elbow_rad = np.arccos(cos_elbow)
        if not elbow_up:
            elbow_rad = -elbow_rad
        
        # Угол плеча
        beta = np.arctan2(
            self.L2 * np.sin(elbow_rad),
            self.L1 + self.L2 * np.cos(elbow_rad)
        )
        
        shoulder_rad = target_angle - beta
        
        shoulder_deg = np.degrees(shoulder_rad)
        elbow_deg = np.degrees(elbow_rad)
        
        # Проверка диапазонов
        _, sh_valid, _ = self.validate_shoulder(shoulder_deg)
        _, el_valid, _ = self.validate_elbow(elbow_deg)
        
        if not sh_valid or not el_valid:
            return None
        
        return (round(shoulder_deg, 1), round(elbow_deg, 1))
    
    # ==================================================================
    # ЗАВИСИМОСТЬ МОМЕНТА В ПЛЕЧЕ ОТ УГЛА
    # ==================================================================
    
    def shoulder_moment_angle_dependency(
        self,
        shoulder_angle_deg: float,
        elbow_angle_deg: float = 0.0,
        forearm_mass_kg: float = 1.8,
        hand_mass_kg: float = 0.5
    ) -> float:
        """
        Зависимость момента в плече от угла плеча.
        
        Момент создаётся весом предплечья и кисти,
        приложенным к центру масс звеньев.
        
        Args:
            shoulder_angle_deg: угол плеча
            elbow_angle_deg: угол локтя (влияет на положение центра масс)
            forearm_mass_kg: масса предплечья
            hand_mass_kg: масса кисти
        
        Returns:
            Момент в плечевом суставе (Н·м)
        """
        g = 9.81
        
        # Центр масс предплечья (примерно на 40% длины от локтя)
        com_forearm_ratio = 0.4
        
        sh_rad = np.radians(shoulder_angle_deg)
        el_rad = np.radians(elbow_angle_deg)
        
        # Положение локтя
        elbow_x = self.L1 * np.sin(sh_rad)
        
        # Положение центра масс предплечья
        forearm_abs_angle = sh_rad + el_rad
        com_forearm_x = elbow_x + com_forearm_ratio * self.L2 * np.sin(forearm_abs_angle)
        
        # Положение кисти (конец предплечья)
        hand_x = elbow_x + self.L2 * np.sin(forearm_abs_angle)
        
        # Момент от силы тяжести (F × плечо)
        torque_forearm = forearm_mass_kg * g * com_forearm_x
        torque_hand = hand_mass_kg * g * hand_x
        
        return round(float(torque_forearm + torque_hand), 3)
    
    # ==================================================================
    # СТАТИСТИКА
    # ==================================================================
    
    def movement_range_percent(self, elbow_angle_deg: float) -> float:
        """
        Процент использования диапазона движения локтя.
        
        0° = полностью разогнут (0%)
        130° = максимально согнут (100%)
        """
        min_a, max_a = self.ELBOW_RANGE
        return ((elbow_angle_deg - min_a) / (max_a - min_a)) * 100
    
    def get_workspace_info(self) -> dict:
        """
        Информация о рабочей зоне руки.
        
        Returns:
            Словарь с границами достижимости
        """
        max_reach = self.L1 + self.L2
        min_reach = abs(self.L1 - self.L2)
        
        return {
            'max_reach_m': round(max_reach, 3),
            'min_reach_m': round(min_reach, 3),
            'workspace_radius_m': round(max_reach, 3),
            'upper_arm_m': self.L1,
            'forearm_m': self.L2
        }


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  ТЕСТ КИНЕМАТИКИ РУКИ")
    print("=" * 55)
    
    kin = KinematicsCalculator()
    
    # Информация о рабочей зоне
    ws = kin.get_workspace_info()
    print(f"\n  Рабочая зона:")
    print(f"  Макс. досягаемость: {ws['max_reach_m']} м")
    print(f"  Мин. досягаемость:  {ws['min_reach_m']} м")
    print(f"  Плечо: {ws['upper_arm_m']} м, Предплечье: {ws['forearm_m']} м")
    
    # Тест 1: прямая кинематика
    print(f"\n  Тест 1: Прямая кинематика")
    test_angles = [
        (0, 0, "рука висит"),
        (45, 0, "плечо 45°"),
        (90, 0, "плечо горизонтально"),
        (0, 90, "локоть 90°"),
        (45, 90, "оба согнуты"),
        (90, 130, "макс. сгибание"),
        (-30, 60, "плечо назад"),
        (100, 0, "вне диапазона"),
    ]
    
    for sh, el, desc in test_angles:
        kin_result = kin.forward_kinematics(sh, el)
        status = "✅" if kin_result.is_valid else "❌"
        print(f"  {status} {desc}: пл={sh}°, лк={el}° → "
              f"запястье=({kin_result.wrist_pos.x:.3f}, {kin_result.wrist_pos.y:.3f}) м"
              f"{' (' + kin_result.violation_message + ')' if not kin_result.is_valid else ''}")
    
    # Тест 2: обратная кинематика
    print(f"\n  Тест 2: Обратная кинематика")
    test_targets = [
        (0.2, -0.4, "вперёд-вниз"),
        (0.0, -0.5, "прямо вниз"),
        (0.3, -0.1, "вперёд"),
    ]
    
    for tx, ty, desc in test_targets:
        ik = kin.inverse_kinematics(tx, ty)
        if ik:
            print(f"  ✅ {desc}: цель=({tx}, {ty}) м → пл={ik[0]:.1f}°, лк={ik[1]:.1f}°")
        else:
            print(f"  ❌ {desc}: цель=({tx}, {ty}) м → НЕДОСТИЖИМА")
    
    # Тест 3: момент в плече
    print(f"\n  Тест 3: Момент в плече от угла")
    for sh in [0, 30, 60, 90]:
        torque = kin.shoulder_moment_angle_dependency(sh, elbow_angle_deg=0)
        print(f"  Плечо={sh}° → момент={torque:.3f} Н·м")
    
    # Тест 4: диапазон движения
    print(f"\n  Тест 4: Диапазон движения локтя")
    for el in [0, 30, 65, 90, 130]:
        pct = kin.movement_range_percent(el)
        bar = "█" * int(pct / 10)
        print(f"  Локоть={el:3d}° → {pct:5.1f}% {bar}")
    
    print("=" * 55)