"""
biomechanics/biomechanics.py

МОДУЛЬ БИОМЕХАНИКИ: расчёт моментов в суставах руки.

ОТВЕТСТВЕННОСТЬ:
- Расчёт момента в локтевом суставе: M(θ) = m·g·l·sin(θ)
- Расчёт момента в плечевом суставе с учётом позы
- Учёт внешней нагрузки (груз в руке)
- Построение графика M(θ) для отчёта
- Определение максимального момента Mmax

ВХОД:  углы из GeometryCalculator, параметры из config
ВЫХОД: Mmax, график M(θ), таблица моментов


Авторы: Иван Смирнов, Леонид Нагорный

Тг: https://t.me/ivansmittt
Cайт: auxexo.ru
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

from config.parameters import AVG_UPPER_ARM, AVG_FOREARM


# ============================================================================
# СТРУКТУРЫ ДАННЫХ
# ============================================================================

@dataclass
class JointMoments:
    """Моменты в суставах для заданной позы"""
    shoulder_angle: float      # градусы
    elbow_angle: float         # градусы
    shoulder_torque: float     # Н·м
    elbow_torque: float        # Н·м
    external_load_kg: float    # кг
    total_elbow_mass: float    # кг (предплечье + кисть + груз)


@dataclass
class MaxMomentResult:
    """Результат поиска максимального момента"""
    max_shoulder_torque: float
    max_elbow_torque: float
    shoulder_at_max: float     # угол плеча при макс. моменте
    elbow_at_max: float        # угол локтя при макс. моменте
    external_load_kg: float
    safety_factor: float = 1.3


# ============================================================================
# КАЛЬКУЛЯТОР БИОМЕХАНИКИ
# ============================================================================

class BiomechanicsCalculator:
    """
    Расчёт моментов в суставах руки по аналитическим формулам.
    
    Формула для локтя:
        M(θ) = m_eff · g · l_com · sin(θ)
    
    где:
        m_eff = m_forearm + m_hand + m_load  — эффективная масса
        l_com — расстояние от локтя до центра масс
        θ — угол локтя (0° = разогнут, 90° = горизонтально)
    
    Для плеча:
        Учитывается масса всей руки (плечо + предплечье + кисть + груз)
        и их центры масс.
    """
    
    # Гравитация
    G = 9.81  # м/с²
    
    # Антропометрические параметры (из литературы)
    FOREARM_MASS_KG = 1.8       # масса предплечья
    HAND_MASS_KG = 0.5          # масса кисти
    UPPER_ARM_MASS_KG = 2.2     # масса плеча
    
    # Положения центров масс (в долях длины сегмента от проксимального сустава)
    UPPER_ARM_COM_RATIO = 0.45   # центр масс плеча
    FOREARM_COM_RATIO = 0.40     # центр масс предплечья
    
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
        self.L1 = upper_arm_length
        self.L2 = forearm_length
        
        # Расстояния до центров масс (м)
        self.upper_arm_com = self.L1 * self.UPPER_ARM_COM_RATIO
        self.forearm_com = self.L2 * self.FOREARM_COM_RATIO
    
    # ==================================================================
    # РАСЧЁТ МОМЕНТОВ
    # ==================================================================
    
    def compute_elbow_torque(
        self,
        elbow_angle_deg: float,
        external_load_kg: float = 0.0
    ) -> float:
        """
        Расчёт момента в локтевом суставе.
        
        M(θ) = (m_forearm · g · l_com_forearm + m_hand · g · L2 + m_load · g · L2) · sin(θ)
        
        Args:
            elbow_angle_deg: угол локтя (0° = разогнут, 90° = горизонтально)
            external_load_kg: масса груза в руке
        
        Returns:
            Момент в Н·м
        """
        theta_rad = np.radians(np.clip(elbow_angle_deg, 0, 130))
        
        # Момент от предплечья
        torque_forearm = self.FOREARM_MASS_KG * self.G * self.forearm_com * np.sin(theta_rad)
        
        # Момент от кисти
        torque_hand = self.HAND_MASS_KG * self.G * self.L2 * np.sin(theta_rad)
        
        # Момент от груза
        torque_load = external_load_kg * self.G * self.L2 * np.sin(theta_rad)
        
        total = torque_forearm + torque_hand + torque_load
        
        return round(float(total), 3)
    
    def compute_shoulder_torque(
        self,
        shoulder_angle_deg: float,
        elbow_angle_deg: float = 0.0,
        external_load_kg: float = 0.0,
        include_upper_arm: bool = False
    ) -> float:
        """
        Расчёт момента в плечевом суставе.
        
        Учитывает массу предплечья, кисти и груза.
        Масса плеча опциональна (include_upper_arm=True для полного анализа,
        False для сравнения с MuJoCo где плечо закреплено на торсе).
        
        Args:
            shoulder_angle_deg: угол плеча (0° = вертикально вниз)
            elbow_angle_deg: угол локтя
            external_load_kg: масса груза
            include_upper_arm: включать ли массу плеча
        
        Returns:
            Момент в Н·м (абсолютное значение)
        """
        sh_rad = np.radians(np.clip(shoulder_angle_deg, -90, 90))
        el_rad = np.radians(np.clip(elbow_angle_deg, 0, 130))
        
        # Положение локтя
        elbow_x = self.L1 * np.sin(sh_rad)
        
        # Абсолютный угол предплечья
        forearm_abs_angle = sh_rad + el_rad
        
        # Плечо момента для центра масс предплечья
        com_forearm_x = elbow_x + self.forearm_com * np.sin(forearm_abs_angle)
        torque_forearm = self.FOREARM_MASS_KG * self.G * abs(com_forearm_x)
        
        # Плечо момента для кисти
        hand_x = elbow_x + self.L2 * np.sin(forearm_abs_angle)
        torque_hand = self.HAND_MASS_KG * self.G * abs(hand_x)
        
        # Момент от груза
        torque_load = external_load_kg * self.G * abs(hand_x)
        
        total = torque_forearm + torque_hand + torque_load
        
        # Опционально: масса плеча
        if include_upper_arm:
            moment_arm_upper = self.upper_arm_com * np.sin(sh_rad)
            torque_upper = self.UPPER_ARM_MASS_KG * self.G * abs(moment_arm_upper)
            total += torque_upper
        
        return round(float(total), 3)
    
    def compute_all(
        self,
        shoulder_angle_deg: float,
        elbow_angle_deg: float,
        external_load_kg: float = 0.0
    ) -> JointMoments:
        """
        Расчёт моментов в обоих суставах.
        
        Returns:
            JointMoments с заполненными полями
        """
        total_mass = (
            self.FOREARM_MASS_KG +
            self.HAND_MASS_KG +
            external_load_kg
        )
        
        return JointMoments(
            shoulder_angle=shoulder_angle_deg,
            elbow_angle=elbow_angle_deg,
            shoulder_torque=self.compute_shoulder_torque(
                shoulder_angle_deg, elbow_angle_deg, external_load_kg
            ),
            elbow_torque=self.compute_elbow_torque(
                elbow_angle_deg, external_load_kg
            ),
            external_load_kg=external_load_kg,
            total_elbow_mass=round(total_mass, 2)
        )
    
    # ==================================================================
    # ГРАФИК M(θ)
    # ==================================================================
    
    def generate_moment_curve(
        self,
        joint: str = "elbow",
        external_load_kg: float = 0.0,
        num_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерация кривой M(θ) для построения графика.
        
        Args:
            joint: "elbow" или "shoulder"
            external_load_kg: масса груза
            num_points: количество точек
        
        Returns:
            (angles_deg, torques_Nm) — массивы для графика
        """
        if joint == "elbow":
            angles = np.linspace(0, 130, num_points)
            torques = np.array([
                self.compute_elbow_torque(a, external_load_kg)
                for a in angles
            ])
        else:
            angles = np.linspace(0, 90, num_points)
            torques = np.array([
                self.compute_shoulder_torque(a, 0, external_load_kg)
                for a in angles
            ])
        
        return angles, torques
    
    def plot_moment_curve(
        self,
        save_path: Optional[str] = None,
        external_loads: List[float] = None
    ):
        """
        Построение графика M(θ) для разных нагрузок.
        
        Args:
            save_path: путь для сохранения (если None — показ на экране)
            external_loads: список нагрузок (по умолчанию [0, 2, 5])
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  matplotlib не установлен")
            return
        
        if external_loads is None:
            external_loads = [0.0, 2.0, 5.0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Локоть
        for load in external_loads:
            angles, torques = self.generate_moment_curve("elbow", load)
            ax1.plot(angles, torques, linewidth=2, label=f"{load} кг")
        
        ax1.set_xlabel("Угол локтя θ (градусы)")
        ax1.set_ylabel("Момент M (Н·м)")
        ax1.set_title("M(θ) — Локтевой сустав")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 130)
        
        # Плечо
        for load in external_loads:
            angles, torques = self.generate_moment_curve("shoulder", load)
            ax2.plot(angles, torques, linewidth=2, label=f"{load} кг")
        
        ax2.set_xlabel("Угол плеча θ (градусы)")
        ax2.set_ylabel("Момент M (Н·м)")
        ax2.set_title("M(θ) — Плечевой сустав")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 90)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 График сохранён: {save_path}")
        else:
            plt.show()
    
    # ==================================================================
    # МАКСИМАЛЬНЫЙ МОМЕНТ
    # ==================================================================
    
    def find_max_moment(
        self,
        external_load_kg: float = 0.0,
        safety_factor: float = 1.3
    ) -> MaxMomentResult:
        """
        Поиск максимального момента для подбора привода.
        
        Перебирает углы и находит максимум.
        
        Args:
            external_load_kg: масса груза
            safety_factor: коэффициент запаса
        
        Returns:
            MaxMomentResult с рекомендациями
        """
        max_elbow = 0.0
        max_shoulder = 0.0
        elbow_at_max = 0.0
        shoulder_at_max = 0.0
        
        # Поиск максимума по локтю
        for angle in np.linspace(0, 130, 100):
            torque = self.compute_elbow_torque(angle, external_load_kg)
            if torque > max_elbow:
                max_elbow = torque
                elbow_at_max = angle
        
        # Поиск максимума по плечу (при максимально нагруженной позе)
        for sh_angle in np.linspace(0, 90, 100):
            torque = self.compute_shoulder_torque(
                sh_angle, elbow_at_max, external_load_kg
            )
            if torque > max_shoulder:
                max_shoulder = torque
                shoulder_at_max = sh_angle
        
        return MaxMomentResult(
            max_shoulder_torque=round(max_shoulder, 3),
            max_elbow_torque=round(max_elbow, 3),
            shoulder_at_max=round(shoulder_at_max, 1),
            elbow_at_max=round(elbow_at_max, 1),
            external_load_kg=external_load_kg,
            safety_factor=safety_factor
        )
    
    def get_actuator_requirements(
        self,
        external_load_kg: float = 0.0,
        safety_factor: float = 1.3
    ) -> Dict:
        """
        Требования к приводам для подбора.
        
        Returns:
            Словарь с ключами:
            - 'M_shoulder_required': требуемый момент плеча (Н·м)
            - 'M_elbow_required': требуемый момент локтя (Н·м)
            - 'max_shoulder_raw': сырой максимум (Н·м)
            - 'max_elbow_raw': сырой максимум (Н·м)
        """
        result = self.find_max_moment(external_load_kg, safety_factor)
        
        return {
            'M_shoulder_required': round(result.max_shoulder_torque * safety_factor, 3),
            'M_elbow_required': round(result.max_elbow_torque * safety_factor, 3),
            'max_shoulder_raw': result.max_shoulder_torque,
            'max_elbow_raw': result.max_elbow_torque,
            'shoulder_at_max': result.shoulder_at_max,
            'elbow_at_max': result.elbow_at_max,
            'external_load_kg': external_load_kg,
            'safety_factor': safety_factor
        }
    
    # ==================================================================
    # ОБРАБОТКА ДАТАСЕТА
    # ==================================================================
    
    def process_dataset(
        self,
        angles_data: List[Dict],
        external_load_kg: float = 0.0
    ) -> List[Dict]:
        """
        Обработка всего датасета углов → расчёт моментов.
        
        Args:
            angles_data: список словарей с ключами 'shoulder_angle', 'elbow_angle'
            external_load_kg: масса груза
        
        Returns:
            Список с добавленными ключами 'shoulder_torque', 'elbow_torque'
        """
        results = []
        
        for row in angles_data:
            sh = float(row.get('shoulder_angle', 0))
            el = float(row.get('elbow_angle', 0))
            
            moments = self.compute_all(sh, el, external_load_kg)
            
            result_row = dict(row)
            result_row['shoulder_torque'] = moments.shoulder_torque
            result_row['elbow_torque'] = moments.elbow_torque
            result_row['total_elbow_mass'] = moments.total_elbow_mass
            
            results.append(result_row)
        
        return results


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  ТЕСТ БИОМЕХАНИКИ")
    print("=" * 55)
    
    bio = BiomechanicsCalculator()
    
    # Тест 1: формула M(θ) для локтя
    print(f"\n  Тест 1: M(θ) для локтя")
    print(f"  {'Угол':<10} {'Без груза':<12} {'2 кг':<12} {'5 кг':<12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
    for angle in [0, 30, 60, 90, 120, 130]:
        t0 = bio.compute_elbow_torque(angle, 0.0)
        t2 = bio.compute_elbow_torque(angle, 2.0)
        t5 = bio.compute_elbow_torque(angle, 5.0)
        print(f"  {angle:3d}°      {t0:6.3f} Н·м    {t2:6.3f} Н·м    {t5:6.3f} Н·м")
    
    # Тест 2: моменты в плече
    print(f"\n  Тест 2: Моменты в плече (локоть разогнут)")
    for angle in [0, 30, 60, 90]:
        t = bio.compute_shoulder_torque(angle, 0, 0.0)
        print(f"  Плечо={angle:3d}° → {t:.3f} Н·м")
    
    # Тест 3: максимальные моменты
    print(f"\n  Тест 3: Максимальные моменты")
    for load in [0.0, 2.0, 5.0]:
        req = bio.get_actuator_requirements(load)
        print(f"\n  Нагрузка {load} кг:")
        print(f"    Плечо:  Mmax={req['max_shoulder_raw']:.3f} Н·м, "
              f"требуемый={req['M_shoulder_required']:.3f} Н·м")
        print(f"    Локоть: Mmax={req['max_elbow_raw']:.3f} Н·м, "
              f"требуемый={req['M_elbow_required']:.3f} Н·м")
        print(f"    Углы: плечо={req['shoulder_at_max']:.1f}°, "
              f"локоть={req['elbow_at_max']:.1f}°")
    
    # Тест 4: обработка датасета
    print(f"\n  Тест 4: Обработка датасета")
    test_data = [
        {'shoulder_angle': '0', 'elbow_angle': '0'},
        {'shoulder_angle': '45', 'elbow_angle': '90'},
        {'shoulder_angle': '90', 'elbow_angle': '130'},
    ]
    results = bio.process_dataset(test_data, external_load_kg=2.0)
    for r in results:
        print(f"  пл={r['shoulder_angle']:>5s}°, лк={r['elbow_angle']:>5s}° → "
              f"пл={r['shoulder_torque']:.3f} Н·м, лк={r['elbow_torque']:.3f} Н·м")
    
    # Тест 5: график (только если matplotlib есть)
    print(f"\n  Тест 5: График M(θ)")
    try:
        import matplotlib
        bio.plot_moment_curve()
        print(f"  ✅ График построен")
    except Exception as e:
        print(f"  ⚠️ Не удалось: {e}")
    
    print("=" * 55)