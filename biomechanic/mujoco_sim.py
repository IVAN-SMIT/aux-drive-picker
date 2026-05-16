"""
biomechanics/mujoco_sim.py

МОДУЛЬ ВИЗУАЛИЗАЦИИ: MuJoCo-симуляция двухзвенной руки.

ОТВЕТСТВЕННОСТЬ (после разделения):
- Загрузка XML-модели руки в MuJoCo
- Визуализация движения по углам из GeometryCalculator
- Валидация моментов (сравнение с BiomechanicsCalculator)
- Учёт внешней нагрузки (груз в руке)
- Только визуализация, без кинематики и биомеханики

ВХОД:  углы из GeometryCalculator (градусы), нагрузка (кг)
ВЫХОД: визуализация, измеренные моменты JointState

Авторы: Иван Смирнов, Леонид Нагорный

Тг: https://t.me/ivansmittt
Cайт: auxexo.ru
"""

import numpy as np
import time
import os
from typing import Optional, Tuple
from dataclasses import dataclass

try:
    import mujoco
    import mujoco.viewer
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


# ============================================================================
# СТРУКТУРА ДАННЫХ
# ============================================================================

@dataclass
class JointState:
    """Состояние суставов после шага симуляции"""
    shoulder_angle: float      # градусы (из сенсора)
    elbow_angle: float         # градусы (из сенсора)
    shoulder_torque: float     # Н·м (из сенсора actuatorfrc)
    elbow_torque: float        # Н·м
    timestamp: float           # секунды от запуска


# ============================================================================
# ЗАГРУЗКА XML
# ============================================================================

_XML_PATH = os.path.join(os.path.dirname(__file__), "arm_model.xml")

def _load_xml() -> str:
    """Загрузка XML-модели из файла"""
    with open(_XML_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================================
# MUJOCO ВИЗУАЛИЗАТОР
# ============================================================================

class MuJoCoArmSim:
    """
    MuJoCo-визуализатор двухзвенной руки.
    
    Использует углы из GeometryCalculator для управления моторами.
    Моменты измеряются через сенсоры actuatorfrc.
    
    Не выполняет кинематических или биомеханических расчётов —
    только визуализация и измерение.
    """
    
    def __init__(self, external_load_kg: float = 0.0):
        """
        Args:
            external_load_kg: масса груза в руке (0 = без груза)
        
        Raises:
            RuntimeError: если MuJoCo не установлен
        """
        if not HAS_MUJOCO:
            raise RuntimeError(
                "MuJoCo не установлен. Установите: pip install mujoco"
            )
        
        self.external_load_kg = external_load_kg
        
        # Загружаем модель
        self.model = mujoco.MjModel.from_xml_string(_load_xml())
        self.data = mujoco.MjData(self.model)
        
        # Устанавливаем массу груза
        if external_load_kg > 0:
            load_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "load"
            )
            if load_body_id >= 0:
                self.model.body_mass[load_body_id] = external_load_kg
                # Делаем груз видимым если есть масса
                self.model.geom_rgba[
                    mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, "load_geom"
                    )
                ] = [0.5, 0.5, 0.5, 1.0]  # непрозрачный
            else:
                print(f"⚠️  Тело 'load' не найдено в модели")
        
        # Сбрасываем в начальное положение
        mujoco.mj_resetData(self.model, self.data)
        
        # Запускаем вьювер (пассивный режим)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Начальные углы: рука висит вертикально
        self.data.ctrl[0] = 0.0   # плечо (радианы)
        self.data.ctrl[1] = 0.0   # локоть (радианы)
        
        self._start_time = time.time()
        self._frame_count = 0
        
        print(f"✅ MuJoCo запущен (нагрузка: {external_load_kg} кг)")
    
    # ==================================================================
    # УПРАВЛЕНИЕ
    # ==================================================================
    
    def set_angles(self, shoulder_deg: float, elbow_deg: float):
        """
        Установка углов из GeometryCalculator.
        
        Преобразование систем координат:
        - GeometryCalculator: 0° = рука вниз, + = вперёд
        - MuJoCo плечо: 0° = вертикально вниз, + = вперёд
        - MuJoCo локоть: 0° = разогнут, - = согнут (range отрицательный)
        
        Args:
            shoulder_deg: угол плеча (градусы)
            elbow_deg: угол локтя (градусы, 0 = разогнут, 130 = макс. сгибание)
        """
        # Плечо: прямое соответствие
        mujoco_shoulder = np.clip(shoulder_deg, -90, 90)
        
        # Локоть: GeometryCalculator 0°=разогнут → MuJoCo 0°=разогнут
        # MuJoCo использует отрицательные углы для сгибания
        mujoco_elbow = -np.clip(elbow_deg, 0, 130)
        
        # Позиционное управление (мгновенное)
        self.data.ctrl[0] = np.radians(mujoco_shoulder)
        self.data.ctrl[1] = np.radians(mujoco_elbow)
    
    # ==================================================================
    # СИМУЛЯЦИЯ
    # ==================================================================
    
    def step(self) -> JointState:
        """
        Один шаг физической симуляции + синхронизация вьювера.
        
        Returns:
            JointState с текущими измеренными значениями
        """
        # Шаг физики
        mujoco.mj_step(self.model, self.data)
        
        # Синхронизация вьювера
        if self.viewer:
            self.viewer.sync()
        
        self._frame_count += 1
        
        # Читаем сенсоры
        # sensordata[0] = shoulder_force (actuatorfrc)
        # sensordata[1] = elbow_force (actuatorfrc)
        # sensordata[2] = shoulder_pos (jointpos)
        # sensordata[3] = elbow_pos (jointpos)
        
        shoulder_angle = np.degrees(self.data.sensordata[2])
        elbow_angle = -np.degrees(self.data.sensordata[3])  # инвертируем знак
        
        shoulder_torque = self.data.sensordata[0]
        elbow_torque = self.data.sensordata[1]
        
        return JointState(
            shoulder_angle=round(float(shoulder_angle), 1),
            elbow_angle=round(float(elbow_angle), 1),
            shoulder_torque=round(float(shoulder_torque), 3),
            elbow_torque=round(float(elbow_torque), 3),
            timestamp=round(time.time() - self._start_time, 3)
        )
    
    def stabilize(self, steps: int = 50):
        """
        Стабилизация симуляции: несколько шагов без изменений.
        
        Args:
            steps: количество шагов стабилизации
        """
        for _ in range(steps):
            self.step()
    
    # ==================================================================
    # ВАЛИДАЦИЯ
    # ==================================================================
    
    def measure_static_torques(
        self,
        shoulder_deg: float,
        elbow_deg: float,
        settle_steps: int = 100
    ) -> JointState:
        """
        Измерение установившихся моментов для заданной позы.
        
        Устанавливает углы, ждёт стабилизации, возвращает моменты.
        Полезно для сравнения с аналитическими формулами.
        
        Args:
            shoulder_deg: угол плеча
            elbow_deg: угол локтя
            settle_steps: шагов для стабилизации
        
        Returns:
            JointState с установившимися моментами
        """
        self.set_angles(shoulder_deg, elbow_deg)
        self.stabilize(settle_steps)
        return self.step()
    
    def validate_against_formula(
        self,
        analytical_elbow_torque: float,
        analytical_shoulder_torque: float,
        shoulder_deg: float,
        elbow_deg: float,
        tolerance: float = 0.15
    ) -> dict:
        """
        Сравнение моментов из MuJoCo с аналитическими формулами.
        
        Берутся абсолютные значения — знак зависит от системы координат.
        
        Args:
            analytical_elbow_torque: момент локтя из BiomechanicsCalculator
            analytical_shoulder_torque: момент плеча из BiomechanicsCalculator
            shoulder_deg, elbow_deg: поза для измерения
            tolerance: допустимое относительное расхождение (0.15 = 15%)
        
        Returns:
            Словарь с результатами сравнения
        """
        measured = self.measure_static_torques(shoulder_deg, elbow_deg)
        
        # Берём абсолютные значения — знак зависит от системы координат
        measured_el = abs(measured.elbow_torque)
        measured_sh = abs(measured.shoulder_torque)
        analytical_el = abs(analytical_elbow_torque)
        analytical_sh = abs(analytical_shoulder_torque)
        
        # Относительная ошибка (по абсолютным значениям)
        if analytical_el > 0.001:
            elbow_error = abs(measured_el - analytical_el) / analytical_el
        else:
            elbow_error = abs(measured_el)  # оба должны быть ~0
        
        if analytical_sh > 0.001:
            shoulder_error = abs(measured_sh - analytical_sh) / analytical_sh
        else:
            shoulder_error = abs(measured_sh)
        
        elbow_ok = elbow_error < tolerance
        shoulder_ok = shoulder_error < tolerance
        
        return {
            'pose': f"пл={shoulder_deg}°, лк={elbow_deg}°",
            'elbow_analytical': round(analytical_elbow_torque, 3),
            'elbow_measured': round(measured.elbow_torque, 3),
            'elbow_abs_analytical': round(analytical_el, 3),
            'elbow_abs_measured': round(measured_el, 3),
            'elbow_error_pct': round(elbow_error * 100, 1),
            'elbow_valid': elbow_ok,
            'shoulder_analytical': round(analytical_shoulder_torque, 3),
            'shoulder_measured': round(measured.shoulder_torque, 3),
            'shoulder_abs_analytical': round(analytical_sh, 3),
            'shoulder_abs_measured': round(measured_sh, 3),
            'shoulder_error_pct': round(shoulder_error * 100, 1),
            'shoulder_valid': shoulder_ok,
            'overall_valid': elbow_ok and shoulder_ok
        }
    
    # ==================================================================
    # СЛУЖЕБНЫЕ
    # ==================================================================
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    @property
    def elapsed(self) -> float:
        return time.time() - self._start_time
    
    def close(self):
        """Закрытие симуляции и освобождение ресурсов"""
        if self.viewer:
            self.viewer.close()
        print("👋 MuJoCo завершён")


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    """
    Быстрый тест визуализации и валидации.
    """
    if not HAS_MUJOCO:
        print("❌ Установите MuJoCo: pip install mujoco")
        exit(1)
    
    print("\n" + "=" * 55)
    print("  ТЕСТ MUJOCO: визуализация + валидация")
    print("=" * 55)
    
    # Тест 1: базовая визуализация
    print("\n1. Базовая визуализация:")
    sim = MuJoCoArmSim(external_load_kg=0.0)
    
    test_sequence = [
        (0, 0, "рука висит"),
        (30, 0, "плечо 30°"),
        (60, 0, "плечо 60°"),
        (90, 0, "плечо горизонтально"),
        (90, 60, "локоть 60°"),
        (90, 120, "локоть 120°"),
        (0, 120, "плечо 0°, локоть 120°"),
        (0, 0, "исходная"),
    ]
    
    for sh, el, desc in test_sequence:
        state = sim.measure_static_torques(sh, el, settle_steps=50)
        print(f"  {desc:20s}: пл={state.shoulder_torque:6.3f} Н·м, "
              f"лк={state.elbow_torque:6.3f} Н·м")
    
    sim.close()
    
    # Тест 2: валидация против формул
    print("\n2. Валидация против BiomechanicsCalculator:")
    from biomechanics import BiomechanicsCalculator
    
    bio = BiomechanicsCalculator()
    sim = MuJoCoArmSim(external_load_kg=0.0)
    
    validation_poses = [(0, 0), (45, 0), (90, 0), (0, 90), (45, 90), (90, 120)]
    
    all_valid = True
    for sh, el in validation_poses:
        analytical_el = bio.compute_elbow_torque(el, 0.0)
        analytical_sh = bio.compute_shoulder_torque(sh, el, 0.0)
        
        result = sim.validate_against_formula(
            analytical_el, analytical_sh, sh, el, tolerance=0.20
        )
        
        status = "✅" if result['overall_valid'] else "❌"
        print(f"  {status} пл={sh}°, лк={el}°: "
              f"локоть={result['elbow_analytical']:.3f}/{result['elbow_measured']:.3f} Н·м "
              f"(ошибка {result['elbow_error_pct']:.1f}%), "
              f"плечо={result['shoulder_analytical']:.3f}/{result['shoulder_measured']:.3f} Н·м "
              f"(ошибка {result['shoulder_error_pct']:.1f}%)")
        
        if not result['overall_valid']:
            all_valid = False
    
    sim.close()
    
    print(f"\n  Итог валидации: {'✅ ВСЕ ПОЗЫ В ДОПУСКЕ' if all_valid else '❌ ЕСТЬ РАСХОЖДЕНИЯ'}")
    print("=" * 55)