"""
actuators/actuator_database.py

База данных приводов для экзоскелета руки.
Только модели, доступные в РФ (китайские и отечественные производители).

Функционал:
- Хранение базы моторов с характеристиками
- Хранение базы редукторов
- Подбор пары «мотор + редуктор» под требуемый момент сустава


Авторы: Иван Смирнов, Леонид Нагорный

Тг: https://t.me/ivansmittt
Cайт: auxexo.ru
"""

import csv
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import os


# ============================================================================
# ТИПЫ
# ============================================================================

class ActuatorType(Enum):
    BLDC = "BLDC Motor"
    DC_MOTOR = "DC Motor"
    SERVO = "Servo Motor"
    STEPPER = "Stepper Motor"


# ============================================================================
# СТРУКТУРЫ ДАННЫХ
# ============================================================================

@dataclass
class Gearbox:
    """Редуктор"""
    id: int
    name: str
    manufacturer: str
    ratio: float              # передаточное число (например, 50 означает 1:50)
    nominal_torque: float     # номинальный момент на выходе, Н·м
    peak_torque: float        # пиковый момент, Н·м
    efficiency: float         # КПД (0-1)
    weight: float             # кг
    backlash_arcmin: float    # люфт, угловые минуты
    price_rub: Optional[float] = None
    supplier: Optional[str] = None
    suitable_for: str = "both"  # "elbow" / "shoulder" / "both"


@dataclass
class Actuator:
    """Структура данных мотора"""
    id: int
    name: str
    manufacturer: str
    type: ActuatorType
    
    # Основные механические параметры
    nominal_torque: float       # Н·м (номинальный момент мотора)
    peak_torque: float          # Н·м (пиковый момент)
    max_speed: float            # об/мин (скорость без нагрузки)
    weight: float               # кг
    
    # Электрические параметры
    voltage: float              # В
    power: float               # Вт
    efficiency: float          # КПД (0-1)
    
    # Дополнительно
    gear_ratio: Optional[float] = None   # встроенный редуктор (для сервоприводов)
    price_rub: Optional[float] = None
    supplier: Optional[str] = None
    
    # Совместимость
    suitable_for: str = "elbow"          # "elbow" / "shoulder" / "both"
    
    @property
    def power_density(self) -> float:
        """Удельная мощность, Вт/кг"""
        return self.power / self.weight if self.weight > 0 else 0.0
    
    @property
    def torque_density(self) -> float:
        """Удельный момент, Н·м/кг"""
        return self.nominal_torque / self.weight if self.weight > 0 else 0.0


@dataclass
class MotorGearboxPair:
    """Пара мотор + редуктор"""
    motor: Actuator
    gearbox: Gearbox
    output_torque: float       # момент на выходе редуктора, Н·м
    output_speed: float        # скорость на выходе, об/мин
    total_weight: float        # общая масса, кг
    total_efficiency: float    # общий КПД
    total_price: Optional[float] = None  # общая цена
    
    @property
    def name(self) -> str:
        return f"{self.motor.name} + {self.gearbox.name}"
    
    @property
    def nominal_torque(self) -> float:
        """Для совместимости со старым кодом"""
        return self.output_torque
    
    @property
    def manufacturer(self) -> str:
        return f"{self.motor.manufacturer} / {self.gearbox.manufacturer}"


# ============================================================================
# БАЗА ДАННЫХ
# ============================================================================

class ActuatorDatabase:
    """
    База данных приводов и редукторов.
    
    Источники:
    - Чип и Дип, Платан, Электронщик (отечественные)
    - AliExpress, Ozon, Wildberries (китайские с доставкой в РФ)
    - Производители РФ: Электропривод, Машиностроитель
    """
    
    def __init__(self):
        self.actuators: List[Actuator] = []
        self.gearboxes: List[Gearbox] = []
        self._initialize_motors()
        self._initialize_gearboxes()
    
    # ==================================================================
    # МОТОРЫ
    # ==================================================================
    
    def _initialize_motors(self):
        """Наполнение базы моторов"""
        
        # ============ ОТЕЧЕСТВЕННЫЕ ============
        
        self.add(Actuator(
            id=1, name="ДПР-42-Ф1-03", manufacturer="Электропривод (РФ)",
            type=ActuatorType.BLDC,
            nominal_torque=0.25, peak_torque=0.75,
            max_speed=4000, weight=0.180,
            voltage=27, power=40, efficiency=0.78,
            gear_ratio=1/36, price_rub=12500,
            supplier="electroprivod.ru", suitable_for="elbow"
        ))
        
        self.add(Actuator(
            id=2, name="ДПР-52-Н1-05", manufacturer="Электропривод (РФ)",
            type=ActuatorType.BLDC,
            nominal_torque=0.45, peak_torque=1.35,
            max_speed=3500, weight=0.310,
            voltage=27, power=90, efficiency=0.80,
            gear_ratio=1/28, price_rub=18500,
            supplier="electroprivod.ru", suitable_for="both"
        ))
        
        self.add(Actuator(
            id=3, name="ДПР-72-Н1-08", manufacturer="Электропривод (РФ)",
            type=ActuatorType.BLDC,
            nominal_torque=0.90, peak_torque=2.70,
            max_speed=3000, weight=0.550,
            voltage=27, power=150, efficiency=0.82,
            gear_ratio=1/22, price_rub=28000,
            supplier="electroprivod.ru", suitable_for="both"
        ))
        
        self.add(Actuator(
            id=4, name="БДПМ-40-0.04", manufacturer="Машиностроитель (РФ)",
            type=ActuatorType.BLDC,
            nominal_torque=0.12, peak_torque=0.36,
            max_speed=5000, weight=0.090,
            voltage=24, power=35, efficiency=0.76,
            price_rub=9800, supplier="mashstroy.ru", suitable_for="elbow"
        ))
        
        self.add(Actuator(
            id=5, name="БДПМ-60-0.10", manufacturer="Машиностроитель (РФ)",
            type=ActuatorType.BLDC,
            nominal_torque=0.35, peak_torque=1.05,
            max_speed=4000, weight=0.200,
            voltage=24, power=80, efficiency=0.79,
            price_rub=15600, supplier="mashstroy.ru", suitable_for="both"
        ))

        # Шаговые
        self.add(Actuator(
            id=6, name="ШД-57-0.4", manufacturer="Электропривод (РФ)",
            type=ActuatorType.STEPPER,
            nominal_torque=0.40, peak_torque=0.55,
            max_speed=2000, weight=0.480,
            voltage=24, power=60, efficiency=0.72,
            price_rub=7500, supplier="electroprivod.ru", suitable_for="elbow"
        ))
        
        self.add(Actuator(
            id=7, name="ШД-86-1.2", manufacturer="Электропривод (РФ)",
            type=ActuatorType.STEPPER,
            nominal_torque=1.20, peak_torque=1.80,
            max_speed=1500, weight=0.950,
            voltage=48, power=120, efficiency=0.74,
            price_rub=12000, supplier="electroprivod.ru", suitable_for="both"
        ))
        
        # Сервоприводы (РФ-сборка)
        self.add(Actuator(
            id=8, name="ServoDrive SD-20", manufacturer="Чип и Дип (РФ)",
            type=ActuatorType.SERVO,
            nominal_torque=2.00, peak_torque=2.80,
            max_speed=60, weight=0.072,
            voltage=7.4, power=20, efficiency=0.82,
            gear_ratio=1/300, price_rub=4200,
            supplier="chipdip.ru", suitable_for="elbow"
        ))
        
        self.add(Actuator(
            id=9, name="ServoDrive SD-40", manufacturer="Чип и Дип (РФ)",
            type=ActuatorType.SERVO,
            nominal_torque=4.00, peak_torque=5.50,
            max_speed=50, weight=0.095,
            voltage=12, power=40, efficiency=0.80,
            gear_ratio=1/280, price_rub=6800,
            supplier="chipdip.ru", suitable_for="both"
        ))

        # ============ КИТАЙСКИЕ ============
        
        # UGO ROBOT
        self.add(Actuator(
            id=10, name="UGO-430-W350", manufacturer="UGO Robot (КНР)",
            type=ActuatorType.SERVO,
            nominal_torque=3.80, peak_torque=5.20,
            max_speed=48, weight=0.080,
            voltage=12, power=32, efficiency=0.79,
            gear_ratio=1/353, price_rub=8500,
            supplier="AliExpress", suitable_for="elbow"
        ))
        
        self.add(Actuator(
            id=11, name="UGO-540-W270", manufacturer="UGO Robot (КНР)",
            type=ActuatorType.SERVO,
            nominal_torque=9.50, peak_torque=14.00,
            max_speed=32, weight=0.155,
            voltage=12, power=42, efficiency=0.78,
            gear_ratio=1/272, price_rub=13500,
            supplier="AliExpress", suitable_for="both"
        ))

        # RDServo
        self.add(Actuator(
            id=12, name="RDS3235", manufacturer="RDServo (КНР)",
            type=ActuatorType.SERVO,
            nominal_torque=3.50, peak_torque=4.50,
            max_speed=55, weight=0.075,
            voltage=7.4, power=28, efficiency=0.81,
            gear_ratio=1/320, price_rub=3200,
            supplier="AliExpress", suitable_for="elbow"
        ))
        
        self.add(Actuator(
            id=13, name="RDS5160", manufacturer="RDServo (КНР)",
            type=ActuatorType.SERVO,
            nominal_torque=5.50, peak_torque=7.00,
            max_speed=42, weight=0.098,
            voltage=12, power=38, efficiency=0.79,
            gear_ratio=1/275, price_rub=5200,
            supplier="AliExpress", suitable_for="both"
        ))

        # Hiwonder
        self.add(Actuator(
            id=14, name="LX-224", manufacturer="Hiwonder (КНР)",
            type=ActuatorType.SERVO,
            nominal_torque=2.40, peak_torque=3.20,
            max_speed=65, weight=0.060,
            voltage=7.4, power=22, efficiency=0.83,
            gear_ratio=1/310, price_rub=2800,
            supplier="Ozon/WB", suitable_for="elbow"
        ))
        
        self.add(Actuator(
            id=15, name="LX-225", manufacturer="Hiwonder (КНР)",
            type=ActuatorType.SERVO,
            nominal_torque=3.00, peak_torque=4.00,
            max_speed=55, weight=0.068,
            voltage=11.1, power=30, efficiency=0.81,
            gear_ratio=1/290, price_rub=4100,
            supplier="Ozon/WB", suitable_for="elbow"
        ))
        
        self.add(Actuator(
            id=16, name="HT-03", manufacturer="Hiwonder (КНР)",
            type=ActuatorType.SERVO,
            nominal_torque=6.00, peak_torque=8.00,
            max_speed=38, weight=0.112,
            voltage=12, power=45, efficiency=0.78,
            gear_ratio=1/260, price_rub=7800,
            supplier="Ozon/WB", suitable_for="both"
        ))

        # DSpower BLDC
        self.add(Actuator(
            id=17, name="DS-S010A", manufacturer="DSpower (КНР)",
            type=ActuatorType.BLDC,
            nominal_torque=0.18, peak_torque=0.54,
            max_speed=6000, weight=0.085,
            voltage=24, power=45, efficiency=0.82,
            price_rub=4200, supplier="AliExpress", suitable_for="elbow"
        ))
        
        self.add(Actuator(
            id=18, name="DS-S020B", manufacturer="DSpower (КНР)",
            type=ActuatorType.BLDC,
            nominal_torque=0.42, peak_torque=1.26,
            max_speed=4500, weight=0.155,
            voltage=24, power=90, efficiency=0.84,
            price_rub=7800, supplier="AliExpress", suitable_for="both"
        ))

        # JGB37
        self.add(Actuator(
            id=19, name="JGB37-550", manufacturer="JGB37 (КНР)",
            type=ActuatorType.DC_MOTOR,
            nominal_torque=0.55, peak_torque=0.80,
            max_speed=300, weight=0.095,
            voltage=12, power=18, efficiency=0.70,
            gear_ratio=1/100, price_rub=850,
            supplier="Чип и Дип", suitable_for="elbow"
        ))
        
        self.add(Actuator(
            id=20, name="JGB37-1000", manufacturer="JGB37 (КНР)",
            type=ActuatorType.DC_MOTOR,
            nominal_torque=0.90, peak_torque=1.30,
            max_speed=200, weight=0.145,
            voltage=12, power=25, efficiency=0.68,
            gear_ratio=1/150, price_rub=1200,
            supplier="Чип и Дип", suitable_for="both"
        ))

        # Feetech
        self.add(Actuator(
            id=21, name="FT5335M", manufacturer="Feetech (КНР)",
            type=ActuatorType.SERVO,
            nominal_torque=3.80, peak_torque=5.00,
            max_speed=50, weight=0.074,
            voltage=7.4, power=30, efficiency=0.80,
            gear_ratio=1/330, price_rub=3500,
            supplier="AliExpress", suitable_for="elbow"
        ))
        
        self.add(Actuator(
            id=22, name="FT5835M", manufacturer="Feetech (КНР)",
            type=ActuatorType.SERVO,
            nominal_torque=5.50, peak_torque=7.50,
            max_speed=40, weight=0.102,
            voltage=12, power=42, efficiency=0.78,
            gear_ratio=1/280, price_rub=5600,
            supplier="AliExpress", suitable_for="both"
        ))
        
        # Miuzei
        self.add(Actuator(
            id=23, name="MZ-996R", manufacturer="Miuzei (КНР)",
            type=ActuatorType.SERVO,
            nominal_torque=1.20, peak_torque=1.50,
            max_speed=60, weight=0.055,
            voltage=6.0, power=12, efficiency=0.75,
            gear_ratio=1/350, price_rub=450,
            supplier="Ozon/WB", suitable_for="elbow"
        ))
        
        self.add(Actuator(
            id=24, name="MZ-1501", manufacturer="Miuzei (КНР)",
            type=ActuatorType.SERVO,
            nominal_torque=1.80, peak_torque=2.20,
            max_speed=55, weight=0.062,
            voltage=7.4, power=18, efficiency=0.76,
            gear_ratio=1/320, price_rub=850,
            supplier="Ozon/WB", suitable_for="elbow"
        ))
        
        self.add(Actuator(
            id=25, name="MG996R", manufacturer="Wavgat (КНР)",
            type=ActuatorType.SERVO,
            nominal_torque=1.00, peak_torque=1.20,
            max_speed=55, weight=0.055,
            voltage=6.0, power=10, efficiency=0.73,
            gear_ratio=1/360, price_rub=380,
            supplier="Ozon/WB", suitable_for="elbow"
        ))
    
    # ==================================================================
    # РЕДУКТОРЫ
    # ==================================================================
    
    def _initialize_gearboxes(self):
        """Наполнение базы редукторов"""
        
        # Планетарные редукторы (Электропривод, РФ)
        self.add_gearbox(Gearbox(
            id=101, name="ПР-42-1:10", manufacturer="Электропривод (РФ)",
            ratio=10, nominal_torque=5.0, peak_torque=8.0,
            efficiency=0.90, weight=0.150, backlash_arcmin=30,
            price_rub=4500, supplier="electroprivod.ru", suitable_for="elbow"
        ))
        
        self.add_gearbox(Gearbox(
            id=102, name="ПР-52-1:25", manufacturer="Электропривод (РФ)",
            ratio=25, nominal_torque=12.0, peak_torque=18.0,
            efficiency=0.88, weight=0.280, backlash_arcmin=25,
            price_rub=6800, supplier="electroprivod.ru", suitable_for="both"
        ))
        
        self.add_gearbox(Gearbox(
            id=103, name="ПР-72-1:50", manufacturer="Электропривод (РФ)",
            ratio=50, nominal_torque=25.0, peak_torque=38.0,
            efficiency=0.85, weight=0.520, backlash_arcmin=20,
            price_rub=12000, supplier="electroprivod.ru", suitable_for="both"
        ))
        
        self.add_gearbox(Gearbox(
            id=104, name="ПР-72-1:100", manufacturer="Электропривод (РФ)",
            ratio=100, nominal_torque=40.0, peak_torque=60.0,
            efficiency=0.82, weight=0.650, backlash_arcmin=15,
            price_rub=18000, supplier="electroprivod.ru", suitable_for="shoulder"
        ))
        
        # Волновые редукторы (китайские клоны Harmonic Drive)
        self.add_gearbox(Gearbox(
            id=201, name="HD-14-1:50", manufacturer="Harmonic Drive (КНР)",
            ratio=50, nominal_torque=7.0, peak_torque=15.0,
            efficiency=0.85, weight=0.120, backlash_arcmin=1,
            price_rub=8500, supplier="AliExpress", suitable_for="elbow"
        ))
        
        self.add_gearbox(Gearbox(
            id=202, name="HD-17-1:50", manufacturer="Harmonic Drive (КНР)",
            ratio=50, nominal_torque=20.0, peak_torque=40.0,
            efficiency=0.85, weight=0.220, backlash_arcmin=1,
            price_rub=15000, supplier="AliExpress", suitable_for="both"
        ))
        
        self.add_gearbox(Gearbox(
            id=203, name="HD-20-1:50", manufacturer="Harmonic Drive (КНР)",
            ratio=50, nominal_torque=34.0, peak_torque=70.0,
            efficiency=0.85, weight=0.380, backlash_arcmin=1,
            price_rub=25000, supplier="AliExpress", suitable_for="shoulder"
        ))
        
        self.add_gearbox(Gearbox(
            id=204, name="HD-17-1:100", manufacturer="Harmonic Drive (КНР)",
            ratio=100, nominal_torque=28.0, peak_torque=56.0,
            efficiency=0.80, weight=0.250, backlash_arcmin=1,
            price_rub=18000, supplier="AliExpress", suitable_for="both"
        ))
        
        # Цилиндрические редукторы (бюджетные)
        self.add_gearbox(Gearbox(
            id=301, name="ЦР-32-1:30", manufacturer="Редуктор-СПб (РФ)",
            ratio=30, nominal_torque=8.0, peak_torque=12.0,
            efficiency=0.82, weight=0.350, backlash_arcmin=40,
            price_rub=3200, supplier="reductor-spb.ru", suitable_for="elbow"
        ))
        
        self.add_gearbox(Gearbox(
            id=302, name="ЦР-50-1:60", manufacturer="Редуктор-СПб (РФ)",
            ratio=60, nominal_torque=18.0, peak_torque=27.0,
            efficiency=0.80, weight=0.620, backlash_arcmin=35,
            price_rub=5500, supplier="reductor-spb.ru", suitable_for="both"
        ))
    
    # ==================================================================
    # МЕТОДЫ ДОБАВЛЕНИЯ
    # ==================================================================
    
    def add(self, actuator: Actuator) -> None:
        """Добавить мотор в базу"""
        self.actuators.append(actuator)
    
    def add_gearbox(self, gearbox: Gearbox) -> None:
        """Добавить редуктор в базу"""
        self.gearboxes.append(gearbox)
    
    # ==================================================================
    # МЕТОДЫ ПОИСКА (СТАРЫЕ — ДЛЯ СОВМЕСТИМОСТИ)
    # ==================================================================
    
    def get_all(self) -> List[Actuator]:
        return self.actuators
    
    def get_by_id(self, actuator_id: int) -> Optional[Actuator]:
        for a in self.actuators:
            if a.id == actuator_id:
                return a
        return None
    
    def filter_by_joint(self, joint: str) -> List[Actuator]:
        return [a for a in self.actuators 
                if a.suitable_for == joint or a.suitable_for == "both"]
    
    def filter_by_torque(self, min_torque: float) -> List[Actuator]:
        return [a for a in self.actuators if a.nominal_torque >= min_torque]
    
    def filter_by_weight(self, max_weight: float) -> List[Actuator]:
        return [a for a in self.actuators if a.weight <= max_weight]
    
    def filter_by_manufacturer(self, manufacturer: str) -> List[Actuator]:
        return [a for a in self.actuators 
                if manufacturer.lower() in a.manufacturer.lower()]
    
    def get_manufacturers(self) -> List[str]:
        return sorted(list(set(a.manufacturer for a in self.actuators)))
    
    def get_suppliers(self) -> List[str]:
        suppliers = set()
        for a in self.actuators:
            if a.supplier:
                suppliers.add(a.supplier)
        return sorted(list(suppliers))
    
    # ==================================================================
    # ПОДБОР ПАРЫ МОТОР + РЕДУКТОР
    # ==================================================================
    
    def find_optimal_pair(
        self,
        required_torque: float,
        joint: str = "elbow",
        max_weight: Optional[float] = None,
        max_price: Optional[float] = None,
        prefer_russian: bool = True,
        safety_factor: float = 1.3
    ) -> Optional[MotorGearboxPair]:
        """
        Поиск оптимальной пары «мотор + редуктор».
        
        Алгоритм:
        1. Перебираем все редукторы, подходящие по суставу и моменту
        2. Для каждого редуктора считаем требуемый момент мотора
        3. Ищем моторы с достаточным моментом
        4. Выбираем лучшую пару по массе и цене
        
        Args:
            required_torque: требуемый момент на суставе, Н·м
            joint: "elbow" или "shoulder"
            max_weight: ограничение по общей массе, кг
            max_price: ограничение по общей цене, руб
            prefer_russian: приоритет РФ-производителям
            safety_factor: запас по моменту
        
        Returns:
            Оптимальная пара или None
        """
        target_torque = required_torque * safety_factor
        
        # Фильтруем редукторы
        suitable_gearboxes = [
            g for g in self.gearboxes
            if g.nominal_torque >= target_torque
            and (g.suitable_for == joint or g.suitable_for == "both")
        ]
        
        if not suitable_gearboxes:
            return None
        
        best_pair = None
        best_score = float('inf')
        
        for gearbox in suitable_gearboxes:
            # Требуемый момент мотора (с учётом КПД редуктора)
            motor_torque_required = target_torque / gearbox.ratio / gearbox.efficiency
            
            # Ищем подходящие моторы
            suitable_motors = [
                m for m in self.actuators
                if m.nominal_torque >= motor_torque_required
                and m.max_speed / gearbox.ratio >= 10  # мин. 10 об/мин на выходе
            ]
            
            for motor in suitable_motors:
                # Выходные параметры пары
                output_torque = motor.nominal_torque * gearbox.ratio * gearbox.efficiency
                output_speed = motor.max_speed / gearbox.ratio
                total_weight = motor.weight + gearbox.weight
                total_efficiency = motor.efficiency * gearbox.efficiency
                
                total_price = None
                if motor.price_rub and gearbox.price_rub:
                    total_price = motor.price_rub + gearbox.price_rub
                
                # Проверка ограничений
                if max_weight and total_weight > max_weight:
                    continue
                if max_price and total_price and total_price > max_price:
                    continue
                
                # Скоринг
                weight_score = total_weight / 1.5  # нормировка на ~1.5 кг
                price_score = (total_price / 50000) if total_price else 0.5
                torque_margin = (output_torque - target_torque) / target_torque
                efficiency_penalty = 1.0 - total_efficiency
                
                russian_bonus = -0.1 if prefer_russian and (
                    "РФ" in motor.manufacturer or "РФ" in gearbox.manufacturer
                ) else 0.0
                
                score = (
                    weight_score * 0.35 +
                    price_score * 0.25 +
                    torque_margin * 0.10 +
                    efficiency_penalty * 0.20 +
                    russian_bonus
                )
                
                if score < best_score:
                    best_score = score
                    best_pair = MotorGearboxPair(
                        motor=motor,
                        gearbox=gearbox,
                        output_torque=round(output_torque, 3),
                        output_speed=round(output_speed, 0),
                        total_weight=round(total_weight, 3),
                        total_efficiency=round(total_efficiency, 3),
                        total_price=round(total_price) if total_price else None
                    )
        
        return best_pair
    
    def find_optimal(
        self,
        required_torque: float,
        joint: str = "elbow",
        max_weight: Optional[float] = None,
        max_price: Optional[float] = None,
        prefer_russian: bool = True,
        safety_factor: float = 1.3
    ) -> Optional[Actuator]:
        """
        Поиск оптимального мотора (старый метод).
        Сначала пробует найти голый мотор, затем пару с редуктором.
        Возвращает Actuator для обратной совместимости.
        """
        # Пробуем пару с редуктором
        pair = self.find_optimal_pair(
            required_torque, joint, max_weight, max_price,
            prefer_russian, safety_factor
        )
        
        if pair:
            # Создаём виртуальный Actuator из пары
            return Actuator(
                id=900 + hash(pair.name) % 100,
                name=pair.name,
                manufacturer=f"{pair.motor.manufacturer} + {pair.gearbox.manufacturer}",
                type=pair.motor.type,
                nominal_torque=pair.output_torque,
                peak_torque=pair.output_torque * 1.5,
                max_speed=pair.output_speed,
                weight=pair.total_weight,
                voltage=pair.motor.voltage,
                power=pair.motor.power,
                efficiency=pair.total_efficiency,
                price_rub=pair.total_price,
                suitable_for=joint
            )
        
        # Если пара не найдена — старый поиск
        target_torque = required_torque * safety_factor
        candidates = [a for a in self.actuators 
                      if a.nominal_torque >= target_torque
                      and (a.suitable_for == joint or a.suitable_for == "both")]
        
        if max_weight is not None:
            candidates = [a for a in candidates if a.weight <= max_weight]
        if max_price is not None:
            candidates = [a for a in candidates 
                          if a.price_rub is not None and a.price_rub <= max_price]
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda a: a.weight)
        return candidates[0]
    
    def get_top_candidates(
        self,
        required_torque: float,
        joint: str = "elbow",
        top_n: int = 5,
        **kwargs
    ) -> List[Actuator]:
        """Топ-N кандидатов (старый метод + пары)"""
        candidates = self.filter_by_torque(required_torque * 1.3)
        candidates = [a for a in candidates 
                      if a.suitable_for == joint or a.suitable_for == "both"]
        candidates.sort(key=lambda a: a.torque_density, reverse=True)
        return candidates[:top_n]
    
    # ==================================================================
    # СТАТИСТИКА И ЭКСПОРТ
    # ==================================================================
    
    def get_statistics(self) -> Dict:
        """Статистика по базе"""
        torques = [a.nominal_torque for a in self.actuators]
        weights = [a.weight for a in self.actuators]
        prices = [a.price_rub for a in self.actuators if a.price_rub]
        efficiencies = [a.efficiency for a in self.actuators]
        
        russian_count = len([a for a in self.actuators if "РФ" in a.manufacturer])
        chinese_count = len([a for a in self.actuators if "КНР" in a.manufacturer])
        
        return {
            "total_models": len(self.actuators),
            "total_gearboxes": len(self.gearboxes),
            "russian_models": russian_count,
            "chinese_models": chinese_count,
            "manufacturers": self.get_manufacturers(),
            "suppliers": self.get_suppliers(),
            "torque_stats": {
                "min": round(min(torques), 3),
                "max": round(max(torques), 3),
                "mean": round(np.mean(torques), 3),
                "median": round(np.median(torques), 3)
            },
            "weight_stats": {
                "min": round(min(weights), 3),
                "max": round(max(weights), 3),
                "mean": round(np.mean(weights), 3)
            },
            "price_stats": {
                "min": round(min(prices), 2) if prices else None,
                "max": round(max(prices), 2) if prices else None,
                "mean": round(np.mean(prices), 2) if prices else None
            },
            "efficiency_stats": {
                "min": round(min(efficiencies), 3),
                "max": round(max(efficiencies), 3),
                "mean": round(np.mean(efficiencies), 3)
            },
            "suitable_for": {
                "elbow": len(self.filter_by_joint("elbow")),
                "shoulder": len(self.filter_by_joint("shoulder")),
                "both": len([a for a in self.actuators if a.suitable_for == "both"])
            }
        }
    
    def export_to_csv(self, filepath: str = "actuators_database.csv") -> str:
        """Экспорт базы в CSV"""
        fieldnames = [
            "ID", "Название", "Производитель", "Тип",
            "Момент ном. (Н·м)", "Момент пик. (Н·м)",
            "Скорость (об/мин)", "Вес (кг)",
            "Напряжение (В)", "Мощность (Вт)", "КПД",
            "Передаточное число", "Цена (руб)",
            "Поставщик", "Сустав"
        ]
        
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            
            for a in self.actuators:
                writer.writerow({
                    "ID": a.id,
                    "Название": a.name,
                    "Производитель": a.manufacturer,
                    "Тип": a.type.value,
                    "Момент ном. (Н·м)": a.nominal_torque,
                    "Момент пик. (Н·м)": a.peak_torque,
                    "Скорость (об/мин)": a.max_speed,
                    "Вес (кг)": a.weight,
                    "Напряжение (В)": a.voltage,
                    "Мощность (Вт)": a.power,
                    "КПД": a.efficiency,
                    "Передаточное число": a.gear_ratio if a.gear_ratio else "—",
                    "Цена (руб)": a.price_rub if a.price_rub else "—",
                    "Поставщик": a.supplier if a.supplier else "—",
                    "Сустав": a.suitable_for
                })
        
        return os.path.abspath(filepath)
    
    def print_summary(self) -> None:
        """Вывод сводки"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 65)
        print("  БАЗА ПРИВОДОВ И РЕДУКТОРОВ (РФ + КНР)")
        print("=" * 65)
        print(f"  Моторов: {stats['total_models']} ({stats['russian_models']} РФ, {stats['chinese_models']} КНР)")
        print(f"  Редукторов: {stats['total_gearboxes']}")
        print(f"  Для локтя: {stats['suitable_for']['elbow']}")
        print(f"  Для плеча: {stats['suitable_for']['shoulder']}")
        print(f"  Универсальных: {stats['suitable_for']['both']}")
        print("=" * 65)
    
    def print_actuator_card(self, actuator: Actuator) -> None:
        """Вывод карточки привода"""
        print("\n" + "─" * 55)
        print(f"  🦾 {actuator.name}")
        print("─" * 55)
        print(f"  Производитель: {actuator.manufacturer}")
        print(f"  Тип: {actuator.type.value}")
        print(f"  Сустав: {actuator.suitable_for}")
        print()
        print(f"  Момент номинальный: {actuator.nominal_torque:.2f} Н·м")
        print(f"  Момент пиковый:     {actuator.peak_torque:.2f} Н·м")
        print(f"  Скорость макс.:     {actuator.max_speed:.0f} об/мин")
        print(f"  Вес:                {actuator.weight:.3f} кг")
        if actuator.price_rub:
            print(f"  Цена:               {actuator.price_rub:,.0f} ₽")
        if actuator.supplier:
            print(f"  Поставщик:          {actuator.supplier}")
        print("─" * 55)


# ============================================================================
# ТЕСТ
# ============================================================================

if __name__ == "__main__":
    db = ActuatorDatabase()
    db.print_summary()
    
    print("\n" + "🔍 ПРИМЕР 1: Подбор пары для локтя (M=4.2 Н·м)")
    pair = db.find_optimal_pair(required_torque=4.2, joint="elbow", prefer_russian=True)
    if pair:
        print(f"  ✅ {pair.name}")
        print(f"     Момент на выходе: {pair.output_torque:.2f} Н·м")
        print(f"     Скорость: {pair.output_speed:.0f} об/мин")
        print(f"     Масса: {pair.total_weight:.3f} кг")
        print(f"     Цена: {pair.total_price:,.0f} ₽" if pair.total_price else "")
    
    print("\n" + "🔍 ПРИМЕР 2: Подбор пары для плеча (M=9.8 Н·м)")
    pair = db.find_optimal_pair(required_torque=9.8, joint="shoulder", prefer_russian=True)
    if pair:
        print(f"  ✅ {pair.name}")
        print(f"     Момент на выходе: {pair.output_torque:.2f} Н·м")
        print(f"     Масса: {pair.total_weight:.3f} кг")
    
    print("\n" + "🔍 ПРИМЕР 3: Подбор пары для плеча с грузом 5 кг (M=35.4 Н·м)")
    pair = db.find_optimal_pair(required_torque=35.4, joint="shoulder", prefer_russian=True)
    if pair:
        print(f"  ✅ {pair.name}")
        print(f"     Момент на выходе: {pair.output_torque:.2f} Н·м")
    else:
        print(f"  ❌ Подходящая пара не найдена")
    
    # Экспорт
    filepath = db.export_to_csv("actuators_database.csv")
    print(f"\n📁 База экспортирована: {filepath}")