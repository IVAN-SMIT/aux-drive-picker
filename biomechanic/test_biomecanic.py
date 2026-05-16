#!/usr/bin/env python3
"""
biomechanic/test_biomechanics.py

ТЕСТОВЫЙ СТЕНД ЭТАПОВ 2+3: Кинематика + Биомеханика + MuJoCo-визуализация.

Что делает:
    1. Загружает датасет углов из Этапа 1 (CSV)
    2. Прогоняет через KinematicsCalculator (проверка диапазонов, прямая кинематика)
    3. Прогоняет через BiomechanicsCalculator (аналитические моменты)
    4. Воспроизводит движение в MuJoCo (визуализация + измеренные моменты)
    5. Сравнивает аналитические и измеренные моменты (валидация)
    6. Выдаёт требования к приводам для всех сценариев нагрузки

Сценарии:
    - Без груза (0 кг)
    - Лёгкий груз (2 кг)
    - Тяжёлый груз (5 кг)

Запуск:
    python biomechanic/test_biomechanics.py
    python biomechanic/test_biomechanics.py --csv outputs/stage1_test_xxx.csv
    python biomechanic/test_biomechanics.py --load 5.0 --no-loop


Авторы: Иван Смирнов, Леонид Нагорный

Тг: https://t.me/ivansmittt
Cайт: auxexo.ru
"""

import sys
import os
import csv
import time
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    print("❌ MuJoCo не установлен. Установите: pip install mujoco")
    print("   Будет выполнен только аналитический расчёт.\n")

from kinematic.kinematics import KinematicsCalculator
from biomechanic.biomechanics import BiomechanicsCalculator

if HAS_MUJOCO:
    from mujoco_sim import MuJoCoArmSim


# ============================================================================
# ТЕСТОВЫЙ КЛАСС
# ============================================================================

class BiomechanicsTest:
    """
    Комплексный тест этапов 2 и 3.
    
    Архитектура:
        CSV (углы)
        → KinematicsCalculator (валидация углов, координаты)
        → BiomechanicsCalculator (аналитические моменты)
        → MuJoCoArmSim (визуализация + измеренные моменты)
        → Сравнение + требования к приводам
    """
    
    # Сценарии нагрузки
    LOAD_SCENARIOS = [
        ("no_load", 0.0),
        ("light", 2.0),
        ("heavy", 5.0),
    ]
    
    def __init__(
        self,
        csv_path: str,
        external_load_kg: float = 0.0,
        speed_factor: float = 1.0,
        loop: bool = True,
        skip_visualization: bool = False
    ):
        """
        Args:
            csv_path: путь к CSV из Этапа 1
            external_load_kg: масса груза для визуализации
            speed_factor: скорость воспроизведения
            loop: зациклить воспроизведение
            skip_visualization: пропустить MuJoCo (только расчёты)
        """
        self.csv_path = csv_path
        self.external_load_kg = external_load_kg
        self.speed_factor = speed_factor
        self.loop = loop
        self.skip_visualization = skip_visualization or not HAS_MUJOCO
        
        # Калькуляторы
        self.kinematics = KinematicsCalculator()
        self.biomechanics = BiomechanicsCalculator()
        self.sim: Optional[MuJoCoArmSim] = None
        
        # Данные
        self.dataset: List[Dict] = []
        self.dataset_smooth: List[Dict] = []
        self.results: List[Dict] = []
        
        # Статистика
        self.kinematics_violations = 0
        self.total_frames = 0
    
    # ==================================================================
    # ЗАГРУЗКА ДАТАСЕТА
    # ==================================================================
    
    def load_dataset(self) -> bool:
        """Загрузка CSV с углами из Этапа 1"""
        if not os.path.exists(self.csv_path):
            print(f"❌ Файл не найден: {self.csv_path}")
            return False
        
        with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            self.dataset = list(reader)
        
        if not self.dataset:
            print("❌ Датасет пуст")
            return False
        
        print(f"✅ Загружено {len(self.dataset)} записей")
        
        # Проверяем колонки
        required = ['shoulder_angle', 'elbow_angle']
        for col in required:
            if col not in self.dataset[0]:
                print(f"❌ Отсутствует колонка: {col}")
                return False
        
        # Сглаживание
        self._smooth_dataset(window=5)
        self.total_frames = len(self.dataset_smooth)
        
        return True
    
    def _smooth_dataset(self, window: int = 5):
        """Сглаживание углов для плавного движения"""
        n = len(self.dataset)
        self.dataset_smooth = []
        
        for i in range(n):
            start = max(0, i - window // 2)
            end = min(n, i + window // 2 + 1)
            
            sh_vals = [float(self.dataset[j]['shoulder_angle']) for j in range(start, end)]
            el_vals = [float(self.dataset[j]['elbow_angle']) for j in range(start, end)]
            
            smooth_row = dict(self.dataset[i])
            smooth_row['shoulder_angle'] = str(round(np.mean(sh_vals), 1))
            smooth_row['elbow_angle'] = str(round(np.mean(el_vals), 1))
            
            self.dataset_smooth.append(smooth_row)
        
        # Статистика
        shoulder_raw = [float(r['shoulder_angle']) for r in self.dataset]
        elbow_raw = [float(r['elbow_angle']) for r in self.dataset]
        shoulder_smooth = [float(r['shoulder_angle']) for r in self.dataset_smooth]
        elbow_smooth = [float(r['elbow_angle']) for r in self.dataset_smooth]
        
        print(f"\n  📊 Статистика углов:")
        print(f"  Плечо:  мин={min(shoulder_smooth):.1f}°, макс={max(shoulder_smooth):.1f}°, "
              f"сред={np.mean(shoulder_smooth):.1f}°")
        print(f"  Локоть: мин={min(elbow_smooth):.1f}°, макс={max(elbow_smooth):.1f}°, "
              f"сред={np.mean(elbow_smooth):.1f}°")
    
        # ==================================================================
    # ЭТАП 2: КИНЕМАТИЧЕСКИЙ АНАЛИЗ
    # ==================================================================
    
    def run_kinematics_analysis(self):
        """Прогон датасета через KinematicsCalculator"""
        if not self.dataset_smooth:
            print("  ❌ Датасет не загружен. Сначала вызовите load_dataset()")
            return
        self.total_frames = len(self.dataset_smooth)
        
        print(f"\n{'='*55}")
        print(f"  ЭТАП 2: КИНЕМАТИЧЕСКИЙ АНАЛИЗ")
        print(f"{'='*55}")
        
        workspace = self.kinematics.get_workspace_info()
        print(f"  Рабочая зона: макс. досягаемость = {workspace['max_reach_m']} м")
        print(f"  Длины: плечо={workspace['upper_arm_m']} м, "
              f"предплечье={workspace['forearm_m']} м")
        
        self.kinematics_violations = 0
        
        # Анализируем несколько характерных поз
        n = self.total_frames
        sample_indices = []
        if n >= 1:
            sample_indices.append(0)
        if n >= 2:
            sample_indices.append(n // 4)
        if n >= 3:
            sample_indices.append(n // 2)
        if n >= 4:
            sample_indices.append(3 * n // 4)
        if n >= 5:
            sample_indices.append(n - 1)
        
        # Убираем дубликаты и сортируем
        sample_indices = sorted(set(min(i, n-1) for i in sample_indices))
        
        print(f"\n  Анализ характерных поз (всего кадров: {n}):")
        for idx in sample_indices:
            row = self.dataset_smooth[idx]
            sh = float(row['shoulder_angle'])
            el = float(row['elbow_angle'])
            
            kin = self.kinematics.forward_kinematics(sh, el)
            range_pct = self.kinematics.movement_range_percent(el)
            
            status = "✅" if kin.is_valid else "❌"
            print(f"  {status} Кадр {idx:4d}: пл={sh:+6.1f}°, лк={el:6.1f}° "
                  f"({range_pct:.0f}% диапазона) → "
                  f"запястье=({kin.wrist_pos.x:.3f}, {kin.wrist_pos.y:.3f}) м")
            
            if not kin.is_valid:
                self.kinematics_violations += 1
        
        # Полная проверка всех кадров
        for row in self.dataset_smooth:
            sh = float(row['shoulder_angle'])
            el = float(row['elbow_angle'])
            
            _, sh_ok, _ = self.kinematics.validate_shoulder(sh)
            _, el_ok, _ = self.kinematics.validate_elbow(el)
            
            if not sh_ok or not el_ok:
                self.kinematics_violations += 1
        
        print(f"\n  Всего нарушений диапазонов: {self.kinematics_violations}/{self.total_frames}")
    
    # ==================================================================
    # ЭТАП 3: БИОМЕХАНИЧЕСКИЙ РАСЧЁТ
    # ==================================================================
    
    def run_biomechanics_analysis(self):
        """Аналитические расчёты моментов для всех сценариев"""
        if not self.dataset_smooth:
            print("  ❌ Датасет не загружен. Сначала вызовите load_dataset()")
            return
        self.total_frames = len(self.dataset_smooth)
        
        print(f"\n{'='*55}")
        print(f"  ЭТАП 3: БИОМЕХАНИЧЕСКИЙ РАСЧЁТ")
        print(f"{'='*55}")
        
        # M(θ) для разных нагрузок
        print(f"\n  📈 M(θ) — Момент в локте (аналитическая формула):")
        print(f"  {'Угол':<8} {'0 кг':<10} {'2 кг':<10} {'5 кг':<10}")
        print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
        for angle in [0, 30, 60, 90, 120, 130]:
            values = []
            for _, load in self.LOAD_SCENARIOS:
                t = self.biomechanics.compute_elbow_torque(angle, load)
                values.append(f"{t:.3f}")
            print(f"  {angle:3d}°     {values[0]:>8s}    {values[1]:>8s}    {values[2]:>8s}")
        
        # Максимальные моменты для всех сценариев
        print(f"\n  🔧 ТРЕБОВАНИЯ К ПРИВОДАМ:")
        print(f"  {'Сценарий':<12} {'Плечо M_max':<14} {'Плечо треб.':<14} "
              f"{'Локоть M_max':<14} {'Локоть треб.':<14}")
        print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
        
        self.actuator_requirements = {}
        
        for name, load in self.LOAD_SCENARIOS:
            req = self.biomechanics.get_actuator_requirements(load, safety_factor=1.3)
            self.actuator_requirements[name] = req
            
            print(f"  {name:<12} {req['max_shoulder_raw']:>7.3f} Н·м    "
                  f"{req['M_shoulder_required']:>7.3f} Н·м    "
                  f"{req['max_elbow_raw']:>7.3f} Н·м    "
                  f"{req['M_elbow_required']:>7.3f} Н·м")
        
        # Обработка всего датасета (для статистики)
        if self.total_frames == 0:
            print(f"\n  ⚠️  Датасет пуст — пакетная обработка пропущена")
            return
        
        print(f"\n  Обработка датасета ({self.total_frames} кадров)...")
        
        self.analytical_results = {}
        
        for name, load in self.LOAD_SCENARIOS:
            processed = self.biomechanics.process_dataset(
                self.dataset_smooth, load
            )
            self.analytical_results[name] = processed
            
            sh_torques = [r['shoulder_torque'] for r in processed]
            el_torques = [r['elbow_torque'] for r in processed]
            
            print(f"  {name}: плечо макс={max(sh_torques):.3f}, "
                  f"локоть макс={max(el_torques):.3f} Н·м")
    
    # ==================================================================
    # ЭТАП 3: MUJOCO ВИЗУАЛИЗАЦИЯ + ВАЛИДАЦИЯ
    # ==================================================================
    
    def run_mujoco_simulation(self):
        """Воспроизведение в MuJoCo + сравнение моментов"""
        if self.skip_visualization:
            print(f"\n{'='*55}")
            print(f"  MUJOCO: ПРОПУЩЕН (не установлен или отключён)")
            print(f"{'='*55}")
            return
        
        print(f"\n{'='*55}")
        print(f"  MUJOCO: ВИЗУАЛИЗАЦИЯ + ВАЛИДАЦИЯ")
        print(f"{'='*55}")
        
        # Валидация формул на нескольких позах
        print(f"\n  Валидация аналитических формул против MuJoCo:")
        
        self.sim = MuJoCoArmSim(external_load_kg=self.external_load_kg)
        self.sim.stabilize(50)
        
        validation_poses = [
            (0, 0, "рука висит"),
            (30, 0, "плечо 30°"),
            (60, 0, "плечо 60°"),
            (90, 0, "плечо горизонтально"),
            (45, 60, "полусогнута"),
            (0, 90, "локоть 90°"),
            (90, 120, "макс. нагрузка"),
        ]
        
        all_valid = True
        for sh, el, desc in validation_poses:
            analytical_el = self.biomechanics.compute_elbow_torque(el, self.external_load_kg)
            analytical_sh = self.biomechanics.compute_shoulder_torque(sh, el, self.external_load_kg)
            
            result = self.sim.validate_against_formula(
                analytical_el, analytical_sh, sh, el, tolerance=0.25
            )
            
            status = "✅" if result['overall_valid'] else "❌"
            print(f"  {status} {desc:20s}: "
                  f"локоть формула={result['elbow_analytical']:.3f} / "
                  f"MuJoCo={result['elbow_measured']:.3f} Н·м "
                  f"(ошибка {result['elbow_error_pct']:.1f}%), "
                  f"плечо формула={result['shoulder_analytical']:.3f} / "
                  f"MuJoCo={result['shoulder_measured']:.3f} Н·м "
                  f"(ошибка {result['shoulder_error_pct']:.1f}%)")
            
            if not result['overall_valid']:
                all_valid = False
        
        print(f"\n  Итог валидации: {'✅ ФОРМУЛЫ ПОДТВЕРЖДЕНЫ' if all_valid else '⚠️ ЕСТЬ РАСХОЖДЕНИЯ'}")
        
        # Воспроизведение датасета
        print(f"\n▶️  ВОСПРОИЗВЕДЕНИЕ ДАТАСЕТА В MUJOCO")
        print(f"   Записей: {self.total_frames}")
        print(f"   Нагрузка: {self.external_load_kg} кг")
        print(f"   Скорость: {self.speed_factor}x")
        if self.loop:
            print(f"   Режим: ЗАЦИКЛЕНО")
        print(f"   Нажмите Ctrl+C для остановки\n")
        
        # Возвращаемся в нулевую позу
        self.sim.set_angles(0.0, 0.0)
        self.sim.stabilize(30)
        
        do_loop = True
        lap = 0
        prev_sh = float(self.dataset_smooth[0]['shoulder_angle'])
        prev_el = float(self.dataset_smooth[0]['elbow_angle'])
        
        self.sim.set_angles(prev_sh, prev_el)
        self.sim.stabilize(10)
        
        while do_loop:
            lap += 1
            if lap > 1:
                print(f"\n🔄 Круг {lap}")
            
            start_time = time.time()
            
            for i in range(self.total_frames - 1):
                curr_sh = float(self.dataset_smooth[i + 1]['shoulder_angle'])
                curr_el = float(self.dataset_smooth[i + 1]['elbow_angle'])
                
                # Интерполяция
                joint_state = None
                for step in range(10):
                    alpha = step / 10.0
                    interp_sh = prev_sh + (curr_sh - prev_sh) * alpha
                    interp_el = prev_el + (curr_el - prev_el) * alpha
                    
                    self.sim.set_angles(interp_sh, interp_el)
                    for _ in range(2):
                        joint_state = self.sim.step()
                
                # Сохраняем с аналитическими значениями для сравнения
                analytical = self.biomechanics.compute_all(
                    curr_sh, curr_el, self.external_load_kg
                )
                
                if joint_state:
                    self.results.append({
                        'frame': i,
                        'shoulder_angle': curr_sh,
                        'elbow_angle': curr_el,
                        'analytical_shoulder_torque': analytical.shoulder_torque,
                        'analytical_elbow_torque': analytical.elbow_torque,
                        'mujoco_shoulder_torque': joint_state.shoulder_torque,
                        'mujoco_elbow_torque': joint_state.elbow_torque,
                        'shoulder_error_pct': round(
                            abs(abs(joint_state.shoulder_torque) - abs(analytical.shoulder_torque)) /
                            max(abs(analytical.shoulder_torque), 0.001) * 100, 1
                        ),
                        'elbow_error_pct': round(
                            abs(abs(joint_state.elbow_torque) - abs(analytical.elbow_torque)) /
                            max(abs(analytical.elbow_torque), 0.001) * 100, 1
                        ),
                        'timestamp': round(time.time() - start_time, 3)
                    })
                
                prev_sh = curr_sh
                prev_el = curr_el
                
                if i % 30 == 0 and joint_state:
                    print(f"  Кадр {i:4d}/{self.total_frames}: "
                          f"пл={curr_sh:+6.1f}°, лк={curr_el:6.1f}° | "
                          f"M_локоть: анал={analytical.elbow_torque:.3f}, "
                          f"MuJoCo={joint_state.elbow_torque:.3f} Н·м")
                
                # Контроль скорости
                if self.speed_factor > 0:
                    elapsed = time.time() - start_time
                    target = (i + 1) / (30.0 * self.speed_factor)
                    sleep_time = target - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            
            elapsed_total = time.time() - start_time
            print(f"\n✅ Круг {lap} завершён за {elapsed_total:.1f} сек")
            
            if not self.loop:
                do_loop = False
    
    # ==================================================================
    # ЗАВЕРШЕНИЕ
    # ==================================================================
    
    def _finalize(self):
        """Сводный отчёт"""
        print(f"\n{'='*55}")
        print(f"  СВОДНЫЙ ОТЧЁТ")
        print(f"{'='*55}")
        
        # Кинематика
        print(f"\n  Этап 2 — Кинематика:")
        print(f"  Нарушений диапазонов: {self.kinematics_violations}/{self.total_frames}")
        
        # Биомеханика
        print(f"\n  Этап 3 — Биомеханика:")
        print(f"  Требования к приводам (с запасом 1.3x):")
        for name, _ in self.LOAD_SCENARIOS:
            req = self.actuator_requirements.get(name, {})
            if req:
                print(f"    {name}: плечо ≥ {req.get('M_shoulder_required', '—')} Н·м, "
                      f"локоть ≥ {req.get('M_elbow_required', '—')} Н·м")
        
        # MuJoCo
        if self.results:
            elbow_errors = [r['elbow_error_pct'] for r in self.results]
            shoulder_errors = [r['shoulder_error_pct'] for r in self.results]
            
            print(f"\n  MuJoCo валидация ({len(self.results)} кадров):")
            print(f"  Средняя ошибка: локоть {np.mean(elbow_errors):.1f}%, "
                  f"плечо {np.mean(shoulder_errors):.1f}%")
            print(f"  Макс. ошибка:    локоть {max(elbow_errors):.1f}%, "
                  f"плечо {max(shoulder_errors):.1f}%")
            
            self._save_csv()
        
        self.shutdown()
    
    def _save_csv(self):
        """Сохранение результатов"""
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        load_str = f"_load{int(self.external_load_kg)}kg" if self.external_load_kg > 0 else ""
        filepath = output_dir / f"full_test_{timestamp}{load_str}.csv"
        
        if self.results:
            fieldnames = list(self.results[0].keys())
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
            print(f"\n  📁 Результаты: {filepath}")
    
    def shutdown(self):
        if self.sim:
            self.sim.close()
        print("\n👋 Тест завершён")


# ============================================================================
# ПОИСК CSV
# ============================================================================

def find_latest_csv(directory: str = "outputs", prefix: str = "stage1_test") -> Optional[str]:
    output_dir = Path(directory)
    if not output_dir.exists():
        return None
    csv_files = sorted(output_dir.glob(f"{prefix}_*.csv"), reverse=True)
    return str(csv_files[0]) if csv_files else None


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Комплексный тест: кинематика + биомеханика + MuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python biomechanic/test_biomechanics.py
  python biomechanic/test_biomechanics.py --load 5.0
  python biomechanic/test_biomechanics.py --load 2.0 --no-loop --speed 0.5
  python biomechanic/test_biomechanics.py --calc-only  # только расчёты без MuJoCo
        """
    )
    parser.add_argument("--csv", type=str, default=None,
                       help="Путь к CSV из Этапа 1")
    parser.add_argument("--load", type=float, default=0.0,
                       help="Масса груза для визуализации, кг (default: 0)")
    parser.add_argument("--speed", type=float, default=1.0,
                       help="Скорость воспроизведения (default: 1.0)")
    parser.add_argument("--no-loop", action="store_true",
                       help="Однократное воспроизведение")
    parser.add_argument("--calc-only", action="store_true",
                       help="Только расчёты, без MuJoCo-визуализации")
    
    args = parser.parse_args()
    
    csv_path = args.csv or find_latest_csv()
    if csv_path is None:
        print("❌ Не найден CSV из Этапа 1.")
        print("   Запустите vision/test_vision.py или укажите --csv")
        sys.exit(1)
    
    print(f"ℹ️  Датасет: {csv_path}")
    
    test = BiomechanicsTest(
        csv_path=csv_path,
        external_load_kg=args.load,
        speed_factor=args.speed,
        loop=not args.no_loop,
        skip_visualization=args.calc_only
    )
    
    try:
        if not test.load_dataset():
            sys.exit(1)
        # Этап 2
        test.run_kinematics_analysis()
        
        # Этап 3 — аналитический
        test.run_biomechanics_analysis()
        
        # Этап 3 — MuJoCo
        test.run_mujoco_simulation()
        
        test._finalize()
        
    except KeyboardInterrupt:
        print("\n⚠️  Прервано пользователем")
        test._finalize()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        test.shutdown()