#!/usr/bin/env python3
"""
aux-dp.py

ИНТЕРАКТИВНЫЙ ЗАПУСК Проекта.

Вместо запоминания аргументов командной строки — удобное меню.
Автоматически запускает main_pipeline.py с выбранными параметрами.

Запуск:
    python aux-dp.py

"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any


# ============================================================================
# КОНФИГУРАЦИЯ ПО УМОЛЧАНИЮ
# ============================================================================

DEFAULT_CONFIG = {
    "side": "left",
    "video_path": "vision/video.mp4",
    "use_camera": False,
    "camera_id": 0,
    "external_load_kg": 0.0,
    "use_mujoco": True,
    "show_plots": True,
    "output_dir": "outputs",
}

CONFIG_FILE = "pipeline_config.json"


# ============================================================================
# МЕНЮ
# ============================================================================

class PipelineLauncher:
    """Интерактивный лаунчер пайплайна."""
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Загрузка сохранённой конфигурации или создание по умолчанию"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                # Объединяем с дефолтными (на случай новых полей)
                config = DEFAULT_CONFIG.copy()
                config.update(saved)
                return config
            except Exception:
                pass
        return DEFAULT_CONFIG.copy()
    
    def _save_config(self):
        """Сохранение конфигурации"""
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def _print_header(self):
        """Заголовок"""
        print("\n" + "=" * 55)
        print("  🦾 ПАЙПЛАЙН ПОДБОРА ПРИВОДА ДЛЯ ЭКЗОСКЕЛЕТА")
        print("=" * 55)
        print("  Изображение → Геометрия → Биомеханика → Привод")
        print("=" * 55)
    
    def _print_menu(self):
        """Главное меню"""
        print(f"\n  ТЕКУЩИЕ НАСТРОЙКИ:")
        print(f"  ┌─────────────────────────────────────────┐")
        
        source = f"📷 Камера {self.config['camera_id']}" if self.config['use_camera'] else f"🎬 {self.config['video_path']}"
        print(f"  │ 1. Источник:   {source:<30}│")
        print(f"  │ 2. Рука:       {self.config['side']:<30}│")
        print(f"  │ 3. Нагрузка:   {self.config['external_load_kg']} кг{'':<25}│")
        print(f"  │ 4. MuJoCo:     {'✅ Вкл' if self.config['use_mujoco'] else '❌ Выкл':<30}│")
        print(f"  │ 5. Графики:    {'✅ Вкл' if self.config['show_plots'] else '❌ Выкл':<30}│")
        print(f"  │ 6. Папка:      {self.config['output_dir']:<30}│")
        print(f"  └─────────────────────────────────────────┘")
        print(f"\n  7. 🚀 ЗАПУСТИТЬ ПАЙПЛАЙН")
        print(f"  8. 💾 Сохранить настройки")
        print(f"  9. 🔄 Сбросить настройки")
        print(f"  0. 👋 Выход")
    
    def _edit_source(self):
        """Выбор источника видео"""
        print(f"\n  ИСТОЧНИК ВИДЕО:")
        print(f"  1. 📷 Веб-камера")
        print(f"  2. 🎬 Видеофайл (по умолчанию: {DEFAULT_CONFIG['video_path']})")
        print(f"  3. 📂 Указать свой путь")
        
        choice = input("  Ваш выбор (1-3): ").strip()
        
        if choice == '1':
            self.config['use_camera'] = True
            try:
                cam_id = int(input("  Индекс камеры (0 — встроенная, 1 — внешняя): ").strip() or "0")
                self.config['camera_id'] = cam_id
            except ValueError:
                self.config['camera_id'] = 0
            print(f"  ✅ Выбрана камера {self.config['camera_id']}")
        
        elif choice == '2':
            self.config['use_camera'] = False
            self.config['video_path'] = DEFAULT_CONFIG['video_path']
            if os.path.exists(self.config['video_path']):
                print(f"  ✅ Видео по умолчанию: {self.config['video_path']}")
            else:
                print(f"  ⚠️ Видео не найдено. Запишите видео или укажите другой путь.")
        
        elif choice == '3':
            self.config['use_camera'] = False
            path = input("  Путь к видеофайлу: ").strip()
            if path and os.path.exists(path):
                self.config['video_path'] = path
                print(f"  ✅ Выбрано: {path}")
            else:
                print(f"  ❌ Файл не найден")
    
    def _edit_side(self):
        """Выбор руки"""
        print(f"\n  ОТСЛЕЖИВАЕМАЯ РУКА:")
        print(f"  1. Левая (left)")
        print(f"  2. Правая (right)")
        
        choice = input("  Ваш выбор (1-2): ").strip()
        
        if choice == '1':
            self.config['side'] = 'left'
        elif choice == '2':
            self.config['side'] = 'right'
        else:
            return
        
        print(f"  ✅ Выбрана рука: {self.config['side']}")
    
    def _edit_load(self):
        """Выбор нагрузки"""
        print(f"\n  ВНЕШНЯЯ НАГРУЗКА:")
        print(f"  1. Без груза (0 кг) — только масса руки")
        print(f"  2. Лёгкий груз (2 кг) — бутылка воды")
        print(f"  3. Тяжёлый груз (5 кг) — инструмент")
        print(f"  4. Своё значение")
        
        choice = input("  Ваш выбор (1-4): ").strip()
        
        if choice == '1':
            self.config['external_load_kg'] = 0.0
        elif choice == '2':
            self.config['external_load_kg'] = 2.0
        elif choice == '3':
            self.config['external_load_kg'] = 5.0
        elif choice == '4':
            try:
                load = float(input("  Масса груза (кг): ").strip())
                self.config['external_load_kg'] = max(0.0, load)
            except ValueError:
                return
        
        print(f"  ✅ Нагрузка: {self.config['external_load_kg']} кг")
    
    def _toggle_mujoco(self):
        """Включение/выключение MuJoCo"""
        self.config['use_mujoco'] = not self.config['use_mujoco']
        status = "включён" if self.config['use_mujoco'] else "выключен"
        print(f"  ✅ MuJoCo {status}")
    
    def _toggle_plots(self):
        """Включение/выключение графиков"""
        self.config['show_plots'] = not self.config['show_plots']
        status = "включены" if self.config['show_plots'] else "выключены"
        print(f"  ✅ Графики {status}")
    
    def _edit_output(self):
        """Выбор папки для результатов"""
        print(f"\n  ПАПКА ДЛЯ РЕЗУЛЬТАТОВ:")
        print(f"  Текущая: {self.config['output_dir']}")
        path = input("  Новая папка (Enter — без изменений): ").strip()
        if path:
            self.config['output_dir'] = path
            print(f"  ✅ Папка: {path}")
    
    def _reset_config(self):
        """Сброс на дефолтные настройки"""
        self.config = DEFAULT_CONFIG.copy()
        if os.path.exists(CONFIG_FILE):
            os.remove(CONFIG_FILE)
        print("  ✅ Настройки сброшены")
    
    def _run_pipeline(self):
        """Запуск пайплайна с текущими настройками"""
        self._save_config()
        
        # Формируем команду
        cmd_parts = [sys.executable, "main_pipeline.py"]
        
        cmd_parts.extend(["--side", self.config['side']])
        cmd_parts.extend(["--output", self.config['output_dir']])
        
        if self.config['external_load_kg'] > 0:
            cmd_parts.extend(["--load", str(self.config['external_load_kg'])])
        
        if self.config['use_camera']:
            cmd_parts.extend(["--camera", str(self.config['camera_id'])])
        else:
            cmd_parts.extend(["--video", self.config['video_path']])
        
        if not self.config['use_mujoco']:
            cmd_parts.append("--no-mujoco")
        
        if not self.config['show_plots']:
            cmd_parts.append("--no-plots")
        
        print(f"\n  🚀 ЗАПУСК ПАЙПЛАЙНА...")
        print(f"  {' '.join(cmd_parts)}\n")
        
        # Запускаем
        import subprocess
        result = subprocess.run(cmd_parts)
        
        if result.returncode == 0:
            print(f"\n  ✅ Пайплайн успешно завершён")
        else:
            print(f"\n  ⚠️ Пайплайн завершился с кодом {result.returncode}")
        
        input("\n  Нажмите Enter для продолжения...")
    
    def run(self):
        """Главный цикл меню"""
        while True:
            self._print_header()
            self._print_menu()
            
            choice = input("\n  Ваш выбор (0-9): ").strip()
            
            if choice == '1':
                self._edit_source()
            elif choice == '2':
                self._edit_side()
            elif choice == '3':
                self._edit_load()
            elif choice == '4':
                self._toggle_mujoco()
            elif choice == '5':
                self._toggle_plots()
            elif choice == '6':
                self._edit_output()
            elif choice == '7':
                self._run_pipeline()
            elif choice == '8':
                self._save_config()
                print("  ✅ Настройки сохранены")
            elif choice == '9':
                self._reset_config()
            elif choice == '0':
                print("\n  👋 До свидания!")
                break
            else:
                print("  ❌ Неверный выбор")
            
            if choice != '7':
                input("\n  Нажмите Enter...")
                os.system('cls' if os.name == 'nt' else 'clear')


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    try:
        launcher = PipelineLauncher()
        launcher.run()
    except KeyboardInterrupt:
        print("\n\n  👋 До свидания!")