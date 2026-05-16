#!/usr/bin/env python3
"""
vision/test_vision.py

ТЕСТОВЫЙ СТЕНД ЭТАПА 1: Компьютерное зрение + геометрия.

Проверяет:
    - Стабильность длин звеньев при изменении расстояния до камеры
    - График в реальном времени
    - Сохранение CSV с результатами

Запуск:
    python vision/test_vision.py
    python vision/test_vision.py --camera 0

    

Авторы: Иван Смирнов, Леонид Нагорный

Тг: https://t.me/ivansmittt
Cайт: auxexo.ru
"""

import sys
import os
import time
import csv
from pathlib import Path
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.parameters import AVG_UPPER_ARM, AVG_FOREARM, DEFAULT_SHOULDER_WIDTH
from vision.hand_tracker import HandTracker
from vision.geometry_calculator import GeometryCalculator

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ============================================================================
# ГРАФИК
# ============================================================================

class LivePlotter:
    def __init__(self, window_seconds=30.0):
        if not HAS_PLT:
            self.active = False
            return
        self.active = True
        self.window_seconds = window_seconds
        self.times = deque(maxlen=1000)
        self.upper_vals = deque(maxlen=1000)
        self.forearm_vals = deque(maxlen=1000)
        
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.fig.canvas.manager.set_window_title('Stage 1 — Depth Test')
        self.line_upper, = self.ax.plot([], [], 'r-', lw=2, label='Upper arm')
        self.line_forearm, = self.ax.plot([], [], 'b-', lw=2, label='Forearm')
        self.ax.axhline(y=AVG_UPPER_ARM, color='r', ls=':', alpha=0.3)
        self.ax.axhline(y=AVG_FOREARM, color='b', ls=':', alpha=0.3)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Length (m)')
        self.ax.set_title('DEPTH NORMALIZATION TEST\nLines must be horizontal!')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim(0, 0.8)
        plt.tight_layout()
        self.start_time = time.time()
    
    def add_point(self, upper, forearm):
        if not self.active:
            return
        t = time.time() - self.start_time
        self.times.append(t)
        self.upper_vals.append(upper)
        self.forearm_vals.append(forearm)
    
    def update(self):
        if not self.active or len(self.times) < 2:
            return
        times_arr = np.array(self.times)
        self.line_upper.set_data(times_arr, np.array(self.upper_vals))
        self.line_forearm.set_data(times_arr, np.array(self.forearm_vals))
        self.ax.set_xlim(max(0, max(times_arr) - self.window_seconds), max(times_arr) + 0.5)
        all_vals = np.concatenate([np.array(self.upper_vals), np.array(self.forearm_vals)])
        self.ax.set_ylim(max(0, np.min(all_vals) - 0.05), np.max(all_vals) + 0.05)
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            pass
    
    def close(self):
        if self.active:
            try:
                plt.close(self.fig)
            except Exception:
                pass


# ============================================================================
# ТЕСТ
# ============================================================================

class VisionTest:
    def __init__(self, side="left", camera_id=0, video_path="vision/video2.mp4", use_camera=False):
        self.side = side
        self.use_camera = use_camera
        self.video_path = video_path
        self.camera_id = camera_id
        
        self.tracker = None
        self.geometry_calc = None
        self.plotter = None
        self.cap = None
        
        self.frame_count = 0
        self.start_time = None
        self.log_entries = []
        self._last_log_time = 0
    
    def initialize(self):
        self.tracker = HandTracker(side=self.side)
        self.geometry_calc = GeometryCalculator()
        self.plotter = LivePlotter()
        
        if self.use_camera:
            self.cap = cv2.VideoCapture(self.camera_id)
            source = f"camera {self.camera_id}"
            self._need_flip = True
        else:
            if not os.path.exists(self.video_path):
                print(f"Video not found: {self.video_path}")
                return False
            self.cap = cv2.VideoCapture(self.video_path)
            source = f"video: {self.video_path}"
            self._need_flip = False
        
        if not self.cap.isOpened():
            print(f"Cannot open {source}")
            return False
        
        cv2.namedWindow("Stage 1 Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Stage 1 Test", 960, 720)
        
        self.start_time = time.time()
        
        print(f"\n{'='*55}")
        print(f"  STAGE 1: VISION + GEOMETRY TEST")
        print(f"{'='*55}")
        print(f"  Source: {source}")
        print(f"  Side: {self.side}")
        print(f"  q=quit, r=reset, space=pause\n")
        
        return True
    
    def run(self):
        if not self.initialize():
            return
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video")
                    break
                
                if self._need_flip:
                    frame = cv2.flip(frame, 1)
                self.frame_count += 1
                
                raw_points = self.tracker.process_frame(frame)
                
                if raw_points is not None and self.tracker.are_points_visible(raw_points):
                    geometry = self.geometry_calc.process(raw_points)
                    
                    if geometry is not None:
                        self.plotter.add_point(geometry.upper_arm_m, geometry.forearm_m)
                        
                        if time.time() - self._last_log_time >= 0.5:
                            self._last_log_time = time.time()
                            self.log_entries.append({
                                'frame': self.frame_count,
                                'time_sec': round(time.time() - self.start_time, 1),
                                'upper_arm_m': geometry.upper_arm_m,
                                'forearm_m': geometry.forearm_m,
                                'shoulder_angle': geometry.shoulder_angle,
                                'elbow_angle': geometry.elbow_angle,
                                'brachial_index': geometry.brachial_index,
                                'tracking_conf': geometry.tracking_confidence
                            })
                
                frame = self.tracker.draw_debug(frame)
                frame = self._draw_status(frame)
                self.plotter.update()
                cv2.imshow("Stage 1 Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self._reset()
            elif key == ord(' ') and not self.use_camera:
                paused = not paused
                print("⏸️ PAUSED" if paused else "▶️ RESUMED")
        
        self._finalize()
    
    def _draw_status(self, frame):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Статус
        status = "TRACKING" if self.tracker.is_tracking() else "NO TRACKING"
        color = (0, 255, 0) if self.tracker.is_tracking() else (0, 0, 255)
        cv2.putText(frame, status, (10, h-20), font, 0.6, color, 2)
        
        # Стабильность
        if len(self.log_entries) >= 20:
            upper_vals = [e['upper_arm_m'] for e in self.log_entries]
            forearm_vals = [e['forearm_m'] for e in self.log_entries]
            upper_rel = (np.max(upper_vals)-np.min(upper_vals))/np.mean(upper_vals)
            forearm_rel = (np.max(forearm_vals)-np.min(forearm_vals))/np.mean(forearm_vals)
            worst = max(upper_rel, forearm_rel)
            
            if worst < 0.15:
                status = "PASSED"
                color = (0, 255, 0)
            elif worst < 0.30:
                status = f"WARN ({worst*100:.0f}%)"
                color = (0, 255, 255)
            else:
                status = f"FAIL ({worst*100:.0f}%)"
                color = (0, 0, 255)
            
            cv2.putText(frame, status, (10, h-45), font, 0.6, color, 2)
        
        cv2.putText(frame, f"Samples: {len(self.log_entries)}", (w-150, h-20), font, 0.5, (150,150,150), 1)
        
        return frame
    
    def _reset(self):
        self.log_entries.clear()
        self.geometry_calc.reset()
        print("🔄 Reset")
    
    def _finalize(self):
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n{'='*55}")
        print(f"  TEST RESULTS")
        print(f"{'='*55}")
        print(f"  Frames: {self.frame_count}")
        print(f"  Samples: {len(self.log_entries)}")
        
        if len(self.log_entries) >= 10:
            upper_vals = [e['upper_arm_m'] for e in self.log_entries]
            forearm_vals = [e['forearm_m'] for e in self.log_entries]
            
            print(f"\n  Upper arm:  mean={np.mean(upper_vals):.4f}m, std={np.std(upper_vals):.4f}m")
            print(f"  Forearm:    mean={np.mean(forearm_vals):.4f}m, std={np.std(forearm_vals):.4f}m")
            
            upper_rel = (np.max(upper_vals)-np.min(upper_vals))/np.mean(upper_vals)
            forearm_rel = (np.max(forearm_vals)-np.min(forearm_vals))/np.mean(forearm_vals)
            passed = max(upper_rel, forearm_rel) < 0.15
            
            print(f"\n  {'PASSED' if passed else 'FAILED'}")
            print(f"  Relative range: upper={upper_rel*100:.1f}%, forearm={forearm_rel*100:.1f}%")
        
        if self.log_entries:
            self._save_csv()
        
        self.shutdown()
    
    def _save_csv(self):
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"stage1_test_{timestamp}.csv"
        
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=self.log_entries[0].keys())
            writer.writeheader()
            writer.writerows(self.log_entries)
        
        print(f"  Saved: {filepath}")
    
    def shutdown(self):
        if self.plotter:
            self.plotter.close()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        if self.tracker:
            self.tracker.close()
        print("👋 Done\n")


# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 1: Vision + Geometry Test")
    parser.add_argument("--side", default="left", choices=["left", "right"])
    parser.add_argument("--camera", type=int, default=-1)
    parser.add_argument("--video", type=str, default="vision/video2.mp4")
    
    args = parser.parse_args()
    
    use_camera = (args.camera >= 0)
    
    test = VisionTest(
        side=args.side,
        camera_id=args.camera if use_camera else 0,
        video_path=args.video,
        use_camera=use_camera
    )
    
    try:
        test.run()
    except KeyboardInterrupt:
        print("\nInterrupted")
        test.shutdown()