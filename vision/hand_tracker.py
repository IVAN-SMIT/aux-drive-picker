"""
vision/hand_tracker.py

Трекер руки через MediaPipe Pose с использованием World Landmarks.

ОТВЕТСТВЕННОСТЬ МОДУЛЯ:
- Захват кадра и прогон через MediaPipe Pose
- Извлечение 3D-координат (World Landmarks) в метрах
- Извлечение видимости каждой точки
- Отрисовка скелета для визуальной отладки

КЛЮЧЕВОЕ ОТЛИЧИЕ ОТ ПРЕДЫДУЩЕЙ ВЕРСИИ:
Используются pose_world_landmarks — реальные 3D-координаты в метрах
относительно центра таза. MediaPipe сам вычисляет глубину,
нам не нужна нормализация через ширину плеч.

ВСЯ МАТЕМАТИКА (углы, длины, проценты) — в geometry_calculator.py


Авторы: Иван Смирнов, Леонид Нагорный

Тг: https://t.me/ivansmittt
Cайт: auxexo.ru
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
import mediapipe as mp


class HandTracker:
    """
    Трекер руки через MediaPipe Pose.
    
    Использует World Landmarks — 3D-координаты в метрах.
    Никакой нормализации глубины вручную не требуется.
    """
    
    # Индексы ключевых точек MediaPipe Pose
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    
    MIN_VISIBILITY = 0.5
    
    def __init__(
        self,
        side: str = "left",
        min_detection_conf: float = 0.6,
        min_tracking_conf: float = 0.5
    ):
        """
        Args:
            side: какую руку отслеживать ("left" / "right")
            min_detection_conf: порог обнаружения
            min_tracking_conf: порог трекинга
        """
        self.side = side
        
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,                    # 2 = максимальная точность для 3D
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )
        
        self.tracking = False
        self.landmarks = None
        self.world_landmarks = None
        self._raw_points: Optional[Dict] = None
        self._frame_shape: Optional[Tuple[int, int]] = None
    
    # ==================================================================
    # ОБРАБОТКА КАДРА
    # ==================================================================
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Обработка одного кадра.
        
        Returns:
            Словарь:
            {
                'shoulder': np.array([x, y, z]),  # 3D world-координаты (метры)
                'elbow':    np.array([x, y, z]),
                'wrist':    np.array([x, y, z]),
                'other_shoulder': np.array([x, y, z]),
                
                # 2D-координаты для отрисовки (пиксели)
                'shoulder_2d': np.array([x, y]),
                'elbow_2d':    np.array([x, y]),
                'wrist_2d':    np.array([x, y]),
                'other_shoulder_2d': np.array([x, y]),
                
                # Видимость
                'vis_shoulder': float,
                'vis_elbow': float,
                'vis_wrist': float,
                'vis_other_shoulder': float,
                
                # Размер кадра
                'frame_width': int,
                'frame_height': int,
            }
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.pose.process(rgb)
        rgb.flags.writeable = True
        
        h, w = frame.shape[:2]
        self._frame_shape = (h, w)
        
        # Проверяем, есть ли оба типа landmarks
        if not result.pose_landmarks or not result.pose_world_landmarks:
            self.tracking = False
            self.landmarks = None
            self.world_landmarks = None
            self._raw_points = None
            return None
        
        self.tracking = True
        self.landmarks = result.pose_landmarks
        self.world_landmarks = result.pose_world_landmarks
        
        points = self._extract_points(
            result.pose_landmarks,
            result.pose_world_landmarks,
            w, h
        )
        self._raw_points = points
        
        return points
    
    def is_tracking(self) -> bool:
        return self.tracking
    
    def are_points_visible(self, points: Optional[Dict] = None) -> bool:
        if points is None:
            points = self._raw_points
        if points is None:
            return False
        
        return (
            points['vis_shoulder'] >= self.MIN_VISIBILITY and
            points['vis_elbow'] >= self.MIN_VISIBILITY and
            points['vis_wrist'] >= self.MIN_VISIBILITY and
            points['vis_other_shoulder'] >= self.MIN_VISIBILITY
        )
    
    @property
    def frame_shape(self) -> Optional[Tuple[int, int]]:
        return self._frame_shape
    
    # ==================================================================
    # ИЗВЛЕЧЕНИЕ ТОЧЕК
    # ==================================================================
    
    def _extract_points(
        self,
        landmarks_2d,
        landmarks_3d,
        w: int, h: int
    ) -> Dict:
        """
        Извлечение 2D и 3D координат ключевых точек.
        
        World Landmarks (3D):
        - Начало координат: центр таза (между бёдрами)
        - X: вправо
        - Y: вверх
        - Z: вперёд (от камеры)
        - Единицы: метры
        """
        lm_2d = landmarks_2d.landmark
        lm_3d = landmarks_3d.landmark
        
        # Индексы для выбранной руки
        if self.side == "left":
            s_idx = self.LEFT_SHOULDER
            e_idx = self.LEFT_ELBOW
            wr_idx = self.LEFT_WRIST
            os_idx = self.RIGHT_SHOULDER
        else:
            s_idx = self.RIGHT_SHOULDER
            e_idx = self.RIGHT_ELBOW
            wr_idx = self.RIGHT_WRIST
            os_idx = self.LEFT_SHOULDER
        
        return {
            # 3D координаты (world, метры)
            'shoulder': np.array([
                lm_3d[s_idx].x,
                lm_3d[s_idx].y,
                lm_3d[s_idx].z
            ]),
            'elbow': np.array([
                lm_3d[e_idx].x,
                lm_3d[e_idx].y,
                lm_3d[e_idx].z
            ]),
            'wrist': np.array([
                lm_3d[wr_idx].x,
                lm_3d[wr_idx].y,
                lm_3d[wr_idx].z
            ]),
            'other_shoulder': np.array([
                lm_3d[os_idx].x,
                lm_3d[os_idx].y,
                lm_3d[os_idx].z
            ]),
            
            # 2D координаты (пиксели, для отрисовки)
            'shoulder_2d': np.array([
                lm_2d[s_idx].x * w,
                lm_2d[s_idx].y * h
            ]),
            'elbow_2d': np.array([
                lm_2d[e_idx].x * w,
                lm_2d[e_idx].y * h
            ]),
            'wrist_2d': np.array([
                lm_2d[wr_idx].x * w,
                lm_2d[wr_idx].y * h
            ]),
            'other_shoulder_2d': np.array([
                lm_2d[os_idx].x * w,
                lm_2d[os_idx].y * h
            ]),
            
            # Видимость
            'vis_shoulder': self._get_visibility(lm_2d[s_idx]),
            'vis_elbow': self._get_visibility(lm_2d[e_idx]),
            'vis_wrist': self._get_visibility(lm_2d[wr_idx]),
            'vis_other_shoulder': self._get_visibility(lm_2d[os_idx]),
            
            # Размер кадра
            'frame_width': w,
            'frame_height': h,
        }
    
    @staticmethod
    def _get_visibility(landmark) -> float:
        if landmark.HasField('visibility'):
            return float(landmark.visibility)
        return 1.0
    
    # ==================================================================
    # ВИЗУАЛИЗАЦИЯ
    # ==================================================================
    
    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        """Отрисовка скелета на кадре"""
        if self.landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                self.landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=2
                ),
                mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 0, 255), thickness=2
                )
            )
        
        # Жирные линии для отслеживаемой руки
        if self._raw_points and self.landmarks:
            h, w = frame.shape[:2]
            lm = self.landmarks.landmark
            
            if self.side == "left":
                s_idx, e_idx, wr_idx = 11, 13, 15
            else:
                s_idx, e_idx, wr_idx = 12, 14, 16
            
            pt_s = (int(lm[s_idx].x * w), int(lm[s_idx].y * h))
            pt_e = (int(lm[e_idx].x * w), int(lm[e_idx].y * h))
            pt_w = (int(lm[wr_idx].x * w), int(lm[wr_idx].y * h))
            
            cv2.line(frame, pt_s, pt_e, (0, 0, 255), 4)
            cv2.line(frame, pt_e, pt_w, (255, 0, 0), 4)
            cv2.circle(frame, pt_s, 8, (0, 255, 0), -1)
            cv2.circle(frame, pt_e, 7, (0, 255, 255), -1)
            cv2.circle(frame, pt_w, 6, (255, 0, 255), -1)
            
            # 3D-координаты для отладки
            if self._raw_points:
                y = 30
                font = cv2.FONT_HERSHEY_SIMPLEX
                pts = self._raw_points
                for label, key in [
                    ("Sh", "shoulder"), ("El", "elbow"), ("Wr", "wrist")
                ]:
                    p = pts[key]
                    cv2.putText(frame, 
                        f"{label}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})m",
                        (10, y), font, 0.4, (200, 200, 200), 1)
                    y += 18
        
        return frame
    
    def close(self):
        self.pose.close()
        self._raw_points = None
        self.landmarks = None
        self.world_landmarks = None