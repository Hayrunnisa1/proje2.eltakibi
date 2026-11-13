"""
Kamerayla El Takibi ve Parmak Sayısıyla Kontrol Edilen Trafik Oyunu
-------------------------------------------------------------------
Geliştirici: Antigravity AI
Kullanılan Teknolojiler: Python, OpenCV, MediaPipe Hands

Kontroller:
- 1 Parmak: 1. Şerit (Sol)
- 2 Parmak: 2. Şerit (Orta)
- 3 Parmak: 3. Şerit (Sağ)
- 0, 4 veya 5 Parmak: Mevcut şeridi koru
- 'R' Tuşu: Oyunu yeniden başlat
- 'Q' Tuşu: Programdan çıkış
"""

import os
import sys
import tempfile
import urllib.request
import random
import time
import numpy as np

# Matplotlib veya MediaPipe geçici dizin uyarısını engelle
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import cv2
import mediapipe as mp

# ==========================================
# 1. AYARLAR VE SABİTLER
# ==========================================
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CAM_WIDTH = 640
ROAD_WIDTH = 640

# Şerit Merkez X Koordinatları (Yol alanı 640 - 1280 arasında)
LANE_X = [
    760,   # 1. Şerit (Sol)
    960,   # 2. Şerit (Orta)
    1160   # 3. Şerit (Sağ)
]

# Renk Tanımlamaları (BGR formatında)
COLOR_ASPHALT = (38, 38, 38)
COLOR_ROAD_BORDER = (245, 245, 245)
COLOR_CURB_RED = (20, 20, 220)
COLOR_CURB_WHITE = (240, 240, 240)
COLOR_CAR_RED = (30, 30, 230)
COLOR_WINDOW = (220, 210, 150)
COLOR_WHEEL = (20, 20, 20)
COLOR_HEADLIGHT = (80, 255, 255)
COLOR_TAILLIGHT = (20, 20, 240)

# Rakip Araç Renk Paleti
OPPONENT_COLORS = [
    (30, 215, 255),   # Sarı
    (235, 140, 30),   # Mavi
    (240, 240, 240),  # Beyaz
    (20, 140, 255),   # Turuncu
    (190, 50, 190),   # Mor
    (60, 210, 80)     # Yeşil
]

# 21 El Landmark Bağlantı Çizgileri
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Başparmak
    (0, 5), (5, 6), (6, 7), (7, 8),           # İşaret parmağı
    (5, 9), (9, 10), (10, 11), (11, 12),      # Orta parmak
    (9, 13), (13, 14), (14, 15), (15, 16),    # Yüzük parmağı
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Serçe parmak ve avuç
]


# ==========================================
# 2. MODEL KONTROLÜ VE EL TAKİBİ
# ==========================================
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def ensure_model_file():
    """MediaPipe Tasks el takip modeli yoksa otomatik olarak indirir."""
    if not os.path.exists(MODEL_PATH):
        print("MediaPipe el takip modeli indiriliyor (hand_landmarker.task)...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Model başarıyla indirildi.")
        except Exception as e:
            print(f"Model indirme hatası: {e}")


class HandDetector:
    """
    MediaPipe ile el tespiti ve parmak sayımı yapan sınıf.
    Hem klasik solutions.hands (Python 3.9-3.12) hem de 
    yeni nesil Tasks API (Python 3.14+) ile tam uyumludur.
    """
    def __init__(self, max_num_hands=1):
        self.max_num_hands = max_num_hands
        self.finger_tips = [8, 12, 16, 20]
        self.finger_pips = [6, 10, 14, 18]

        # Sürüm uyumluluk kontrolü
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.mode = "solutions"
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                max_num_hands=max_num_hands,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
        else:
            self.mode = "tasks"
            ensure_model_file()
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_num_hands,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6
            )
            self.detector = vision.HandLandmarker.create_from_options(options)

    def draw_hand_mesh(self, frame, landmarks):
        """Tespit edilen 21 eklem noktasını ve bağlantı çizgilerini çizer."""
        # Bağlantı çizgileri
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = landmarks[start_idx]
            pt2 = landmarks[end_idx]
            cv2.line(frame, pt1, pt2, (0, 220, 255), 2)

        # Eklem noktaları
        for idx, (cx, cy) in enumerate(landmarks):
            if idx in [4, 8, 12, 16, 20]:
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)
            else:
                cv2.circle(frame, (cx, cy), 5, (0, 255, 100), -1)

    def find_hands(self, frame_bgr):
        """Görüntüdeki eli tespit eder, çizim yapar ve açık parmak sayısını döner."""
        h, w, _ = frame_bgr.shape
        finger_count = 0
        hand_type = "El Bulunamadi"
        landmarks_list = []

        if self.mode == "solutions":
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = np.ascontiguousarray(frame_rgb)
            results = self.hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    if results.multi_handedness:
                        label = results.multi_handedness[idx].classification[0].label
                        hand_type = "Sag El" if label == "Left" else "Sol El"
                    for lm in hand_lms.landmark:
                        landmarks_list.append((int(lm.x * w), int(lm.y * h)))
                    break
        else:
            # Tasks API
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = np.ascontiguousarray(frame_rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.detector.detect(mp_image)

            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                hand_lms = result.hand_landmarks[0]
                if result.handedness and len(result.handedness) > 0:
                    label = result.handedness[0][0].category_name
                    hand_type = "Sag El" if label == "Left" else "Sol El"

                for lm in hand_lms:
                    landmarks_list.append((int(lm.x * w), int(lm.y * h)))

        # Eklemleri çiz ve parmakları say
        if len(landmarks_list) == 21:
            self.draw_hand_mesh(frame_bgr, landmarks_list)

            # 1. Dört Parmak: İşaret (8), Orta (12), Yüzük (16), Serçe (20)
            for tip_idx, pip_idx in zip(self.finger_tips, self.finger_pips):
                if landmarks_list[tip_idx][1] < landmarks_list[pip_idx][1]:
                    finger_count += 1

            # 2. Başparmak: X-ekseni karşılaştırması
            # Sağ el: Başparmak ucu (4) sola açılır (x < eklem 3)
            # Sol el: Başparmak ucu (4) sağa açılır (x > eklem 3)
            if hand_type == "Sag El":
                if landmarks_list[4][0] < landmarks_list[3][0]:
                    finger_count += 1
            else:
                if landmarks_list[4][0] > landmarks_list[3][0]:
                    finger_count += 1

        return frame_bgr, finger_count, hand_type, landmarks_list


# ==========================================
# 3. ENGEL VE ARAÇ SINIFLARI
# ==========================================
class Obstacle:
    """Trafikteki diğer arabaları ve yol engellerini yöneten sınıf."""
    def __init__(self, lane_idx, obstacle_type="car"):
        self.lane_idx = lane_idx
        self.x = LANE_X[lane_idx]
        self.y = -120
        self.type = obstacle_type
        self.passed = False

        if self.type == "car":
            self.width = 65
            self.height = 110
            self.color = random.choice(OPPONENT_COLORS)
        else:
            self.width = 75
            self.height = 45
            self.color = (0, 140, 255)

    def update(self, speed):
        self.y += speed

    def get_hitbox(self):
        margin_x = 6
        margin_y = 6
        return (
            int(self.x - self.width // 2 + margin_x),
            int(self.y - self.height // 2 + margin_y),
            int(self.x + self.width // 2 - margin_x),
            int(self.y + self.height // 2 - margin_y)
        )

    def draw(self, canvas):
        cx, cy = int(self.x), int(self.y)

        if self.type == "car":
            w, h = self.width, self.height
            # Tekerlekler
            tw, th = 10, 20
            cv2.rectangle(canvas, (cx - w//2 - 3, cy - h//2 + 10), (cx - w//2 + tw - 3, cy - h//2 + 10 + th), COLOR_WHEEL, -1)
            cv2.rectangle(canvas, (cx + w//2 - tw + 3, cy - h//2 + 10), (cx + w//2 + 3, cy - h//2 + 10 + th), COLOR_WHEEL, -1)
            cv2.rectangle(canvas, (cx - w//2 - 3, cy + h//2 - 30), (cx - w//2 + tw - 3, cy + h//2 - 10), COLOR_WHEEL, -1)
            cv2.rectangle(canvas, (cx + w//2 - tw + 3, cy + h//2 - 30), (cx + w//2 + 3, cy + h//2 - 10), COLOR_WHEEL, -1)

            # Gövde
            cv2.rectangle(canvas, (cx - w//2, cy - h//2), (cx + w//2, cy + h//2), self.color, -1)
            cv2.rectangle(canvas, (cx - w//2, cy - h//2), (cx + w//2, cy + h//2), (20, 20, 20), 2)

            # Ön ve Arka Camlar
            cv2.rectangle(canvas, (cx - w//2 + 8, cy + 5), (cx + w//2 - 8, cy + 25), COLOR_WINDOW, -1)
            cv2.rectangle(canvas, (cx - w//2 + 10, cy - h//2 + 15), (cx + w//2 - 10, cy - h//2 + 30), COLOR_WINDOW, -1)
            # Tavan
            cv2.rectangle(canvas, (cx - w//2 + 10, cy - h//2 + 30), (cx + w//2 - 10, cy + 5), (self.color[0]//2, self.color[1]//2, self.color[2]//2), -1)

            # Farlar
            cv2.circle(canvas, (cx - w//2 + 10, cy + h//2 - 4), 6, COLOR_HEADLIGHT, -1)
            cv2.circle(canvas, (cx + w//2 - 10, cy + h//2 - 4), 6, COLOR_HEADLIGHT, -1)
            # Arka Stop Lambaları
            cv2.circle(canvas, (cx - w//2 + 10, cy - h//2 + 4), 5, COLOR_TAILLIGHT, -1)
            cv2.circle(canvas, (cx + w//2 - 10, cy - h//2 + 4), 5, COLOR_TAILLIGHT, -1)

        elif self.type == "barrier":
            w, h = self.width, self.height
            # Ayaklar
            cv2.rectangle(canvas, (cx - w//2 + 6, cy + h//2 - 6), (cx - w//2 + 18, cy + h//2 + 6), (50, 50, 50), -1)
            cv2.rectangle(canvas, (cx + w//2 - 18, cy + h//2 - 6), (cx + w//2 - 6, cy + h//2 + 6), (50, 50, 50), -1)

            # Bariyer Gövdesi
            cv2.rectangle(canvas, (cx - w//2, cy - h//2), (cx + w//2, cy + h//2), (0, 140, 255), -1)
            cv2.rectangle(canvas, (cx - w//2, cy - h//2), (cx + w//2, cy + h//2), (20, 20, 20), 2)

            # Beyaz Çapraz Çizgiler
            for offset in range(-w//2, w//2, 18):
                pt1 = (cx + offset, cy - h//2)
                pt2 = (cx + offset + 12, cy + h//2)
                cv2.line(canvas, pt1, pt2, (255, 255, 255), 4)

            # Uyarı Işığı
            cv2.circle(canvas, (cx, cy - h//2 - 5), 7, (0, 255, 255), -1)
            cv2.circle(canvas, (cx, cy - h//2 - 5), 4, (0, 180, 255), -1)


# ==========================================
# 4. OYUNCU ARABASI SINIFI
# ==========================================
class PlayerCar:
    """Kırmızı spor oyuncu arabası ve şerit kontrolü."""
    def __init__(self):
        self.current_lane = 1       # 2. Şerit (Orta)
        self.x = float(LANE_X[1])
        self.y = 590
        self.width = 65
        self.height = 115
        self.speed_lerp = 0.22      # Akıcı şerit değişimi

    def set_lane_by_finger(self, finger_count):
        if finger_count == 1:
            self.current_lane = 0  # 1. Şerit (Sol)
        elif finger_count == 2:
            self.current_lane = 1  # 2. Şerit (Orta)
        elif finger_count == 3:
            self.current_lane = 2  # 3. Şerit (Sağ)

    def update(self):
        target_x = LANE_X[self.current_lane]
        self.x += (target_x - self.x) * self.speed_lerp

    def get_hitbox(self):
        margin_x = 7
        margin_y = 7
        return (
            int(self.x - self.width // 2 + margin_x),
            int(self.y - self.height // 2 + margin_y),
            int(self.x + self.width // 2 - margin_x),
            int(self.y + self.height // 2 - margin_y)
        )

    def draw(self, canvas):
        cx, cy = int(self.x), int(self.y)
        w, h = self.width, self.height

        # 1. Far Işık Hüzmesi
        overlay = canvas.copy()
        beam_poly = np.array([
            [cx - 20, cy - h//2],
            [cx - 45, cy - h//2 - 90],
            [cx + 45, cy - h//2 - 90],
            [cx + 20, cy - h//2]
        ], np.int32)
        cv2.fillPoly(overlay, [beam_poly], (80, 255, 255))
        cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)

        # 2. Tekerlekler
        tw, th = 11, 22
        cv2.rectangle(canvas, (cx - w//2 - 4, cy - h//2 + 18), (cx - w//2 + tw - 4, cy - h//2 + 18 + th), COLOR_WHEEL, -1)
        cv2.rectangle(canvas, (cx + w//2 - tw + 4, cy - h//2 + 18), (cx + w//2 + 4, cy - h//2 + 18 + th), COLOR_WHEEL, -1)
        cv2.rectangle(canvas, (cx - w//2 - 4, cy + h//2 - 38), (cx - w//2 + tw - 4, cy + h//2 - 16), COLOR_WHEEL, -1)
        cv2.rectangle(canvas, (cx + w//2 - tw + 4, cy + h//2 - 38), (cx + w//2 + 4, cy + h//2 - 16), COLOR_WHEEL, -1)

        # 3. Kırmızı Gövde
        cv2.rectangle(canvas, (cx - w//2, cy - h//2), (cx + w//2, cy + h//2), COLOR_CAR_RED, -1)
        cv2.rectangle(canvas, (cx - w//2, cy - h//2), (cx + w//2, cy + h//2), (10, 10, 130), 2)

        # 4. Spor Beyaz Yarış Şeritleri
        cv2.rectangle(canvas, (cx - 7, cy - h//2), (cx - 2, cy + h//2), (245, 245, 245), -1)
        cv2.rectangle(canvas, (cx + 2, cy - h//2), (cx + 7, cy + h//2), (245, 245, 245), -1)

        # 5. Ön Cam
        pts_front_glass = np.array([
            [cx - w//2 + 10, cy - 8],
            [cx + w//2 - 10, cy - 8],
            [cx + w//2 - 14, cy - 35],
            [cx - w//2 + 14, cy - 35]
        ], np.int32)
        cv2.fillPoly(canvas, [pts_front_glass], (230, 210, 140))
        cv2.polylines(canvas, [pts_front_glass], True, (30, 30, 30), 1)

        # 6. Siyah Tavan ve Arka Cam
        cv2.rectangle(canvas, (cx - w//2 + 12, cy - 8), (cx + w//2 - 12, cy + 20), (35, 35, 35), -1)
        pts_back_glass = np.array([
            [cx - w//2 + 12, cy + 20],
            [cx + w//2 - 12, cy + 20],
            [cx + w//2 - 15, cy + 38],
            [cx - w//2 + 15, cy + 38]
        ], np.int32)
        cv2.fillPoly(canvas, [pts_back_glass], (190, 180, 120))

        # 7. Ön Farlar
        cv2.circle(canvas, (cx - w//2 + 10, cy - h//2 + 6), 6, COLOR_HEADLIGHT, -1)
        cv2.circle(canvas, (cx + w//2 - 10, cy - h//2 + 6), 6, COLOR_HEADLIGHT, -1)

        # 8. Arka Stop Lambaları
        cv2.rectangle(canvas, (cx - w//2 + 6, cy + h//2 - 6), (cx - w//2 + 18, cy + h//2 - 2), COLOR_TAILLIGHT, -1)
        cv2.rectangle(canvas, (cx + w//2 - 18, cy + h//2 - 6), (cx + w//2 - 6, cy + h//2 - 2), COLOR_TAILLIGHT, -1)


# ==========================================
# 5. TRAFİK OYUN MOTORU
# ==========================================
class TrafficGame:
    """Trafik akışı, dalgalar, skor, hız ve çarpışma kontrolü."""
    def __init__(self):
        self.player = PlayerCar()
        self.obstacles = []
        self.score = 0
        self.high_score = 0
        self.game_over = False
        
        self.base_speed = 7.0
        self.current_speed = self.base_speed
        self.road_scroll_offset = 0
        self.dash_length = 40
        self.dash_gap = 35
        self.dash_total = self.dash_length + self.dash_gap

        # Ardışık dalgalar arası dikey mesafe: En az oyuncu araba boyunun 3.5 katı (115 * 3.6 = 414 piksel)
        self.min_spawn_gap = int(self.player.height * 3.6)
        self.spawn_interval = self.min_spawn_gap
        self.spawn_distance_counter = 0

        # Koridor sürekliliği takibi (çapraz/kapanan koridorları engellemek için)
        self.last_open_lanes = [0, 1, 2]
        self.last_wave_obstacle_count = 0

    def reset(self):
        if self.score > self.high_score:
            self.high_score = self.score
        self.player = PlayerCar()
        self.obstacles.clear()
        self.score = 0
        self.current_speed = self.base_speed
        self.spawn_distance_counter = 0
        self.spawn_interval = self.min_spawn_gap
        self.last_open_lanes = [0, 1, 2]
        self.last_wave_obstacle_count = 0
        self.game_over = False

    def check_collision(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

    def spawn_wave(self):
        """
        Gelişmiş ve Güvenli Dalga Üretimi:
        1. Asla 3 şerit birden kapanmaz (en az 1 şerit daima açık).
        2. Ardışık dalgalar arası mesafe araba boyunun en az 3.5 katıdır.
        3. Birbirini takip eden iki dalgada geçiş koridorunu kapatan çapraz/çıkmaz
           tuzaklar engellenir (örn. 0 açıkken bir sonraki dalgada ani 2 açık yapılamaz).
        4. Üst üste çift engel çıkması sınırlandırılarak güvenli kaçış koridoru garanti edilir.
        """
        all_lanes = {0, 1, 2}

        # Eğer önceki dalga 2 engelli ise, oyuncuya rahat geçiş için bu dalgada tek engel gelsin
        if self.last_wave_obstacle_count >= 2:
            num_obstacles = 1
        else:
            # %65 tek engel (ferah geçiş), %35 çift engel
            num_obstacles = 1 if random.random() < 0.65 else 2

        valid_combinations = []

        if num_obstacles == 1:
            for lane in [0, 1, 2]:
                open_lanes = all_lanes - {lane}
                # Açık kalan 2 şeritten en az biri önceki açık şeritle örtüşüyor veya doğrudan komşu mu?
                if any(ol in self.last_open_lanes or any(abs(ol - prev_ol) <= 1 for prev_ol in self.last_open_lanes) for ol in open_lanes):
                    valid_combinations.append([lane])
        else:
            # 2 engel koyulurken geriye kalan 1 açık şerit, önceki açık şeritle doğrudan komşu veya aynı olmalı
            # (Örn: Önceki dalgada sadece Şerit 0 açıksa, yeni dalgada Şerit 2 tek açık OLAMAZ)
            for open_candidate in [0, 1, 2]:
                can_reach = any(abs(open_candidate - prev_ol) <= 1 for prev_ol in self.last_open_lanes)
                if can_reach:
                    chosen_obs = list(all_lanes - {open_candidate})
                    valid_combinations.append(chosen_obs)

        if not valid_combinations:
            chosen_lanes = [random.choice([0, 1, 2])]
        else:
            chosen_lanes = random.choice(valid_combinations)

        # Seçilen şeritlere engelleri oluştur
        for lane in chosen_lanes:
            obs_type = "car" if random.random() < 0.65 else "barrier"
            self.obstacles.append(Obstacle(lane_idx=lane, obstacle_type=obs_type))

        # Sonraki dalga için açık şeritleri ve engel sayısını kaydet
        self.last_open_lanes = list(all_lanes - set(chosen_lanes))
        self.last_wave_obstacle_count = len(chosen_lanes)

    def update(self, finger_count):
        if self.game_over:
            return

        self.player.set_lane_by_finger(finger_count)
        self.player.update()

        self.road_scroll_offset = (self.road_scroll_offset + self.current_speed) % self.dash_total

        self.spawn_distance_counter += self.current_speed
        if self.spawn_distance_counter >= self.spawn_interval:
            self.spawn_wave()
            self.spawn_distance_counter = 0

        player_hitbox = self.player.get_hitbox()
        remaining_obstacles = []

        for obs in self.obstacles:
            obs.update(self.current_speed)

            if self.check_collision(player_hitbox, obs.get_hitbox()):
                self.game_over = True
                if self.score > self.high_score:
                    self.high_score = self.score

            if not obs.passed and obs.y > self.player.y + 40:
                obs.passed = True
                self.score += 1
                if self.current_speed < 18.0:
                    self.current_speed = self.base_speed + (self.score * 0.35)

            if obs.y < WINDOW_HEIGHT + 100:
                remaining_obstacles.append(obs)

        self.obstacles = remaining_obstacles

    def draw_road(self, canvas):
        cv2.rectangle(canvas, (640, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), COLOR_ASPHALT, -1)

        # Bordürler
        curb_width = 16
        block_h = 40
        offset_curb = int(self.road_scroll_offset) % (block_h * 2)

        for y in range(-block_h * 2, WINDOW_HEIGHT + block_h * 2, block_h):
            actual_y = y + offset_curb
            color = COLOR_CURB_RED if ((y // block_h) % 2 == 0) else COLOR_CURB_WHITE
            cv2.rectangle(canvas, (640, actual_y), (640 + curb_width, actual_y + block_h), color, -1)
            cv2.rectangle(canvas, (WINDOW_WIDTH - curb_width, actual_y), (WINDOW_WIDTH, actual_y + block_h), color, -1)

        # Sınır Çizgileri
        cv2.line(canvas, (640 + curb_width + 4, 0), (640 + curb_width + 4, WINDOW_HEIGHT), COLOR_ROAD_BORDER, 3)
        cv2.line(canvas, (WINDOW_WIDTH - curb_width - 4, 0), (WINDOW_WIDTH - curb_width - 4, WINDOW_HEIGHT), COLOR_ROAD_BORDER, 3)

        # Kesikli Şerit Çizgileri
        lane_dividers_x = [860, 1060]
        offset_dash = int(self.road_scroll_offset)

        for div_x in lane_dividers_x:
            for y in range(-self.dash_total, WINDOW_HEIGHT + self.dash_total, self.dash_total):
                y_start = y + offset_dash
                y_end = y_start + self.dash_length
                cv2.line(canvas, (div_x, y_start), (div_x, y_end), (255, 255, 255), 4)

    def draw(self, canvas):
        self.draw_road(canvas)

        for obs in self.obstacles:
            obs.draw(canvas)

        self.player.draw(canvas)

        # Skor ve Hız Paneli
        hud_overlay = canvas.copy()
        cv2.rectangle(hud_overlay, (670, 15), (1250, 75), (20, 20, 20), -1)
        cv2.addWeighted(hud_overlay, 0.65, canvas, 0.35, 0, canvas)
        cv2.rectangle(canvas, (670, 15), (1250, 75), (0, 215, 255), 2)

        score_text = f"SKOR: {self.score}"
        speed_text = f"HIZ: {int(self.current_speed * 10)} km/h"
        high_text = f"REKOR: {self.high_score}"

        cv2.putText(canvas, score_text, (690, 55), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(canvas, speed_text, (900, 55), cv2.FONT_HERSHEY_DUPLEX, 0.75, (100, 255, 100), 2)
        cv2.putText(canvas, high_text, (1090, 55), cv2.FONT_HERSHEY_DUPLEX, 0.75, (220, 220, 220), 2)

        # Kaza Durumu
        if self.game_over:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (640, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 50), -1)
            cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

            box_x1, box_y1 = 690, 220
            box_x2, box_y2 = 1230, 480
            cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (25, 25, 25), -1)
            cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 240), 3)

            cv2.putText(canvas, "KAZA YAPTINIZ!", (box_x1 + 65, box_y1 + 70),
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 255), 3)

            final_score_str = f"Toplam Skor: {self.score}"
            cv2.putText(canvas, final_score_str, (box_x1 + 140, box_y1 + 130),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)

            cv2.putText(canvas, "Yeniden Baslamak icin 'R' tusuna basin", (box_x1 + 25, box_y1 + 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
            cv2.putText(canvas, "Cikis yapmak icin 'Q' tusuna basin", (box_x1 + 60, box_y1 + 225),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)


# ==========================================
# 6. ARAYÜZ VE BİLGİ KUTULARI (HUD)
# ==========================================
def draw_hand_hud(canvas, finger_count, hand_type, current_lane):
    hud_overlay = canvas.copy()
    cv2.rectangle(hud_overlay, (15, 15), (420, 200), (25, 25, 25), -1)
    cv2.addWeighted(hud_overlay, 0.7, canvas, 0.3, 0, canvas)
    cv2.rectangle(canvas, (15, 15), (420, 200), (0, 200, 255), 2)

    cv2.putText(canvas, "EL TAKIP & KONTROL", (30, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 215, 255), 2)
    cv2.line(canvas, (30, 55), (405, 55), (80, 80, 80), 1)

    hand_color = (0, 255, 100) if hand_type != "El Bulunamadi" else (0, 0, 220)
    cv2.putText(canvas, f"Tespit: {hand_type}", (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, hand_color, 2)

    cv2.putText(canvas, f"Acik Parmak: {finger_count}", (30, 120),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    lane_names = ["1. Serit (SOL)", "2. Serit (ORTA)", "3. Serit (SAG)"]
    lane_str = lane_names[current_lane]
    cv2.putText(canvas, f"Hedef: {lane_str}", (30, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    cv2.putText(canvas, "1: Sol | 2: Orta | 3: Sag", (30, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


# ==========================================
# 7. ANA ÇALIŞTIRMA FONKSİYONU (MAIN)
# ==========================================
def main():
    print("=" * 60)
    print("Kamerayla El Takibi Trafik Oyunu Başlatılıyor...")
    print("Kontroller: 1 Parmak=Sol | 2 Parmak=Orta | 3 Parmak=Sağ")
    print("Yeniden Başlatmak: 'R' | Çıkış: 'Q'")
    print("=" * 60)

    # 1. Gerçek Webcam Başlatma
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("HATA: Web kamerası (VideoCapture(0)) açılamadı!")
        print("Lütfen kameranızın bağlı ve başka bir uygulama tarafından kullanılmadığını kontrol edin.")
        return

    detector = HandDetector(max_num_hands=1)
    game = TrafficGame()

    window_name = "El Takibi ile Trafik Oyunu - OpenCV & MediaPipe"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.time()

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("Kameradan görüntü alınamadı, çıkılıyor...")
            break

        # 2. Görüntüyü Yatay Aynala (Doğal ayna etkisi)
        frame = cv2.flip(frame, 1)

        # 3. Kameranın en-boy oranını bozmadan sol panele (640x720) yerleştir
        fh, fw = frame.shape[:2]
        cam_view = np.zeros((WINDOW_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
        scale = min(CAM_WIDTH / fw, WINDOW_HEIGHT / fh)
        new_w, new_h = int(fw * scale), int(fh * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h),
                                   interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        offset_x = (CAM_WIDTH - new_w) // 2
        offset_y = (WINDOW_HEIGHT - new_h) // 2
        cam_view[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_frame

        # 4. MediaPipe ile 21 Eklem Noktası Tespiti ve Parmak Sayımı
        cam_view, finger_count, hand_type, _ = detector.find_hands(cam_view)

        # 5. Açık Parmak Sayısına Göre Oyunu Yönet (1: Sol, 2: Orta, 3: Sağ)
        game.update(finger_count)

        # 6. Birleşik Tuval (Sol: Kamera & El Takibi, Sağ: Trafik Oyunu)
        canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        canvas[0:WINDOW_HEIGHT, 0:CAM_WIDTH] = cam_view
        game.draw(canvas)

        # 7. Sol Üst Bilgi Paneli (Tespit, Parmak Sayısı, Aktif Şerit)
        draw_hand_hud(canvas, finger_count, hand_type, game.player.current_lane)

        # 8. FPS Gösterimi
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30
        prev_time = curr_time
        cv2.putText(canvas, f"FPS: {int(fps)}", (CAM_WIDTH - 110, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(window_name, canvas)

        # 9. Tuş Kontrolleri
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break
        elif key == ord('r') or key == ord('R'):
            game.reset()

    cap.release()
    cv2.destroyAllWindows()
    print("Program kapatıldı.")
    cv2.destroyAllWindows()
    print("Program başarıyla sonlandırıldı.")


if __name__ == "__main__":
    main()
