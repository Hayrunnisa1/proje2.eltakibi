# 🚗 Kamerayla El Takibi ve Parmak Sayısıyla Kontrol Edilen Trafik Oyunu

Bu proje, bilgisayar kamerası (Webcam) üzerinden **MediaPipe** ve **OpenCV** kütüphaneleri kullanılarak gerçek zamanlı el takibi ve parmak sayımı yapan, tespit edilen parmak sayısıyla kontrol edilen dinamik ve eğlenceli bir **2D Trafik ve Engellerden Kaçış Oyunu**dur.

---

## 📌 Proje Özellikleri

### 🖐️ 1. Gerçek Zamanlı El Takibi ve Parmak Sayma
- **Doğal Aynalama:** Kamera görüntüsü yatayda ters çevrilir (`cv2.flip`), böylece kullanıcının hareketleri ayna karşısındaymış gibi sezgisel hissettirir.
- **21 Eklem Noktası:** MediaPipe Hands modülüyle eldeki 21 landmark noktası ve bağlantı iskeleti çizilir.
- **Akıllı Parmak Algoritması:**
  - **İşaret, Orta, Yüzük, Serçe:** Parmak uçlarının ($tip$) $y$-koordinatı ile alt eklemlerinin ($pip$) $y$-koordinatı karşılaştırılarak açık/kapalı durumu tespit edilir.
  - **Başparmak:** Sağ ve sol el ayrımı (`multi_handedness`) yapılarak başparmak ucunun $x$-koordinatına göre açılma yönü doğru şekilde hesaplanır.
- **Okunaklı HUD:** Tespit edilen el türü (Sağ/Sol), açık parmak sayısı ve seçili şerit ekranın sol üstündeki yarı saydam panelde gösterilir.

---

### 🏎️ 2. Trafik ve Engellerden Kaçış Oyunu
- **3 Şeritli Dinamik Asfalt Yol:** Ekranın sağ tarafında dikey olarak akan, kesikli şerit çizgilerine ve kırmızı-beyaz bordür taşlarına sahip 3 şeritli yol.
- **Kırmızı Spor Araba:** Aerodinamik kırmızı gövde, beyaz yarış şeritleri, pencereler, siyah spor tekerlekler, farlar ve yol aydınlatma hüzmesi ile tamamen OpenCV ilkel şekilleriyle çizilmiş oyuncu aracı.
- **Akıcı Şerit Değişimi (Smooth Lerp):** Araba şeritler arasında aniden zıplamak yerine yumuşak bir kayma animasyonu ile geçer.
- **Çeşitli Engeller:**
  1. **Rakip Araçlar:** Sarı, mavi, turuncu, beyaz, mor gibi farklı renklerde trafikte ilerleyen araçlar.
  2. **Yol Bariyerleri:** Turuncu-beyaz çapraz çizgili ve uyarı ışıklı trafik dubaları/barikatları.
- **Adil Oyun ve Güvenli Koridor Kuralı (Fair Wave Spawn & Corridor Continuity):**
  - Ardışık dalgalar arasındaki dikey mesafe, oyuncu arabasının boyunun **en az 3.5 katı** (~415 piksel) olarak ayarlanmıştır; böylece şerit değiştirmek için yeterli zaman ve mesafe tanınır.
  - Aynı dalgada asla 3 şerit birden kapanmaz (en az 1 şerit daima açık kalır).
  - Birbirini takip eden iki dalgada oyuncunun geçişini imkansız kılan çapraz çıkmazlar (örneğin sol şerit açıkken bir sonraki dalgada ani sağ şerit engeli) matematiksel olarak önlenir; açık koridorlar komşu veya kesintisiz şekilde devam eder.
- **Skor ve Hızlanma:** Atlatılan her engel için skor +1 artar ve oyun hızı kademeli olarak yükselir.
- **Kaza ve Yeniden Başlama:** Çarpışma anında kaza ekranı açılır ve toplam skor görüntülenir.

---

## 🎮 Kontroller

| El Hareketi / Tuş | İşlev |
| :--- | :--- |
| ☝️ **1 Parmak Açık** | **1. Şerit (Sol)** şeride geçer |
| ✌️ **2 Parmak Açık** | **2. Şerit (Orta)** şeride geçer |
| 🤟 **3 Parmak Açık** | **3. Şerit (Sağ)** şeride geçer |
| ✊ **0, 4 veya 5 Parmak** | Mevcut şeridi korur (araba son şeridinde kalır) |
| **`R` Tuşu** | Kazadan sonra oyunu yeniden başlatır |
| **`Q` veya `ESC` Tuşu** | Programı kapatır |

---

## 🛠️ Kurulum Adımları

### 1. Depoyu Klonlayın veya İndirin
```bash
git clone https://github.com/kullaniciadi/el-takibi-trafik-oyunu.git
cd el-takibi-trafik-oyunu
```

### 2. Gerekli Kütüphaneleri Yükleyin
Sisteminizde Python 3.9+ yüklü olduğundan emin olun ve terminalde şu komutu çalıştırın:

```bash
pip install -r requirements.txt
```

> **Not:** `requirements.txt` içeriği:
> - `opencv-python`
> - `mediapipe`
> - `numpy`

---

## 🚀 Oyunu Çalıştırma

Kameranızın bağlı olduğundan emin olduktan sonra terminalden projeyi başlatın:

```bash
python main.py
```

### 💡 İpuçları
- Kameranızın elinizi net görebileceği iyi aydınlatılmış bir ortamda oynayın.
- Elinizi kameraya doğru yaklaşık 40 - 70 cm mesafeden gösterin.
- Parmaklarınızı net bir şekilde açıp kapatarak şeritler arasında hızla geçiş yapabilirsiniz.

---

## 📂 Dosya Yapısı

```
el.takibi.oyunu/
│
├── main.py              # El takibi, oyun motoru, çizimler ve ana döngü
├── requirements.txt     # Gerekli Python bağımlılıkları
└── README.md            # Detaylı proje dokümantasyonu
```

---

## 📜 Lisans
Bu proje açık kaynaklı olup eğitim ve kişisel gelişim amaçlı geliştirilmiştir. Dilediğiniz gibi geliştirebilir ve paylaşabilirsiniz.

