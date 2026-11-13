# Proje 2: Kamerayla El Takibi ve Parmak Kontrollü Araba Oyunu

Bu proje, OpenCV ve MediaPipe kütüphaneleri kullanılarak geliştirilmiş gerçek zamanlı bir el takibi uygulamasıdır

## Proje Gereksinimleri

### Temel İstenenler
1. **Canlı Görüntü Akışı:** OpenCV kullanılarak bilgisayara bağlı webcam'den gerçek zamanlı ve ayna modunda görüntü alınmaktadır.
2. **El ve Eklem Tespiti:** MediaPipe Hands modeli ile görüntüdeki el algılanmakta ve el üzerindeki **21 farklı eklem noktası (landmark)** ile bağlantıları canlı görüntü üzerine çizilmektedir.
3. **Açık Parmak Sayma:** Parmak uçlarının ve alt eklemlerinin koordinatları karşılaştırılarak o an kaç parmağın açık olduğu hesaplanmakta ve anlık olarak ekrana yazdırılmaktadır.

### ⭐ Bonus: Parmak Sayısı / Hareketi ile Oyun Kontrolü
Projenin bonus aşaması olarak, klavye veya fare gerektirmeyen bir **Trafik / Engelden Kaçma Oyunu** entegre edilmiştir:
- **Kontrol Mekanizması:** Kameradan algılanan açık parmak sayısına / el konumuna göre oyuncunun aracı şeritler arasında hareket ettirilir.
- **Oyun Döngüsü:** Ekranda yukarıdan aşağıya doğru gelen engellerden kaçıldıkça skor artar; çarpışma durumunda oyun sonlanır ve yeniden başlatılabilir.

---

## 🛠️ Kullanılan Teknolojiler
- **Python 3.x**
- **OpenCV (`opencv-python`):** Kamera erişimi, görüntü işleme ve oyun arayüzü çizimleri
- **MediaPipe (`mediapipe`):** Gerçek zamanlı el tespiti ve 21 eklem noktası çıkarımı

---
