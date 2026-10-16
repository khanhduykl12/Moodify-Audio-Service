# Moodify Audio Feature Analysis Service 🎵🔍
> Dịch vụ phân tích và trích xuất đặc trưng âm học (Acoustic Feature Extraction) từ sóng âm dành cho hệ sinh thái phát nhạc và gợi ý theo cảm xúc **Moodify**.

---

## 📌 1. Tổng quan kiến trúc & Công nghệ

Service này chịu trách nhiệm "nghe" trực tiếp file âm thanh tải lên (`.mp3`, `.wav`, `.m4a`...), phân rã tín hiệu sóng âm và tính toán các đặc trưng âm học (Acoustic Features) tương tự như hệ sinh thái **Spotify Web API / The Echo Nest**.

* **Ngôn ngữ & Framework:** Python 3.10+, FastAPI (High performance ASGI).
* **Cốt lõi xử lý tín hiệu âm thanh (Audio DSP):** **Essentia C++ Library** (Phát triển bởi Music Technology Group - Đại học Pompeu Fabra, Barcelona).
* **Môi trường vận hành:** Docker Container (Linux x86_64) để tận dụng bộ biên dịch C++ tăng tốc xử lý tín hiệu thời gian thực.
* **Cổng dịch vụ (Port):** `8002` (trong docker-compose) hoặc `8000` nội bộ container.

---

## ⚙️ 2. Quy trình xử lý sóng âm (Processing Pipeline)

```
[File Audio .mp3/.wav] 
          │
          ▼
   1. MonoLoader (44.1 kHz, 16-bit Float)
          │
          ├────────────────────────────────────────────────────────┐
          ▼                                                        ▼
  2. Rhythm Analysis                                       3. Tonal Analysis
  (RhythmExtractor2013)                                      (KeyExtractor)
   ➔ BPM (Tempo)                                             ➔ Key & Scale (Major / Minor)
          │                                                        │
          ├──────────────────────────┬─────────────────────────────┤
          ▼                          ▼                             ▼
  4. Temporal Energy         5. Spectral Centroid          6. Rhythm Regularity
     (Energy Integration)       (SpectralCentroidTime)        (Danceability - DFA)
   ➔ Energy                   ➔ Âm sắc sáng/tối             ➔ Danceability
          │                          │                             │
          └──────────────────────────┴─────────────────────────────┘
                                     │
                                     ▼
                    7. Heuristic Multi-Modal Fusion
                     ➔ Valence (Sắc thái Vui / Buồn)
                     ➔ Acousticness (Độ mộc của âm thanh)
                                     │
                                     ▼
                        [JSON Response trả về Client]
```

---

## 🧮 3. Giải thích chi tiết thuật toán & Cách tính từng chỉ số

### 3.1. BPM / Tempo (Tốc độ nhịp đập mỗi phút)
* **Thuật toán:** `es.RhythmExtractor2013(method="multifeature")` (Zapata et al., 2013).
* **Nguyên lý:**
  1. Sử dụng **Hàm phát hiện xung nhịp (Onset Detection Function - ODF)** để định vị các điểm bùng nổ năng lượng đột ngột (tiếng kick drum, tiếng snare, tiếng gảy dây).
  2. Tách sóng âm thành các dải tần độc lập (sub-band).
  3. Áp dụng phép **Tự tương quan (Autocorrelation)** để tìm chu kỳ lặp lại $\Delta t$ mạnh nhất giữa các phách nhịp.
  4. Tính tốc độ:
     $$\text{BPM} = \frac{60}{\Delta t}$$

---

### 3.2. Key Signature (Điệu tính bài hát: C Major, A Minor...)
* **Thuật toán:** `es.KeyExtractor()` (Thuật toán Krumhansl-Schmuckler).
* **Nguyên lý:**
  1. Trích xuất vector **HPCP (Harmonic Pitch Class Profile)** gồm năng lượng của 12 nốt bán âm trong thang âm: $[C, C\sharp, D, D\sharp, E, F, F\sharp, G, G\sharp, A, A\sharp, B]$.
  2. Tính tổng thời lượng và cường độ âm mà mỗi nốt nhạc vang lên trong toàn bộ bài hát.
  3. Tính hệ số tương quan (Correlation Coefficient) của vector này với **24 vector mẫu chuẩn** (12 điệu Trưởng - Major và 12 điệu Thứ - Minor).
  4. Điệu nào đạt độ tương đồng cao nhất sẽ được chọn (ví dụ: `A Minor`, `C Major`).

---

### 3.3. Danceability (Độ nảy & Tính bắt tai khi khiêu vũ)
* **Thuật toán:** `es.Danceability()`
* **Nguyên lý:**
  1. Áp dụng phương pháp phân tích dao động xu hướng **DFA (Detrended Fluctuation Analysis)** trên tín hiệu nhịp điệu.
  2. Đo lường **tính ổn định và độ dễ dự đoán (Predictability)** của tiết tấu:
     - Nhạc có phách đều, nhịp 4/4 gãy gọn, lặp lại ổn định (*House, Disco, Pop, EDM*) $\rightarrow$ Não bộ dễ dàng hòa nhịp để nhún nhảy $\rightarrow$ **Danceability cao ($0.7 - 0.95$)**.
     - Nhạc tự do, thay đổi nhịp thất thường, không có beat gõ (*nhạc thiền, thính phòng, vocal tự do*) $\rightarrow$ **Danceability thấp ($0.1 - 0.4$)**.

---

### 3.4. Energy (Năng lượng âm thanh)
* **Đoạn mã:**
  ```python
  energy_val = float(es.Energy()(audio))
  normalized_energy = round(min(energy_val / (len(audio) * 0.02 + 1e-6), 1.0), 3)
  ```
* **Nguyên lý:**
  1. Tích phân bình phương biên độ tín hiệu theo thời gian (tương đương công suất hiệu dụng RMS - Root Mean Square):
     $$\text{Energy}_{\text{raw}} = \sum_{n=1}^{N} x[n]^2$$
  2. Chia cho tổng số mẫu $N$ (`len(audio)`) để chuẩn hóa theo độ dài bài hát (tránh việc bài hát dài hơn bị tính năng lượng cao hơn bài ngắn).
  3. Giá trị nằm trong khoảng $[0.05, 1.0]$. Nhạc dồn dập, âm lượng tổng thể lớn, nhiều nhạc cụ điện tử $\rightarrow$ **Energy gần 1.0**.

---

### 3.5. Valence (Sắc thái cảm xúc: Buồn ➔ Vui)
* **Đoạn mã:**
  ```python
  spec_centroid = float(es.SpectralCentroidTime()(audio))
  spectral_ratio = min(max(spec_centroid / 4000.0, 0.0), 1.0)
  mode_weight = 0.6 if scale.lower() == "major" else 0.4
  valence = round(0.4 * spectral_ratio + 0.3 * danceability + 0.3 * mode_weight, 3)
  ```
* **Nguyên lý (Mô hình kết hợp đa chiều):**
  1. **Trọng tâm phổ tần số (Spectral Centroid):**
     $$C = \frac{\sum f \cdot |X(f)|}{\sum |X(f)|}$$
     Đo xem năng lượng âm thanh dồn nhiều ở dải trầm (Bass/u tối) hay dải cao (Treble/sáng sủa). Trọng tâm phổ càng cao, âm sắc nghe càng "sáng" (bright), kích thích cảm giác tích cực.
  2. **Thang âm nhạc lý (Scale Mode):** Thang âm Trưởng (`Major`) gợi cảm giác tươi vui, lạc quan ($0.6$), thang âm Thứ (`Minor`) gợi cảm giác trầm mặc, u hoài ($0.4$).
  3. **Độ nảy tiết tấu (Danceability):** Tiết tấu sôi động mang lại cảm xúc hưng phấn ($0.3$).
  4. **Kết hợp:** $Valence \in [0.05, 0.95]$:
     - Gần $0.0$: Cực kỳ u buồn, tang tóc, trầm cảm.
     - Gần $1.0$: Vui vẻ, hân hoan, tràn đầy hy vọng.

---

### 3.6. Acousticness (Độ mộc của âm thanh)
* **Đoạn mã:**
  ```python
  high_energy_factor = normalized_energy * 0.7 + (1.0 if bpm > 125 else 0.0) * 0.2
  acousticness = round(max(0.05, min(1.0 - high_energy_factor, 0.95)), 3)
  ```
* **Nguyên lý:**
  1. Nhạc điện tử (EDM, Synthwave, Dubstep) sử dụng sóng âm nhân tạo, dải động bị nén chặt (ít khoảng lặng), nhịp dồn dập ($BPM > 125$).
  2. Nhạc mộc (Acoustic: Guitar gỗ, Piano cổ điển, Trống Jazz, Hát mộc) có độ động tự nhiên cao, âm thanh ngân vang và tắt tự nhiên (Natural Decay), nhiều khoảng thở giữa các nốt.
  3. Thuật toán thiết lập hàm nghịch đảo với hệ số năng lượng điện tử và nhịp dồn dập để ước lượng độ mộc từ $0.05$ đến $0.95$.

---

## 📡 4. Đặc tả API (API Endpoints)

### 4.1. Kiểm tra trạng thái máy chủ
* **Endpoint:** `GET /health`
* **Response:**
  ```json
  {
    "status": "ok",
    "service": "essentia-audio-extractor",
    "engine": "essentia"
  }
  ```

### 4.2. Phân tích đặc trưng âm học bài hát
* **Endpoint:** `POST /analyze`
* **Content-Type:** `multipart/form-data`
* **Tham số Body:**
  * `file`: File âm thanh (`.mp3`, `.wav`, `.ogg`...)
* **Response mẫu:**
  ```json
  {
    "bpm": 128.0,
    "keySignature": "C Major",
    "energy": 0.812,
    "danceability": 0.745,
    "valence": 0.685,
    "acousticness": 0.142,
    "durationMs": 214500
  }
  ```

---

## 🧪 5. Cách kiểm tra độ chính xác của Model

### Phương pháp 1: Kiểm thử bằng 3 bài hát đối lập (Thực nghiệm nhanh)
1. **Bài EDM / Vinahouse tốc độ cao:**
   * *Kỳ vọng:* BPM $128 - 145$, Energy $> 0.8$, Danceability $> 0.75$, Acousticness $< 0.15$.
2. **Bài Ballad / Nhạc buồn mộc (Acoustic Guitar/Piano):**
   * *Kỳ vọng:* BPM $60 - 80$, Energy $< 0.4$, Scale ra **Minor**, Valence $< 0.4$, Acousticness $> 0.6$.
3. **Bài Pop vui tươi:**
   * *Kỳ vọng:* Scale ra **Major**, Valence $> 0.65$.

### Phương pháp 2: So sánh trực tiếp với Spotify Web API
* Tra cứu audio features chuẩn của bài hát trên Spotify qua [Chosic.com](https://www.chosic.com) hoặc [SongData.io](https://songdata.io).
* Đưa cùng file mp3 đó qua API `POST /analyze`.
* Đánh giá: Sai số BPM $\le \pm 2$, độ lệch Energy / Valence $\le 0.15$ chứng minh thuật toán đạt độ tin cậy tương đương chuẩn công nghiệp.

### Phương pháp 3: Kiểm tra BPM và Tone bằng phần mềm DJ chuyên dụng
* Kéo file vào **VirtualDJ** hoặc **Rekordbox**.
* So sánh chỉ số BPM và Key hiển thị trên phần mềm với kết quả JSON của service.

---

## 🚀 6. Hướng dẫn khởi chạy

### Chạy bằng Docker (Khuyên dùng - Full tính năng C++ DSP)
```bash
# Từ thư mục gốc của dự án:
docker compose up -d audio-analysis-service
```
Truy cập Swagger UI kiểm thử: **`http://localhost:8002/docs`**

### Chạy cục bộ bằng Python trên Windows (Fallback mode)
```bash
cd Moodify-Audio-Service
run.bat
```
*(Lưu ý: Môi trường Windows thuần không có trình biên dịch C++ Essentia sẽ tự động chuyển sang cơ chế Fallback Mode).*
