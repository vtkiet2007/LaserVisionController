# LaserVisionController

Hệ thống điều khiển chuyển động XY kết hợp thị giác máy tính (OpenCV, PySide6, NumPy, PySerial) và bộ điều khiển **MKS DLC32** (GRBL / SimpleLaser, 115200 baud) cho vùng làm việc hiệu chuẩn **210 × 120 mm** (phục vụ định vị phôi nhựa vàng baseline và mảnh gỗ thừa ngành Kiến trúc tại VGU).

## Quy ước tọa độ & Biến đổi 2 tầng

Hệ thống tách bạch vùng nhìn camera với hành trình thực tế của máy qua chuỗi biến đổi 2 tầng:

`Pixel (u, v) ─[Tầng 1: Homography 4 góc]─> Vision mm (x, y) ─[Tầng 2: Ma trận hiệu chuẩn corr_M]─> Machine mm (X, Y)`

- **Thứ tự click 4 góc vùng làm việc (Tầng 1):**
  1. Top-Left `(0, 120)`
  2. Top-Right `(210, 120)`
  3. Bottom-Right `(210, 0)`
  4. Bottom-Left `(0, 0)`
- **Nhân ma trận đồng nhất (Homogeneous):** Nhân ma trận $3 \times 3$ với vectơ $[x, y, 1]^T$ và chia cho thành phần thứ 3 trước khi kiểm tra giới hạn hành trình máy (`strict_bounds=True`). Điểm ngoài biên hợp lệ sẽ bị từ chối thay vì ép biên.
- **Trạng thái phôi:** `DETECTED` → `STABLE` → `STALE` → `LOST`.
- **An toàn MVP:** Cờ `ALLOW_LASER_FIRING = False` khóa phát tia laser ở giai đoạn thử nghiệm; luôn tách biệt dừng mềm trên giao diện với mạch ngắt khẩn cấp vật lý.

## Cài đặt & Chạy ứng dụng

```bash
git clone [https://github.com/](https://github.com/)<username>/LaserVisionController.git
cd LaserVisionController
pip install -r requirements.txt
python main.py