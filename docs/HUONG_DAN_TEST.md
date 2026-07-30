# Hướng dẫn test OCR Service

Tài liệu này để người không cần biết code vẫn tự test được.

---

## Thông tin server

| | |
|---|---|
| Địa chỉ | `http://172.16.1.22:8000` |
| Yêu cầu | Đã kết nối **VPN công ty** |
| Trạng thái | Đang test, chưa chính thức |

Kiểm tra kết nối trước — mở link này trên trình duyệt:

```
http://172.16.1.22:8000/health
```

Thấy dòng chữ có `"status": "ok"` là vào được. Không thấy gì → kiểm tra lại VPN.

---

## Cách 1 — Test bằng trình duyệt (dễ nhất, không cần cài gì)

**Bước 1.** Mở trình duyệt, vào:

```
http://172.16.1.22:8000/docs
```

**Bước 2.** Bấm vào dòng **`POST /ocr`** cho nó mở ra.

**Bước 3.** Bấm nút **`Try it out`** (góc phải).

**Bước 4.** Ở ô `file`, bấm **Choose File** và chọn ảnh hoặc PDF cần đọc.

**Bước 5.** Ở ô `orientation`, chọn:
- **`auto`** — ảnh chụp ngoài công trường, biển báo chụp chéo, chữ nghiêng
- **`upright`** — bản scan phẳng, chụp thẳng (nhanh hơn một chút)
- Để trống cũng được, hệ thống tự dùng `auto`

**Bước 6.** Bấm nút xanh **`Execute`**.

Kết quả hiện ngay bên dưới, ở ô **Response body**.

---

## Cách 2 — Gửi file bằng dòng lệnh

```bash
curl -F "file=@duong/dan/toi/anh.jpg" http://172.16.1.22:8000/ocr
```

Ảnh chụp chéo:

```bash
curl -F "file=@anh.jpg" "http://172.16.1.22:8000/ocr?orientation=auto"
```

---

## Đọc kết quả

Kết quả trả về có dạng:

```json
{
  "pages": [
    {
      "texts":  ["緊急出口", "EMERGENCY EXIT", "GATE 3"],
      "scores": [0.991, 0.964, 0.887],
      "boxes":  [ ... ]
    }
  ]
}
```

- **`texts`** — chữ đọc được, mỗi dòng một phần tử
- **`scores`** — **độ tin cậy** của từng dòng, từ 0 đến 1
- **`boxes`** — vị trí của dòng chữ đó trên ảnh (toạ độ 4 góc)
- **`pages`** — mỗi trang PDF là một phần tử. Ảnh thường chỉ có 1

Về `scores`:

| Giá trị | Ý nghĩa |
|---|---|
| trên 0.90 | Đọc chắc chắn |
| 0.80 – 0.90 | Nhiều khả năng đúng, nên liếc lại |
| dưới 0.80 | Đáng nghi, cần người xác nhận |

⚠️ Điểm cao **không đảm bảo đúng**. Máy có thể tự tin mà vẫn đọc sai — nhất là chữ Hán nét nhỏ. Luôn đối chiếu với ảnh gốc.

---

## Nên test file gì

Để đánh giá đúng năng lực, nên thử đủ các loại sau:

1. **Biển báo, bảng tên ngoài công trường** — chụp bằng điện thoại, kể cả chụp chéo
2. **Nhãn thiết bị** — số serial, mã tài sản, chữ nhỏ
3. **Bản scan giấy phép / permit** — có ô kẻ, có mã số, có ngày tháng
4. **Hợp đồng / báo giá có bảng**
5. **Tài liệu tiếng Trung phồn thể** — đây là ngôn ngữ chính cần đánh giá
6. **Tài liệu lẫn tiếng Trung + tiếng Anh trên cùng một dòng**
7. **Bản photocopy mờ, chụp thiếu sáng** — trường hợp xấu nhất

Càng giống tài liệu dùng thật hàng ngày càng tốt.

---

## Cái gì là bình thường, cái gì là lỗi

**Bình thường:**
- Đọc sai vài ký tự trên ảnh mờ hoặc chữ rất nhỏ
- Không đọc được chữ viết tay
- Trả về các dòng rời rạc, **không giữ được hình dạng bảng** — đây là giới hạn đã biết của bản hiện tại
- Ảnh trắng trả về kết quả rỗng

**Là lỗi, cần báo lại:**
- Trả về chữ **không hề có** trong ảnh
- **Bỏ sót nguyên một dòng** chữ rõ ràng, dễ đọc
- Báo lỗi đỏ `500`
- Chờ quá lâu không thấy trả kết quả

---

## Giới hạn hiện tại

| Loại file | Có đọc được không |
|---|---|
| Ảnh JPG, PNG | ✅ |
| PDF bản scan, **tối đa 10 trang** | ✅ |
| PDF trên 10 trang | ❌ báo lỗi `413` |
| File trên 50 MB | ❌ báo lỗi `413` |
| **Word (.docx), Excel (.xlsx)** | ❌ **không đọc được** |

**Vì sao không đọc được Word/Excel:** OCR đọc **hình ảnh**. File Word/Excel không phải ảnh, chữ trong đó đã là dữ liệu số sẵn — muốn lấy text thì đọc thẳng bằng công cụ khác, chính xác 100% và không cần OCR.

Nếu file Word/Excel có **ảnh scan chèn bên trong**: chuyển sang PDF trước rồi mới đưa vào đây.

**Bảng biểu:** bản hiện tại trả về các ô rời rạc, chưa ghép lại thành bảng. Tính năng giữ nguyên cấu trúc bảng đang được đánh giá riêng.

---

## Khi gặp lỗi, gửi lại giúp

1. **File đã test** (hoặc mô tả loại tài liệu nếu không gửi được)
2. **Kết quả nhận được** — copy phần Response body
3. **Kết quả đúng phải là gì** — chữ thật trên tài liệu

Phần 3 là quan trọng nhất. Không có nó thì không đo được sai bao nhiêu.
