# tests/samples/hard — mẫu khó + chấm điểm tự động

```bash
python scripts/make_hard_samples.py                    # tạo lại (đã commit sẵn, không cần chạy)
python scripts/score_samples.py --url http://172.16.1.22:8000
python scripts/score_samples.py --orientation auto -v  # xem từng dòng
```

Vì `make_hard_samples.py` **biết chính xác chữ nó đã vẽ**, `ground_truth.json` được sinh
tự động — không cần ai gõ tay để có số accuracy.

## Ba con số, không phải một

| | Là gì | Vì sao cần tách |
|---|---|---|
| **char accuracy** | % ký tự đúng, **trên những dòng nó tìm thấy** | Cao mà recall thấp = "đọc đẹp nhưng bỏ sót" |
| **recall** | Tìm thấy bao nhiêu % số dòng có thật | Dòng không phát hiện được thì **không xuất hiện** trong char accuracy — đây là lỗi âm thầm |
| **invented** | Số dòng trả về mà không khớp gì thật | Bịa chữ |

Chỉ tốt khi **cả ba** đều tốt. Tối ưu một cái rất dễ và vô nghĩa.

## Mỗi file tấn công cái gì

| file | đánh vào |
|---|---|
| `20_permit_form.png` | form có ô kẻ, nhãn, mã số, ngày tháng, TC+EN lẫn nhau |
| `21_invoice_table.png` | **bảng có kẻ ô** — đúng cái fast lane không dựng lại được |
| `22_site_photo.jpg` | chụp chéo + ánh sáng lệch + nén JPEG + mờ |
| `23_nameplate.jpg` | chữ nhỏ dày, tương phản thấp trên nền kim loại |
| `24_two_column.png` | thứ tự đọc qua 2 cột |
| `25_low_contrast.png` | xám trên xám, bản in mờ |
| `26_noisy_scan.jpg` | bản photocopy xấu: nghiêng + nhiễu hạt + JPEG nặng |
| `27_complex.pdf` | 3 trang gộp các loại trên |

## Giới hạn — đọc kỹ chỗ này

Đây là tài liệu **render ra rồi làm xấu đi bằng code**. Nó không giả được: vân giấy,
rung tay khi chụp, mực lem, con dấu đè lên chữ, **chữ viết tay**.

Con số ở đây là **mốc chống hồi quy** — biết khi nào đổi model/config làm kết quả tệ đi.
**Không phải** câu trả lời cho câu hỏi kinh doanh. Cái đó vẫn cần tài liệu HK thật, chấm
bằng đúng script này:

```bash
# bỏ tài liệu thật vào tests/fixtures/ (đã gitignore), tự viết ground_truth.json cùng format
python scripts/score_samples.py --dir tests/fixtures
```
