import cv2
import numpy as np

# QRコード検出器を初期化
qrCodeDetector = cv2.QRCodeDetector()

cap = cv2.VideoCapture(0)


def update(old):
    @np.vectorize
    def sum_around(i, j):
        return old[max(0, i - 1) : i + 2, max(0, j - 1) : j + 2].sum() - old[i, j]

    around = np.fromfunction(sum_around, old.shape, dtype=int)
    new = np.where(old, ((2 <= around) & (around <= 3)), (around == 3))
    return new


cell = None
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # QRコードを検出し、その位置を取得
    decodedText, points, straight_code = qrCodeDetector.detectAndDecode(frame)

    if points is not None:
        points = points[0]

        # Validate and draw lines around QR code
        if len(points) == 4 and all(len(point) == 2 for point in points):
            count = 0

            if straight_code is not None:
                # frameにQRコードの内容を描画
                resized = cv2.resize(straight_code, (50, 50))
                binary = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY)[1]
                # binary = cv2.bitwise_not(binary)
                if cell is None:
                    cell = binary.astype(np.bool_)
                else:
                    cell = update(cell)
                src = np.array(
                    [
                        [0, 0],
                        [cell.shape[1], 0],
                        [cell.shape[1], cell.shape[0]],
                        [0, cell.shape[0]],
                    ],
                ).astype(np.float32)
                transform = cv2.getPerspectiveTransform(
                    src,
                    points,
                )
                image_rgba = cv2.cvtColor((1 - cell.astype(np.uint8)) * 255, cv2.COLOR_GRAY2RGBA)
                warped = cv2.warpPerspective(
                    image_rgba,
                    transform,
                    (frame.shape[1], frame.shape[0]),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=[255, 255, 255, 0],
                )
                mask = (warped[:, :, 3] == 0).astype(np.uint8)  # 透過マスク
                mask = cv2.merge([mask, mask, mask])
                frame_masked = frame * mask
                frame = frame_masked + warped[:, :, :3] * (1 - mask[:, :, :3])
        else:
            print("Invalid points structure.")
    else:
        if cell is not None:
            count += 1
            if count > 15:
                cell = None
                count = 0

    cv2.imshow("QR Code Scanner", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
