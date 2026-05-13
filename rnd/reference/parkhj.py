import cv2
import numpy as np
import pyrealsense2 as rs
import easyocr
import requests
import time
import threading
from PIL import Image, ImageDraw, ImageFont
from queue import Queue, Empty

# ==============================
# DeepL API KEY (반드시 수정!)
# ==============================
DEEPL_API_KEY = "1e8f1e6d-24b6-498f-9cc4-0d0ad9f8110d:fx"

# ==============================
# 설정값
# ==============================
WHITE_LOWER = (0, 0, 180)
WHITE_UPPER = (180, 50, 255)

MIN_AREA_RATIO = 0.01
MAX_AREA_RATIO = 0.5

MIN_WHITE_RATIO = 0.6
MIN_GRAD_MEAN = 10.0

MIN_ASPECT = 0.6
MAX_ASPECT = 1.6

OCR_INTERVAL = 0.5
ALPHA = 0.6

TRACK_TIMEOUT = 0.5  # A4 detection 유지 시간

# ==============================
# 번역 쓰레드 Queue
# ==============================
translate_queue = Queue()
latest_translation = "(번역 대기중)"

# ==============================
# DeepL 번역 함수
# ==============================
def translate_to_korean_deepl(text_eng: str) -> str:
    if not text_eng.strip():
        return "(텍스트 없음)"

    url = "https://api-free.deepl.com/v2/translate"
    headers = {
        "Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"
    }
    data = {
        "text": text_eng,
        "source_lang": "EN",
        "target_lang": "KO"
    }

    try:
        res = requests.post(url, headers=headers, data=data, timeout=2.0)
        res.raise_for_status()
        j = res.json()
        trans = j.get("translations", [])
        if not trans:
            return "(번역 오류)"
        return trans[0].get("text", "(번역 오류)")
    except:
        return "(번역 오류)"

# ==============================
# 비동기 번역 쓰레드
# ==============================
def translate_thread():
    global latest_translation

    while True:
        try:
            text_eng = translate_queue.get(timeout=1)
        except Empty:
            continue

        latest_translation = translate_to_korean_deepl(text_eng)

# 번역 스레드 시작
threading.Thread(target=translate_thread, daemon=True).start()

# ==============================
# A4 후보 판단 함수
# ==============================
def is_a4_candidate(quad, frame_shape):
    h, w = frame_shape[:2]
    frame_area = w * h

    area = cv2.contourArea(quad)
    if area < MIN_AREA_RATIO * frame_area or area > MAX_AREA_RATIO * frame_area:
        return False

    x, y, bw, bh = cv2.boundingRect(quad)
    if bw == 0 or bh == 0:
        return False

    aspect = bw / float(bh)
    if not (MIN_ASPECT < aspect < MAX_ASPECT):
        return False

    return True

def check_brightness_hist(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total = np.sum(hist)
    white = np.sum(hist[200:256])
    if total == 0:
        return False, 0
    return (white / total) > MIN_WHITE_RATIO, white / total

def check_text_gradient(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    grad_mean = float(np.mean(np.abs(gx) + np.abs(gy)))
    return grad_mean > MIN_GRAD_MEAN, grad_mean

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_warp(image, quad):
    pts = quad.reshape(4,2).astype("float32")
    rect = order_points(pts)

    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB) * 1.5)

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB) * 1.5)

    maxW = max(maxW, 40)
    maxH = max(maxH, 40)

    dst = np.array([
        [0,0], [maxW-1,0],
        [maxW-1,maxH-1], [0,maxH-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))

    return warped, M, rect, dst

def create_text_image(text, size):
    w, h = size
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 35)
    except:
        font = ImageFont.load_default()

    words = text.split(" ")
    lines = []
    curr = ""
    max_width = w - 40

    for word in words:
        tmp = curr + word + " "
        if draw.textlength(tmp, font=font) < max_width:
            curr = tmp
        else:
            lines.append(curr)
            curr = word + " "
    if curr:
        lines.append(curr)

    y = 30
    for line in lines[:10]:
        draw.text((20, y), line, font=font, fill="black")
        y += 45

    return np.array(img)

# ==============================
# 메인
# ==============================
def main():
    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'])

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    print("RealSense started.")

    last_center = None
    last_quad = None
    last_overlay = None
    last_mask = None
    last_seen_time = 0
    last_ocr_time = 0

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        output = frame.copy()
        h, w = frame.shape[:2]
        now = time.time()

        # ========== 1) 흰색 필터 ==========
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_white = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

        # ========== 2) A4 후보 ==========
        contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if len(approx) != 4:
                continue
            if not cv2.isContourConvex(approx):
                continue
            if not is_a4_candidate(approx, frame.shape):
                continue

            x,y,bw,bh = cv2.boundingRect(approx)
            roi = frame[y:y+bh, x:x+bw]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            ok_b,_ = check_brightness_hist(gray)
            ok_g,_ = check_text_gradient(gray)

            if not(ok_b and ok_g):
                continue

            M = cv2.moments(approx)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
            else:
                cx = x + bw//2
                cy = y + bh//2

            candidates.append({
                "quad": approx,
                "center": (cx,cy),
                "area": cv2.contourArea(approx)
            })

        # ========== 3) 선택 + Tracking 유지 ==========
        best = None
        if candidates:
            if last_center is None:
                best = max(candidates, key=lambda c:c["area"])
            else:
                lx,ly = last_center
                best = min(
                    candidates,
                    key=lambda c:(c["center"][0]-lx)**2 + (c["center"][1]-ly)**2
                )
            last_center = best["center"]
            last_quad = best["quad"]
            last_seen_time = now
            box_color = (0,255,0)
        else:
            if now - last_seen_time <= TRACK_TIMEOUT:
                best = {"quad": last_quad}
                box_color = (0,255,255)
            else:
                last_quad = None
                last_overlay = None
                last_mask = None

        # ========== 4) OCR + 번역 ==========
        if last_quad is not None and (now - last_ocr_time) > OCR_INTERVAL:
            warped, M, src, dst = four_point_warp(frame, last_quad)

            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3,3), 0)
            gray = cv2.convertScaleAbs(gray, alpha=1.8)
            ocr_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            results = reader.readtext(ocr_img, detail=1)
            text_eng = " ".join([r[1] for r in results]).strip()

            translate_queue.put(text_eng)

            wh = (warped.shape[1], warped.shape[0])
            text_img = create_text_image(latest_translation, wh)

            Minv = cv2.getPerspectiveTransform(dst, src)
            overlay = cv2.warpPerspective(text_img, Minv, (w, h))

            mask = np.zeros((h,w), np.uint8)
            cv2.fillConvexPoly(mask, last_quad.reshape(4,2), 255)

            last_overlay = overlay
            last_mask = mask

            last_ocr_time = now

        # ========== 5) overlay 블렌딩 ==========
        if last_overlay is not None and last_mask is not None:
            mask_f = (last_mask.astype(np.float32)/255.0)[...,None]
            out_f = output.astype(np.float32)
            ov_f = last_overlay.astype(np.float32)

            output = out_f*(1-mask_f*ALPHA) + ov_f*(mask_f*ALPHA)
            output = output.astype(np.uint8)

        # ========== A4 outline ==========
        if last_quad is not None:
            cv2.drawContours(output, [last_quad], -1, box_color, 3)

        cv2.imshow("RealSense AR Translate", output)
        cv2.imshow("White Mask", mask_white)

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
