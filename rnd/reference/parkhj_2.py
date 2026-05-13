import cv2
import numpy as np
import pyrealsense2 as rs
import easyocr
import requests
import time
import threading
from PIL import Image, ImageDraw, ImageFont
from queue import Queue, Empty

# =====================================
# 1. DeepL
# =====================================
DEEPL_API_KEY = "1e8f1e6d-24b6-498f-9cc4-0d0ad9f8110d:fx"

CANNY_LOW  = 50
CANNY_HIGH = 150
MIN_AREA = 20000
MAX_AREA = 200000
OCR_INTERVAL = 0.5
ALPHA = 0.3

missing_frame_count = 0 
MAX_MISSING_FRAMES = 5

# 번역 요청/캐시
translate_queue = Queue()
translation_cache = {}      # { full_eng(str): kor(str) }
requested_texts  = set()    # 이미 요청 넣은 영어 문장들


# =====================================
# DeepL 번역 스레드
# =====================================
def translate(text_eng: str) -> str:
    if not text_eng.strip():
        return "(텍스트 없음)"

    url = "https://api-free.deepl.com/v2/translate"
    headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"}
    data = {"text": text_eng, "source_lang": "EN", "target_lang": "KO"}

    try:
        r = requests.post(url, headers=headers, data=data, timeout=2.0)
        r.raise_for_status()
        j = r.json()
        t = j.get("translations", [])
        if not t:
            return "(번역 오류)"
        return t[0].get("text", "(번역 오류)")
    except:
        return "(번역 오류)"


def translate_thread():
    global translation_cache
    while True:
        try:
            text_eng = translate_queue.get(timeout=1)
        except Empty:
            continue

        trans = translate(text_eng)
        translation_cache[text_eng] = trans


threading.Thread(target=translate_thread, daemon=True).start()


# =====================================
# 사각형 warp 함수
# =====================================
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_quad(image, quad):
    pts = quad.reshape(4, 2).astype("float32")
    rect = order_points(pts)

    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))
    maxW = max(maxW, 40)
    maxH = max(maxH, 40)

    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))

    return warped, M, rect, dst


# =====================================
# 🔥 문장 단위 클러스터링
# =====================================
def group_words(results):
    if len(results) == 0:
        return []

    words = []
    for r in results:
        bbox, text, _ = r
        pts = np.array(bbox)

        # bbox 상단 y 평균
        top_y = int((pts[0][1] + pts[1][1]) / 2)
        # 좌우 순서 안정화 → 최소 x
        min_x = int(np.min(pts[:,0]))

        words.append((min_x, top_y, bbox, text))

    # 줄 순으로 y 정렬
    words.sort(key=lambda x: x[1])

    clusters = []
    current = [words[0]]

    LINE_GAP = 20
    WORD_GAP = 130

    for i in range(1, len(words)):
        prev = current[-1]
        curr = words[i]

        if abs(prev[1] - curr[1]) < LINE_GAP:
            if abs(prev[0] - curr[0]) < WORD_GAP:
                current.append(curr)
            else:
                clusters.append(current)
                current = [curr]
        else:
            clusters.append(current)
            current = [curr]

    clusters.append(current)

    # 각 클러스터 내 x 정렬
    for c in clusters:
        c.sort(key=lambda x: x[0])

    return clusters


# =====================================
# overlay 생성 (warp 공간)
# =====================================
def create_overlay(h, w, word_states):
    """
    word_states: { key: {"eng":..., "kor":..., "bbox":[[x,y],...]} }
    """
    overlay = np.ones((h, w, 3), dtype=np.uint8) * 255
    pil_img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil_img)

    for key, st in word_states.items():
        kor = st["kor"]
        bbox = np.array(st["bbox"])

        # bbox 영역
        min_x = int(np.min(bbox[:,0]))
        max_x = int(np.max(bbox[:,0]))
        min_y = int(np.min(bbox[:,1]))
        max_y = int(np.max(bbox[:,1]))

        # OCR 글씨 높이
        height = max_y - min_y
        if height < 5:
            height = 5

        # 1) 번역 글씨 크기 (글씨 높이에 비례)
        font_size = int(height * 0.5)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                font_size
            )
        except:
            font = ImageFont.load_default()

        # 2) 번역 글씨 위치 (글씨 아래 + 높이 * 0.7)
        text_y = min(max_y + int(height * 0.5), h - height) 

        text = kor
        tb = draw.textbbox((0,0), text, font=font)
        tw = tb[2] - tb[0]

        text_x = min_x + (max_x - min_x)//2 - tw//2

        draw.text((text_x, text_y), text, fill=(0,0,0), font=font)

    return np.array(pil_img)

# main 함수 밖에 정의 (혹은 main 함수 안에 nested function으로 정의)
def select_and_translate_roi(frame, reader):
    """
    프레임에서 ROI를 선택하고, 해당 영역의 텍스트를 OCR 및 번역 후 팝업으로 표시
    """
    # 1. 원본 프레임 복사
    temp_frame = frame.copy()
    
    # 2. ROI 선택 창 표시
    win_name = "Select ROI for Translation"
    cv2.namedWindow(win_name)
    
    # 선택 영역 좌표 (x, y, w, h)
    ret = cv2.selectROI(win_name, temp_frame)
    cv2.destroyWindow(win_name) # ROI 선택 후 창 닫기
    
    x, y, w, h = ret
    
    if w > 0 and h > 0:
        # 3. ROI 영역 잘라내기
        roi_img = frame[int(y):int(y+h), int(x):int(x+w)]
        
        # 4. OCR 수행
        print("[ROI OCR] Running OCR on selected area...")
        roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        # EasyOCR 실행 (WARP 없이 바로 실행)
        results = reader.readtext(roi_gray, detail=1) 
        
        if not results:
            print("[ROI OCR] No text found in ROI.")
            return

        # 5. 텍스트 그룹화 및 번역 요청
        clusters = group_words(results) # 기존 함수 활용
        
        all_texts = []
        for cluster in clusters:
            full_eng = " ".join([word[3] for word in cluster]).strip()
            if full_eng:
                all_texts.append(full_eng)
                
        full_eng_text = "\n".join(all_texts)
        
        if not full_eng_text:
            print("[ROI OCR] No meaningful text to translate.")
            return

        # 6. 동기 번역 (DeepL 요청)
        print("[ROI OCR] Found: '{}'".format(full_eng_text.replace('\n', ' ')))
        kor_translation = translate(full_eng_text)
        
        # 7. 번역 결과 팝업 표시
        if kor_translation:
        
            # 7-1. 오버레이 합성 (원본 이미지 기반)
            # roi_img (크롭된 이미지)를 PIL 이미지로 변환
            pil_roi_img = Image.fromarray(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_roi_img)

            # 폰트 로드 (기존 AR 번역 로직과 동일)
            try:
                # 적절한 폰트 크기 설정 (예: 20)
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                    20 
                )
            except:
                font = ImageFont.load_default()

            # 번역된 텍스트를 ROI 이미지 위에 그립니다.
            # 텍스트가 여러 줄일 경우를 대비하여 줄 바꿈 처리
            lines = kor_translation.split('\n')
            y_offset = 10 # 상단 여백
            
            for line in lines:
                # 텍스트의 크기와 위치 계산
                tb = draw.textbbox((0, 0), line, font=font)
                tw = tb[2] - tb[0]
                
                # 텍스트 배경 박스 (반투명 검은색)
                text_bg_box = [(10, y_offset), (10 + tw + 10, y_offset + 30)] # 30은 대략적인 줄 높이
                draw.rectangle(text_bg_box, fill=(0, 0, 0, 180)) # RGBA로 반투명 검은색
                
                # 텍스트 그리기
                draw.text((15, y_offset), line, fill=(255, 255, 255), font=font)
                y_offset += 35 # 다음 줄 간격
                
            # PIL 이미지를 다시 OpenCV 형식으로 변환 (RGB->BGR)
            result_img_cv = cv2.cvtColor(np.array(pil_roi_img), cv2.COLOR_RGB2BGR)

            # 7-2. 결과 창 표시 및 대기
            cv2.namedWindow("ROI Translation Result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("ROI Translation Result", result_img_cv)
            
            # [핵심] 사용자가 닫을 때까지 대기 (스레드에서 실행되므로 메인 영상은 안 멈춤)
            cv2.waitKey(0) 
            
            # 창 닫기
            cv2.destroyWindow("ROI Translation Result")
            cv2.destroyWindow(win_name)

# =====================================
# 5. 메인 (여러 장 A4 지원)
# =====================================
def main():
    global missing_frame_count, MAX_MISSING_FRAMES
    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'])

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    print("RealSense started.")

    last_overlay = None
    last_mask = None
    last_ocr_time = 0.0

    while True:
        frames = pipeline.wait_for_frames()
        f = frames.get_color_frame()
        if not f:
            continue

        frame = np.asanyarray(f.get_data())
        output = frame.copy()
        h, w = frame.shape[:2]
        now = time.time()

        # ----- Canny -----
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g_blur = cv2.GaussianBlur(g, (5,5), 0)
        edges = cv2.Canny(g_blur, CANNY_LOW, CANNY_HIGH)
        kernel = np.ones((5,5), np.uint8)
        edges = cv2.dilate(edges, kernel, 1)
        edges = cv2.erode(edges, kernel, 1)

        # ----- contour → 여러 개 quad 후보 -----
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quads = []

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)

            if len(approx) == 4:
                if not cv2.isContourConvex(approx):
                    continue
                area = cv2.contourArea(approx)
                if area < MIN_AREA or area > MAX_AREA:
                    continue
                hull = cv2.convexHull(cnt) # 원래 컨투어로 컨벡스 헐 계산
                hull_area = cv2.contourArea(hull)
                if hull_area == 0: # 0으로 나누는 오류 방지
                    continue
                solidity = float(area) / hull_area
                if solidity < 0.8: # 80% 미만이면 제외
                    continue
                x,y,w2,h2 = cv2.boundingRect(approx)
                asp = w2/float(h2)
                if 0.4 < asp < 2.0:
                    quads.append(approx)

        # ----- mask (여러 종이 union) -----
        mask = None

        if len(quads) > 0:
            # [성공] 감지되었으므로 카운터 리셋
            missing_frame_count = 0 
            
            mask = np.zeros((h, w), np.uint8)
            for q in quads:
                cv2.fillConvexPoly(mask, q.reshape(4,2), 255)
        
        else:
            # [실패] 감지 안 됨 -> 카운터 증가
            missing_frame_count += 1
            
            # [판단] 유예 기간(5프레임)을 넘겼는가?
            if missing_frame_count > MAX_MISSING_FRAMES:
                last_mask = None
                last_overlay = None
            else:
                # 5프레임이 안 지났으면, mask는 없어도(None)
                # last_overlay와 last_mask를 지우지 않고 그대로 둡니다.
                pass

        # ----- OCR + 번역 (여러 종이) -----
        if mask is not None and (now - last_ocr_time) > OCR_INTERVAL:
            overlay_total = np.zeros_like(frame, dtype=np.uint8)
            all_texts = []

            for quad in quads:
                warped, M, src, dst = warp_quad(frame, quad)
                warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

                results = reader.readtext(warped_gray, detail=1)

                clusters = group_words(results)

                # 이 종이 한 장에 대한 상태
                local_states = {}
                state_idx = 0

                for cluster in clusters:
                    full_eng = " ".join([word[3] for word in cluster]).strip()
                    if not full_eng:
                        continue

                    all_texts.append(full_eng)

                    # 문장 bbox
                    pts_all = []
                    for word in cluster:
                        pts_all.extend(word[2])
                    pts_all = np.array(pts_all)

                    bbox_sentence = [
                        [int(np.min(pts_all[:,0])), int(np.min(pts_all[:,1]))],
                        [int(np.max(pts_all[:,0])), int(np.min(pts_all[:,1]))],
                        [int(np.max(pts_all[:,0])), int(np.max(pts_all[:,1]))],
                        [int(np.min(pts_all[:,0])), int(np.max(pts_all[:,1]))]
                    ]

                    # 번역 캐시 조회 / 요청
                    kor = translation_cache.get(full_eng)
                    if kor is None:
                        kor = "(번역 대기중)"
                        if full_eng not in requested_texts:
                            translate_queue.put(full_eng)
                            requested_texts.add(full_eng)

                    state = {
                        "eng": full_eng,
                        "kor": kor,
                        "bbox": bbox_sentence
                    }
                    local_states[state_idx] = state
                    state_idx += 1

                # overlay 생성 (warp 공간)
                if len(local_states) > 0:
                    warp_h, warp_w = warped.shape[:2]
                    warped_overlay = create_overlay(warp_h, warp_w, local_states)

                    # warp-back
                    Minv = cv2.getPerspectiveTransform(dst, src)
                    ov = cv2.warpPerspective(warped_overlay, Minv, (w, h))

                    # 여러 종이 overlay를 합성 (최댓값 사용)
                    overlay_total = np.maximum(overlay_total, ov)

            if all_texts:
                print("[OCR]", " | ".join(all_texts))
            else:
                print("[OCR] (no text)")

            last_overlay = overlay_total
            last_mask = mask
            last_ocr_time = now

        # ----- 반투명 합성 -----
        if last_overlay is not None and last_mask is not None:
            mask_f = (last_mask.astype(np.float32)/255.0)[...,None]
            out_f = output.astype(np.float32)
            ov_f = last_overlay.astype(np.float32)

            output = out_f*(1 - mask_f*ALPHA) + ov_f*(mask_f*ALPHA)
            output = output.astype(np.uint8)

        # 윤곽선 디버깅 (모든 종이)
        for q in quads:
            cv2.drawContours(output, [q], -1, (0,255,0), 3)

        cv2.imshow("RealSense AR Translate", output)
        cv2.imshow("Edges", edges)

        k = cv2.waitKey(1)&0xFF
        if k in [27, ord('q')]:
            break
        
        if k == ord('s'):
            # 메인 영상을 잠시 멈추고 현재 프레임 복사
            frame_to_process = frame.copy() 
            roi_thread = threading.Thread(
                target=select_and_translate_roi, 
                args=(frame_to_process, reader), 
                daemon=True # 메인 프로그램 종료 시 함께 종료되도록 설정
            )
            roi_thread.start()

    pipeline.stop()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
