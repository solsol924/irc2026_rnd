import cv2
import numpy as np
import pyrealsense2 as rs
import easyocr
import requests
import time

# ==============================
# ① DeepL API 키 설정
# ==============================
# 여기 문자열을 네가 발급받은 키로 교체하세요.
DEEPL_API_KEY = "1e8f1e6d-24b6-498f-9cc4-0d0ad9f8110d:fx"


# ==============================
# ② DeepL 번역 함수
# ==============================
def translate_to_korean_deepl(text_eng: str) -> str:
    """
    영어 문자열을 DeepL API를 이용해 한국어로 번역.
    실패 시 '(번역 오류)' 반환.
    """
    if not text_eng.strip():
        return "(텍스트 없음)"

    url = "https://api-free.deepl.com/v2/translate"
    headers = {
        "Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"
    }
    data = {
        "text": text_eng,
        "source_lang": "EN",  # 입력 언어
        "target_lang": "KO"   # 출력 언어
    }

    try:
        res = requests.post(url, headers=headers, data=data, timeout=2.0)
        res.raise_for_status()
        j = res.json()
        # DeepL 응답 형식: {'translations': [{'detected_source_language':'EN', 'text':'번역결과'}]}
        translations = j.get("translations", [])
        if not translations:
            return "(번역 오류)"
        return translations[0].get("text", "(번역 오류)")
    except Exception as e:
        print("[DeepL Error]", e)
        return "(번역 오류)"


# ==============================
# ③ RealSense + OCR + DeepL 테스트
# ==============================
def main():
    print("Initializing EasyOCR (english)...")
    reader = easyocr.Reader(['en'])

    print("Starting RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    pipeline.start(config)

    last_ocr_time = 0.0
    OCR_INTERVAL = 1.0  # 1초마다 OCR + 번역

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            output = frame.copy()

            now = time.time()
            if now - last_ocr_time > OCR_INTERVAL:
                last_ocr_time = now

                # ---------- OCR ----------
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                results = reader.readtext(gray, detail=1)

                text_eng = " ".join([r[1] for r in results]).strip()

                # ---------- DeepL 번역 ----------
                text_kor = translate_to_korean_deepl(text_eng)

                # 터미널 출력
                print("====================================")
                print("영문 인식:", text_eng)
                print("한글 번역:", text_kor)
                print("====================================")

                # 화면에 간단하게 표시 (앞부분만)
                cv2.putText(output, f"EN: {text_eng[:40]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(output, f"KO: {text_kor[:40]}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("OCR + DeepL Translate Test", output)
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q')]:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
