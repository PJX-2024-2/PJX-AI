import cv2
import numpy as np
import os
import sys
import time
import mimetypes
import uuid
import json
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

def validate_image_path(image_path):
    # 이미지 경로 및 형식 검증
    if not os.path.isfile(image_path):
        logger.error(f"File does not exist: {image_path}")
        return False
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type not in ['image/jpeg', 'image/png', 'image/jpg']:
        logger.error(f"Unsupported image format: {image_path} (MIME type: {mime_type})")
        return False
    return True

def preprocess_receipt_image(image_path):
    """
    이미지 전처리
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image from: {image_path}")
        return None

    # 원본 이미지 복사
    original = image.copy()

    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 모폴리지 닫기 연산으로 노이즈 제거 및 그림자 완화
    img_height, img_width = gray.shape
    kernel_size = max(5, int(img_width * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 그림자 제거를 위한 배경 추정 및 차감
    background = cv2.GaussianBlur(closed, (15, 15), 0)
    diff = cv2.absdiff(gray, background)

    # 정규화하여 명암 대비 향상
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # 노이즈 제거
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)

    # 이진화 (Otsu Thresholding)
    _, thresh = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 팽창을 통해 텍스트 선명화
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.dilate(thresh, kernel_dilate, iterations=1)  

    return processed

def perform_ocr(api_url, secret_key, image_info, session, timeout=20):
    """
    OCR API를 호출하여 이미지에서 텍스트를 추출
    """
    image_type, image_path = image_info
    if not validate_image_path(image_path):
        return image_type, ""

    try:
        # 이미지 전처리
        processed_image = preprocess_receipt_image(image_path)
        if processed_image is None:
            logger.error(f"Preprocessing failed for: {image_path}")
            return image_type, ""

        # 전처리된 이미지를 메모리 버퍼로 인코딩
        success, encoded_image = cv2.imencode('.png', processed_image)
        if not success:
            logger.error(f"Image encoding failed for: {image_path}")
            return image_type, ""

        # 인코딩된 이미지를 바이트로 변환
        image_bytes = encoded_image.tobytes()

        # MIME 타입 설정
        mime_type, _ = mimetypes.guess_type(image_path)
        format_type = mime_type.split('/')[-1] if mime_type else 'png'
        unique_name = f"{image_type}_{uuid.uuid4()}.{format_type}"

        # OCR API 요청 JSON 설정
        request_json = {
            'images': [{'format': format_type, 'name': unique_name}],
            'requestId': str(uuid.uuid4()),
            'version': 'V2',
            'timestamp': int(round(time.time() * 1000))
        }
        payload = {'message': json.dumps(request_json).encode('UTF-8')}
        files = {'file': (unique_name, image_bytes, mime_type if mime_type else 'image/png')}
        headers = {'X-OCR-SECRET': secret_key}

        # OCR API 요청
        response = session.post(api_url, headers=headers, data=payload, files=files, timeout=timeout)

        if response.status_code == 200:
            ocr_result = response.json()
            text_results = " ".join([
                field['inferText']
                for image in ocr_result.get('images', [])
                for field in image.get('fields', [])
            ])
            logger.info(f"OCR succeeded for: {image_type}")
            return image_type, text_results
        else:
            logger.error(f"OCR request failed for: {image_type}, Status Code: {response.status_code}, Response: {response.text}")
            return image_type, ""
    except requests.exceptions.Timeout:
        logger.error(f"OCR request timeout for: {image_type}")
    except requests.exceptions.RequestException as e:
        logger.error(f"OCR request exception for: {image_type}, Exception: {e}")
    except Exception as e:
        logger.error(f"Unexpected exception during OCR processing for: {image_type}, Exception: {e}")
    return image_type, ""

def perform_summarization(api_url, api_key, prompt, session, timeout=25):
    """
    OpenAI API를 호출하여 텍스트 요약을 수행
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {'role': 'system', 'content': 'You are a helper that organizes and summarizes data.'},
            {'role': 'user', 'content': prompt}
        ],
        'max_tokens': 1000,
        'temperature': 0.2,
    }

    try:
        response = session.post(api_url, headers=headers, json=data, timeout=timeout)
        if response.status_code == 200:
            result = response.json()
            output = result['choices'][0]['message']['content']
            logger.info("Summarization succeeded.")
            return output.strip()
        else:
            logger.error(f"Summarization request failed. Status Code: {response.status_code}, Response: {response.text}")
            return None
    except requests.exceptions.Timeout:
        logger.error("Summarization request timeout.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Summarization request exception: {e}")
    except Exception as e:
        logger.error(f"Unexpected exception during summarization: {e}")
    return None

def main(*image_paths):
    """
    OCR 및 요약 작업 수행
    """
    # OCR API 정보
    ocr_api_url = os.getenv('OCR_API_URL')
    if not ocr_api_url:
        logger.error("Environment variable OCR_API_URL is not set.")
        sys.exit(1)
    secret_key = os.getenv('SECRET_KEY')

    if not secret_key:
        logger.error("Environment variable SECRET_KEY is not set.")
        sys.exit(1)

    # OpenAI API 정보
    openai_api_url = os.getenv('OPENAI_API_URL')
    if not openai_api_url:
        logger.error("Environment variable OPENAI_API_URL is not set.")
        sys.exit(1)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("Environment variable OPENAI_API_KEY is not set.")
        sys.exit(1)

    # 이미지 경로들 검증
    valid_image_paths = []
    for i, path in enumerate(image_paths):
        image_type = f'Receipt{i+1}'
        if validate_image_path(path):
            valid_image_paths.append((image_type, path))
        else:
            logger.warning(f"Invalid image path: {path}. Skipping this file.")

    if not valid_image_paths:
        logger.error("No valid image files provided.")
        sys.exit(1)

    # 세션 설정
    session = requests.Session()

    # OCR 수행
    extracted_texts = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_image = {
            executor.submit(perform_ocr, ocr_api_url, secret_key, image_info, session): image_info
            for image_info in valid_image_paths
        }
        for future in as_completed(future_to_image):
            image_type, text = future.result()
            extracted_texts[image_type] = text

    # OCR 결과 결합
    combined_text = "\n".join(extracted_texts.values())
    if not combined_text.strip():
        logger.error("OCR result is empty.")
        sys.exit(1)

    # 프롬프트 설정
    prompt = f"""
    다음은 상품 목록과 관련된 데이터로, '[상품명] [단가] [수량] [금액]'으로 구성되어 있습니다.
    각 항목별로 번호를 매기고, **상품명과 총액(단가 * 수량)**만 아래와 같은 형식으로 정리해주세요:

    0: [상품명] $[총액]
    1: [상품명] $[총액]
    ...

    입력 데이터:
    {combined_text}
    """

    # 요약 요청
    summary_result = perform_summarization(openai_api_url, openai_api_key, prompt, session)

    # 결과 출력
    if summary_result:
        print(summary_result)
    else:
        logger.error("Failed to retrieve summary result.")
        sys.exit(1)

if __name__ == '__main__':
    image_file_paths = sys.argv[1:]
    num_images = len(image_file_paths)
    logger.debug(f"Received {num_images} image(s): {image_file_paths}")
    if not (1 <= num_images <= 3):
        logger.error("You must provide at least 1 and at most 3 image files.")
        sys.exit(1)

    main(*image_file_paths)
