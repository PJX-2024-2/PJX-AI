import requests
import uuid
import json
import os
import sys
import time
import mimetypes
import logging
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

    #이미지 파일의 존재 여부와 지원되는 형식을 검증.
    if not os.path.isfile(image_path):
        logger.error(f"파일이 존재하지 않습니다: {image_path}")
        return False
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type not in ['image/jpeg', 'image/png', 'image/jpg']:
        logger.error(f"지원되지 않는 이미지 형식입니다: {image_path} (MIME type: {mime_type})")
        return False
    return True

def perform_ocr(api_url, secret_key, image_info, session, timeout=20):
    
    # Naver OCR API를 사용하여 이미지에서 텍스트를 추출.
    image_type, image_path = image_info
    if not validate_image_path(image_path):
        return image_type, ""

    try:
        with open(image_path, 'rb') as image_file:
            mime_type, _ = mimetypes.guess_type(image_path)
            format_type = mime_type.split('/')[-1] if mime_type else 'jpg'
            unique_name = f"{image_type}_{uuid.uuid4()}.{format_type}"
            
            request_json = {
                'images': [{'format': format_type, 'name': unique_name}],
                'requestId': str(uuid.uuid4()),
                'version': 'V2',
                'timestamp': int(round(time.time() * 1000))  
            }
            payload = {'message': json.dumps(request_json).encode('UTF-8')}
            files = {'file': (unique_name, image_file, mime_type)}
            headers = {'X-OCR-SECRET': secret_key}

            response = session.post(api_url, headers=headers, data=payload, files=files, timeout=timeout)

            if response.status_code == 200:
                ocr_result = response.json()
                text_results = " ".join([
                    field['inferText'] 
                    for image in ocr_result.get('images', []) 
                    for field in image.get('fields', [])
                ])
                logger.info(f"OCR 성공: {image_type}에서 텍스트 추출 완료.")
                return image_type, text_results
            else:
                logger.error(f"OCR 요청 실패: {image_type}, 상태 코드 {response.status_code}, 응답 내용: {response.text}")
                return image_type, ""
    except requests.exceptions.Timeout:
        logger.error(f"OCR 요청 타임아웃: {image_type}")
    except requests.exceptions.RequestException as e:
        logger.error(f"OCR 요청 중 예외 발생: {image_type}, 예외: {e}")
    except Exception as e:
        logger.error(f"OCR 처리 중 예외 발생: {image_type}, 예외: {e}")
    return image_type, ""

def perform_summarization(api_url, api_key, prompt, session, timeout=25):

    # OpenAI API를 사용하여 텍스트를 요약.
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {'role': 'system', 'content': '당신은 데이터를 정리하고 요약하는 도우미입니다.'}, 
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
            logger.info("OpenAI 요약 성공.")
            return output.strip()
        else:
            logger.error(f"OpenAI API 요청 실패: 상태 코드 {response.status_code}, 응답 내용: {response.text}")
            return None
    except requests.exceptions.Timeout:
        logger.error("OpenAI API 요청 타임아웃.")
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenAI API 요청 중 예외 발생: {e}")
    except Exception as e:
        logger.error(f"OpenAI 요약 처리 중 예외 발생: {e}")
    return None

def main(*image_paths):
    
    # 메인 함수로, 이미지 파일에서 텍스트를 추출하고 요약을 수행.
    # OCR API 정보
    ocr_api_url = os.getenv('OCR_API_URL')
    if not ocr_api_url:
        logger.error("환경 변수 OCR_API_URL가 설정되지 않았습니다.")
        sys.exit(1)
    secret_key = os.getenv('SECRET_KEY')

    if not secret_key:
        logger.error("환경 변수 SECRET_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    # OpenAI API 정보
    openai_api_url = os.getenv('OPENAI_API_URL')
    if not openai_api_url:
        logger.error("환경 변수 OPENAI_API_URL가 설정되지 않았습니다.")
        sys.exit(1)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("환경 변수 OPENAI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    # 이미지 경로들 검증
    valid_image_paths = []
    for i, path in enumerate(image_paths):
        image_type = f'영수증{i+1}'
        if validate_image_path(path):
            valid_image_paths.append((image_type, path))
        else:
            logger.warning(f"유효하지 않은 이미지 경로: {path}. 이 이미지는 건너뜁니다.")

    if not valid_image_paths:
        logger.error("유효한 이미지 파일이 없습니다.")
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
        logger.error("OCR 결과가 비어 있습니다.")
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
        logger.error("요약 결과를 얻지 못했습니다.")
        sys.exit(1)

if __name__ == '__main__':
    if not (1 <= len(sys.argv) <= 3):
        logger.error("사용법: python gpt.py [이미지 파일 경로 1] [이미지 파일 경로 2] ... (최소 1개, 최대 3개)")
        sys.exit(1)

    # 명령줄 인자로 전달된 파일 경로
    image_file_paths = sys.argv[1:]
    main(*image_file_paths)
