import requests
import uuid
import json
import os
import sys
import time

from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

def perform_ocr(api_url, secret_key, image_paths):
    extracted_texts = []

    for image_type, image_path in image_paths.items():
        request_json = {
            'images': [{'format': 'jpg', 'name': 'demo'}],
            'requestId': str(uuid.uuid4()),
            'version': 'V2',
            'timestamp': int(round(time.time() * 1000))  
        }
        payload = {'message': json.dumps(request_json).encode('UTF-8')}

        # 이미지 파일 처리
        try:
            with open(image_path, 'rb') as image_file:
                files = [('file', image_file)]
                headers = {'X-OCR-SECRET': secret_key}

                # 요청 전송
                response = requests.post(api_url, headers=headers, data=payload, files=files)

                # 응답 처리
                if response.status_code == 200:
                    ocr_result = response.json()
                    text_results = " ".join([
                        field['inferText'] 
                        for image in ocr_result.get('images', []) 
                        for field in image.get('fields', [])
                    ])
                    extracted_texts.append(text_results)
                else:
                    error_msg = f"OCR 요청 실패: 상태 코드 {response.status_code}, 응답 내용: {response.text}"
                    print(error_msg, file=sys.stderr)
                    extracted_texts.append("")  # 실패 시 빈 문자열 추가
        except Exception as e:
            error_msg = f"OCR 처리 중 예외 발생: {e}"
            print(error_msg, file=sys.stderr)
            extracted_texts.append("")  # 예외 발생 시 빈 문자열 추가

    # 하나로 합쳐서 반환
    combined_text = "\n".join(extracted_texts)
    return combined_text

def main(*image_paths):
    # OCR API 정보
    api_url = 'https://732m3jqsb8.apigw.ntruss.com/custom/v1/34539/dade4b4c1391aface40becf596ef2f4535b638cbde1d7942ead71ef101eec156/general'
    secret_key = os.getenv('SECRET_KEY')

    if not secret_key:
        print("환경 변수 SECRET_KEY가 설정되지 않았습니다.", file=sys.stderr)
        sys.exit(1)

    # 이미지 경로들
    image_paths_dict = {f'영수증{i+1}': path for i, path in enumerate(image_paths)}

    # OCR 수행
    extracted_text = perform_ocr(api_url, secret_key, image_paths_dict)

    if not extracted_text.strip():
        print("OCR 결과가 비어 있습니다.", file=sys.stderr)
        sys.exit(1)

    # OpenAI API 키 가져오기
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("환경 변수 OPENAI_API_KEY가 설정되지 않았습니다.", file=sys.stderr)
        sys.exit(1)

    # 프롬프트 설정
    prompt = f"""
    다음은 상품 목록과 관련된 데이터로, '[상품명] [단가] [수량] [금액]'으로 구성되어 있습니다.
    각 항목별로 번호를 매기고, **상품명과 총액(단가 * 수량)**만 아래와 같은 형식으로 정리해주세요:

    0: [상품명] [총액]
    1: [상품명] [총액]
    ...

    입력 데이터:
    {extracted_text}
    """

    # 요약 요청
    summary_result = perform_summarization(openai_api_key, prompt)

    # 결과 출력
    if summary_result:
        print(summary_result)
    else:
        print("요약 결과를 얻지 못했습니다.", file=sys.stderr)
        sys.exit(1)

def perform_summarization(api_key, prompt):
    # OpenAI API 엔드포인트 설정
    api_url = 'https://api.openai.com/v1/chat/completions'

    # 요청 헤더 설정
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    # 요청 데이터 설정
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {'role': 'system', 'content': '당신은 데이터를 정리하는 도우미입니다.'},
            {'role': 'user', 'content': prompt}
        ],
        'max_tokens': 1000,
        'temperature': 0.2,
    }

    # 요청 전송 및 예외 처리
    try:
        response = requests.post(api_url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            output = result['choices'][0]['message']['content']
            return output.strip()
        else:
            error_msg = f"OpenAI API 요청 실패: 상태 코드 {response.status_code}, 응답 내용: {response.text}"
            print(error_msg, file=sys.stderr)
            return None
    except requests.exceptions.RequestException as e:
        error_msg = f"OpenAI API 요청 중 예외 발생: {e}"
        print(error_msg, file=sys.stderr)
        return None

if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("사용법: python gpt.py [이미지 파일 경로 1] [이미지 파일 경로 2] ... (최대 3개)", file=sys.stderr)
        sys.exit(1)

    # 명령줄 인자로 전달된 파일 경로
    image_file_paths = sys.argv[1:]
    main(*image_file_paths)
