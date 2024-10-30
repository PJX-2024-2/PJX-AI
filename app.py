import subprocess
import os
import logging

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
load_dotenv()

# 로깅 설정
logging.basicConfig(
    filename='error.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# CORS 설정: 두 URL을 허용하고, credentials 지원 추가
cors = CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "https://pjx-client-4bsx.vercel.app"]}}, supports_credentials=True)

# 모든 요청 전 요청 정보 로깅
@app.before_request
def log_request_info():
    logging.debug(f"Request Headers: {dict(request.headers)}")
    logging.debug(f"Request Body: {request.get_data()}")

# 모든 응답에 CORS 헤더 추가 및 응답 헤더 로깅
@app.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin')
    allowed_origins = ["http://localhost:5173", "https://pjx-client-4bsx.vercel.app"]
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization']
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS']
    else:
        response.headers['Access-Control-Allow-Origin'] = 'null'
    logging.debug(f"Response Headers: {dict(response.headers)}")
    return response

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status='healthy'), 200

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled exception: {e}", exc_info=True)
    response = jsonify({'error': 'An unexpected error occurred', 'details': str(e)})
    response.status_code = 500
    # 응답 헤더에 CORS 헤더 추가
    origin = request.headers.get('Origin')
    allowed_origins = ["http://localhost:5173", "https://pjx-client-4bsx.vercel.app"]
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization']
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS']
    logging.debug(f"Error Response Headers: {dict(response.headers)}")
    return response

@app.route('/api/v1/receipt/analyze', methods=['POST'])
def process_request():
    try:
        # 파일 유무 확인
        if 'files' not in request.files:
            logging.error('No files part in the request')
            return jsonify({'error': 'No files part'}), 400

        # 파일 리스트 확인
        files = request.files.getlist('files')
        if not files:
            logging.error('No selected files in the request')
            return jsonify({'error': 'No selected files'}), 400

        # 파일 개수 제한 확인
        if len(files) > 3:
            logging.error('More than 3 files uploaded')
            return jsonify({'error': 'Maximum 3 files allowed'}), 400

        image_paths = {}
        temp_dir = '/tmp'

        for idx, file in enumerate(files, start=1):
            if file.filename == '':
                logging.error('One of the files has no filename')
                return jsonify({'error': 'One of the files has no filename'}), 400
            file_path = os.path.join(temp_dir, f'image_{idx}_{file.filename}')
            file.save(file_path)
            image_type = f'영수증{idx}'
            image_paths[image_type] = file_path

        # openai.py 실행, 입력 이미지 경로 전달
        try:
            result = subprocess.check_output(['python3', 'openai.py'] + list(image_paths.values()), stderr=subprocess.STDOUT)
            result_data = result.decode('utf-8')
            return jsonify({'result': result_data}), 200
        except subprocess.CalledProcessError as e:
            logging.error(f"Subprocess error: {e.output.decode('utf-8')}")
            return jsonify({'error': 'Error processing images', 'details': e.output.decode('utf-8')}), 500
        except Exception as e:
            logging.error(f"Unexpected error in subprocess: {e}")
            return jsonify({'error': 'Unexpected error occurred during image processing', 'details': str(e)}), 500

    except Exception as err:
        logging.error(f"Unhandled exception: {err}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred', 'details': str(err)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
