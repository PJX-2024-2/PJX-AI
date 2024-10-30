import subprocess
import os
import logging

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
load_dotenv()

# 로깅 설정
logging.basicConfig(filename='error.log', level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# CORS 설정: 두 URL을 허용
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "https://pjx-client-4bsx.vercel.app"]}}, supports_credentials=True)

# 모든 응답에 CORS 헤더 추가
@app.after_request
def add_cors_headers(response):
    return response

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status='healthy'), 200

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
            return jsonify({'result': result.decode('utf-8')}), 200
        except subprocess.CalledProcessError as e:
            logging.error(f"Subprocess error: {e.output.decode('utf-8')}")
            return jsonify({'error': 'Error processing images', 'details': e.output.decode('utf-8')}), 500
        except Exception as e:
            logging.error(f"Unexpected error in subprocess: {e}")
            return jsonify({'error': 'Unexpected error occurred during image processing', 'details': str(e)}), 500

    except Exception as err:
        logging.error(f"Unhandled exception: {err}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred', 'details': str(err)}), 500

# 애플리케이션 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
