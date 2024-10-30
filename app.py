import subprocess
import os

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
load_dotenv()

# CORS 설정: 두 URL을 허용
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "https://pjx-client-4bsx.vercel.app"]}}, supports_credentials=True)

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status='healthy'), 200

@app.route('/api/v1/receipt/analyze', methods=['POST'])
def process_request():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400
    
    # 파일 리스트 확인
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No selected files'}), 400
    # 3개 이상일 경우 에러 반환
    if len(files) > 3:
        return jsonify({'error': 'Maximum 3 files allowed'}), 400

    image_paths = {}
    temp_dir = '/tmp'
    
    for idx, file in enumerate(files, start=1):
        if file.filename == '':
            return jsonify({'error': 'One of the files has no filename'}), 400
        file_path = os.path.join(temp_dir, f'image_{idx}_{file.filename}')
        file.save(file_path)
        image_type = f'영수증{idx}'
        image_paths[image_type] = file_path

    # main.py 실행, 입력 이미지 경로 
    try:
        result = subprocess.check_output(['python3', 'openai.py'] + list(image_paths.values()), stderr=subprocess.STDOUT)
        return jsonify({'result': result.decode('utf-8')}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({'error': e.output.decode('utf-8')}), 500

# 0.0.0.0 IP 주소, 5000 포트
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
