import subprocess
import os
import logging
import json
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity, exceptions
import jwt  # JWT 디코딩을 위해 추가

app = Flask(__name__)
load_dotenv()

# JWT 설정
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['JWT_ALGORITHM'] = 'HS512'  # 추가된 부분: JWT 알고리즘 설정

if not app.config['JWT_SECRET_KEY']:
    raise ValueError("JWT_SECRET_KEY 환경 변수가 설정되지 않았습니다.")
app.logger.debug(f"JWT_SECRET_KEY loaded: {app.config['JWT_SECRET_KEY']}")
app.logger.debug(f"JWT_ALGORITHM set to: {app.config['JWT_ALGORITHM']}")  # 추가된 부분: 알고리즘 로그

jwt_manager = JWTManager(app)

# 로깅 설정
log_file = 'error.log'  # 로그 파일 경로
if not os.path.exists(log_file):
    open(log_file, 'a').close()

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
file_handler.setFormatter(formatter)

app.logger.addHandler(file_handler)
app.logger.setLevel(logging.DEBUG)

# CORS 설정: 두 URL을 허용
cors = CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",
            "https://pjx-client-4bsx.vercel.app"
        ]
    }
}, supports_credentials=True)

# 보호된 엔드포인트 리스트
protected_endpoints = [
    'process_request',
    'analyze_spending'
]

@app.before_request
def log_request_info():
    app.logger.debug(f"Request Path: {request.path}")
    app.logger.debug(f"Request Headers: {dict(request.headers)}")
    app.logger.debug(f"Request Body: {request.get_data()}")
    
    # 현재 요청된 엔드포인트 확인
    endpoint = request.endpoint
    if endpoint in protected_endpoints:
        # Authorization 헤더에서 JWT 토큰 추출
        auth_header = request.headers.get('Authorization', None)
        if auth_header:
            try:
                token_type, token = auth_header.split()
                if token_type.lower() != 'bearer':
                    app.logger.warning("Authorization header does not start with Bearer")
                app.logger.debug(f"Received JWT Token: {token}")

                # 토큰 디코딩 및 클레임 확인 (서명 검증 없이)
                try:
                    decoded_token = jwt.decode(token, options={"verify_signature": False})
                    app.logger.debug(f"Decoded JWT Token Payload: {decoded_token}")
                except jwt.DecodeError as e:
                    app.logger.error(f"Error decoding JWT Token: {e}")
            except ValueError:
                app.logger.warning("Authorization header is malformed")
        else:
            app.logger.debug("No Authorization header found for protected endpoint")
    else:
        app.logger.debug("Unprotected endpoint accessed; skipping JWT token processing")

# JWT 오류 핸들러
@jwt_manager.unauthorized_loader
def unauthorized_response(callback):
    app.logger.error(f"Unauthorized: {callback}")
    return jsonify({'error': 'Missing Authorization Header'}), 401

@jwt_manager.invalid_token_loader
def invalid_token_callback(reason):
    app.logger.error(f"Invalid JWT Token: {reason}")
    return jsonify({'error': 'Invalid JWT Token', 'details': reason}), 422

@jwt_manager.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    app.logger.warning(f"Expired JWT Token: {jwt_payload}")
    return jsonify({'error': 'Expired JWT Token'}), 401

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/health', methods=['GET'])
def health_check():
    response = jsonify(status='healthy')
    response.status_code = 200
    app.logger.debug(f"Health check response: {response.get_data(as_text=True)}")
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled exception: {e}", exc_info=True)
    response = jsonify({'error': 'An unexpected error occurred', 'details': str(e)})
    response.status_code = 500
    app.logger.debug(f"Error Response Headers: {dict(response.headers)}")
    return response

@app.route('/api/v1/receipt/analyze', methods=['POST'])
@jwt_required()
def process_request():
    try:
        # 파일 유무 확인
        if 'files' not in request.files:
            app.logger.error('No files part in the request')
            return jsonify({'error': 'No files part'}), 400

        # 파일 리스트 확인
        files = request.files.getlist('files')
        if not files:
            app.logger.error('No selected files in the request')
            return jsonify({'error': 'No selected files'}), 400

        # 파일 개수 제한 확인
        if len(files) > 3:
            app.logger.error('More than 3 files uploaded')
            return jsonify({'error': 'Maximum 3 files allowed'}), 400

        image_paths = {}
        temp_dir = '/tmp'

        for idx, file in enumerate(files, start=1):
            if file.filename == '':
                app.logger.error('One of the files has no filename')
                return jsonify({'error': 'One of the files has no filename'}), 400
            file_path = os.path.join(temp_dir, f'image_{idx}_{file.filename}')
            file.save(file_path)
            image_type = f'영수증{idx}'
            image_paths[image_type] = file_path

        # receipt_ocr.py 실행, 입력 이미지 경로 전달
        try:
            result = subprocess.check_output(['python3', 'receipt_ocr.py'] + list(image_paths.values()), stderr=subprocess.STDOUT)
            result_data = result.decode('utf-8')
            app.logger.debug(f"Subprocess result: {result_data}")
            return jsonify({'result': result_data}), 200
        except subprocess.CalledProcessError as e:
            app.logger.error(f"Subprocess error: {e.output.decode('utf-8')}")
            return jsonify({'error': 'Error processing images', 'details': e.output.decode('utf-8')}), 500
        except Exception as e:
            app.logger.error(f"Unexpected error in subprocess: {e}")
            return jsonify({'error': 'Unexpected error occurred during image processing', 'details': str(e)}), 500

    except Exception as err:
        app.logger.error(f"Unhandled exception: {err}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred', 'details': str(err)}), 500

@app.route('/api/v1/spending/analyze', methods=['POST'])
@jwt_required()
def analyze_spending():
    try:
        current_user = get_jwt_identity()
        app.logger.debug(f"Current User ID from JWT: {current_user}")

        data = request.json
        user_id = data.get("user_id")

        if not user_id:
            app.logger.error("필수 파라미터가 누락되었습니다.")
            return jsonify({'error': '필수 파라미터가 누락되었습니다. user_id를 확인하세요.'}), 400

        # 현재 날짜 기준 월 계산
        current_month = datetime.now().month
        app.logger.debug(f"Current month determined as: {current_month}")

        # analyze.py 실행, 입력 파라미터 전달
        try:
            command = ['python3', 'analyze.py', str(user_id), str(current_month)]
            app.logger.debug(f"Executing command: {' '.join(command)}")
            result = subprocess.check_output(
                command,
                stderr=subprocess.STDOUT
            )
            result_data = result.decode('utf-8')
            analysis_result = json.loads(result_data)
            app.logger.debug(f"Analysis result JSON: {analysis_result}")
            return jsonify(analysis_result), 200
        except subprocess.CalledProcessError as e:
            error_output = e.output.decode('utf-8')
            app.logger.error(f"Subprocess error: {error_output}")
            return jsonify({'error': 'Error processing analysis', 'details': error_output}), 500
        except json.JSONDecodeError as e:
            app.logger.error(f"JSON decode error: {e}")
            app.logger.debug(f"Result data: {result_data}")
            return jsonify({'error': 'Invalid JSON response from analysis script', 'details': str(e)}), 500
        except Exception as e:
            app.logger.error(f"Unexpected error in subprocess: {e}", exc_info=True)
            return jsonify({'error': 'Unexpected error occurred during spending analysis', 'details': str(e)}), 500

    except Exception as err:
        app.logger.error(f"Unhandled exception: {err}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred', 'details': str(err)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
