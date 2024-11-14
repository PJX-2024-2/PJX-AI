import pymysql
import os
import openai
import pandas as pd
from dotenv import load_dotenv
import json
import logging
import time
import re
from decimal import Decimal
from datetime import datetime
import argparse

# 환경 변수 로드
load_dotenv()

# 로깅 설정 (DEBUG 레벨로 변경하여 더 상세한 로그를 남김)
logging.basicConfig(
    filename='analyze.log',  # 별도의 로그 파일 설정
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 필수 환경 변수 검증
required_vars = ['DB_HOST', 'DB_USER', 'DB_PASSWORD', 'DB_NAME', 'DB_PORT', 'OPENAI_API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    logging.error(f"Missing environment variables: {', '.join(missing_vars)}")
    raise EnvironmentError("필수 환경 변수가 설정되지 않았습니다.")

# OpenAI API 키 설정
openai.api_key = os.getenv('OPENAI_API_KEY')

# 데이터베이스 연결 정보 가져오기
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

def fetch_spending_data(kakao_id, month):
    """특정 kakao_id와 월의 지출 데이터를 가져옵니다."""
    logging.debug(f"Fetching spending data for kakao_id={kakao_id}, month={month}")
    sql = """
        SELECT id, description, amount
        FROM spending
        WHERE kakao_id = %s AND MONTH(date) = %s
    """
    try:
        with pymysql.connect(**db_config) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, (kakao_id, month))
                results = cursor.fetchall()
        df = pd.DataFrame(results)
        logging.info("지출 데이터를 성공적으로 가져왔습니다.")
        return df
    except pymysql.MySQLError as e:
        logging.error(f"지출 데이터 조회 중 오류 발생: {e}", exc_info=True)
        raise

def fetch_monthly_goal(user_id):
    """특정 user_id의 월간 목표 금액을 가져옵니다."""
    logging.debug(f"Fetching monthly goal for user_id={user_id}")
    sql = """
        SELECT monthly_goal
        FROM spending_goal
        WHERE user_id = %s
    """
    try:
        with pymysql.connect(**db_config) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, (user_id,))
                result = cursor.fetchone()
        if result:
            logging.info(f"user_id={user_id}의 월간 목표 금액: {result['monthly_goal']}원")
            return Decimal(result['monthly_goal'])
        else:
            logging.warning(f"user_id={user_id}에 해당하는 월간 목표 금액이 없습니다.")
            return Decimal('0.00')
    except pymysql.MySQLError as e:
        logging.error(f"월간 목표 금액 조회 중 오류 발생: {e}", exc_info=True)
        raise

def categorize_products(product_names, batch_size=20):
    """상품명을 카테고리별로 분류합니다."""
    logging.debug("Starting product categorization")
    category_mapping = {}
    unique_names = list(set(product_names))
    total_batches = len(unique_names) // batch_size + 1

    for i in range(0, len(unique_names), batch_size):
        batch = unique_names[i:i+batch_size]
        prompt = (
            "다음은 다양한 상품명 리스트입니다. 각 상품명을 '식품', '생활용품', '주류', '기타' 중 하나의 카테고리로 분류해주세요. "
            "카테고리화 결과는 '상품명: 카테고리' 형식으로 한 줄에 하나씩 작성해주세요.\n\n"
        )
        prompt += "\n".join(f"- {name}" for name in batch)

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 유용한 카테고리 분류 도우미입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            response_content = response.choices[0].message['content'].strip()
            logging.debug(f"Batch {i//batch_size + 1} 응답: {response_content}")

            for line in response_content.split('\n'):
                if ':' in line:
                    product, category = line.split(':', 1)
                    product = product.strip().lstrip('- ').strip()
                    category = category.strip() if category.strip() in ['식품', '생활용품', '주류', '기타'] else '기타'
                    category_mapping[product] = category
                else:
                    logging.warning(f"잘못된 형식의 라인: {line}")

            logging.info(f"Batch {i//batch_size + 1} 카테고리 분류 완료.")
        except Exception as e:
            logging.error(f"카테고리 분류 중 오류 발생: {e}", exc_info=True)

    logging.debug("Product categorization completed")
    return category_mapping

def format_expenses_for_analysis(df, monthly_goal):
    """지출 내역을 분석을 위한 문자열 형식으로 변환합니다."""
    logging.debug("Formatting expenses for analysis")
    df['카테고리'] = df['카테고리'].fillna('미분류')
    formatted = df.to_string(index=False, columns=['id', 'description', 'amount', '카테고리'])
    return formatted, monthly_goal

def create_analysis_prompt(formatted_expenses, monthly_goal):
    """분석 요청을 위한 프롬프트를 생성합니다."""
    logging.debug("Creating analysis prompt")
    prompt = (
        "아래는 사용자의 최근 한 달 간 지출 내역입니다. 이 지출 내역을 분석하여 한국어로 다음 정보를 JSON 형식으로 **반드시** 제공해주세요. "
        "응답은 반드시 JSON 코드 블록(```json`으로 시작하여 ```로 끝나야 합니다.)으로 작성해주세요.\n\n"
        "1. 총 지출 금액\n"
        "2. 주요 지출 카테고리\n"
        "3. 월간 예산 대비 초과 또는 절약 금액\n"
        "4. 지출 패턴 또는 트렌드\n"
        "5. 지출 절약을 위한 추천 사항\n\n"
        f"월간 예산: {monthly_goal}원\n\n"
        f"{formatted_expenses}\n"
        "응답은 반드시 JSON 코드 블록 안에 작성해주세요. 예시는 다음과 같습니다:\n\n"
        "```json\n"
        "{\n"
        "    \"총 지출 금액\": \"100,000원\",\n"
        "    \"주요 지출 카테고리\": \"식품\",\n"
        "    \"월간 예산 대비 초과 금액\": \"초과: 23,530원\",\n"
        "    \"지출 패턴\": \"식품 및 주방용품에 주로 지출하며, 대부분 식자재 및 가공식품 구매\",\n"
        "    \"지출 절약을 위한 추천 사항\": \"식료품 구매 시 할인 행사 및 식자재 구매에 집중하여 예산을 절약하거나, 필요 이상의 구매를 줄여보세요\"\n"
        "}\n"
        "```"
    )
    return prompt

def analyze_expenses(prompt, total_spending, retries=3):
    """OpenAI API를 사용하여 지출 내역을 분석합니다."""
    logging.debug("Starting expense analysis with OpenAI")
    backoff = 1
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 유용한 지출 분석 도우미입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.5
            )
            response_content = response.choices[0].message['content'].strip()
            logging.debug(f"분석 응답: {response_content}")

            # JSON 블록 추출
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*?\})', response_content, re.DOTALL)

            if json_match:
                try:
                    analysis = json.loads(json_match.group(1))
                    analysis['총 지출 금액'] = float(total_spending)
                    logging.info("지출 분석 성공.")
                    return analysis
                except json.JSONDecodeError as json_err:
                    logging.error(f"JSON 파싱 오류: {json_err}", exc_info=True)
                    logging.debug(f"응답 내용: {json_match.group(1)}")
            else:
                logging.error("JSON 블록을 찾을 수 없습니다.")
                logging.debug(f"응답 내용: {response_content}")
        except Exception as e:
            logging.error(f"지출 분석 중 오류 발생: {e}", exc_info=True)

        if attempt < retries - 1:
            logging.info(f"재시도 중... {backoff}초 대기")
            time.sleep(backoff)
            backoff *= 2

    logging.error("지출 분석 실패.")
    return {}

def calculate_budget_difference(monthly_goal, total_spending):
    """월간 예산 대비 초과 또는 절약 금액을 계산합니다."""
    logging.debug("Calculating budget difference")
    difference = monthly_goal - total_spending
    if difference > 0:
        return f"{difference:.2f}원 절약"
    elif difference < 0:
        return f"{-difference:.2f}원 초과"
    else:
        return "예산과 지출이 동일합니다."

class DecimalEncoder(json.JSONEncoder):
    """Decimal 타입을 float으로 변환하는 JSON 인코더."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

def main():
    """주요 실행 함수."""
    parser = argparse.ArgumentParser(description='지출 분석 스크립트')
    parser.add_argument('kakao_id', type=int, help='카카오 ID')
    parser.add_argument('current_month', type=int, help='현재 월 (1-12)')
    parser.add_argument('user_id', type=int, help='사용자 ID')

    args = parser.parse_args()
    kakao_id = args.kakao_id
    current_month = args.current_month
    user_id = args.user_id

    logging.info(f"Starting analysis for kakao_id={kakao_id}, current_month={current_month}, user_id={user_id}")

    # 지출 데이터 가져오기
    try:
        df = fetch_spending_data(kakao_id=kakao_id, month=current_month)
    except Exception as e:
        logging.error("지출 데이터를 가져오는 데 실패했습니다.", exc_info=True)
        print(json.dumps({'error': 'Failed to fetch spending data'}, ensure_ascii=False))
        return

    if df.empty:
        logging.error("지출 데이터가 없습니다.")
        print(json.dumps({'error': 'No spending data found'}, ensure_ascii=False))
        return

    # 월간 목표 금액 가져오기
    try:
        monthly_goal = fetch_monthly_goal(user_id)
    except Exception as e:
        logging.error(f"user_id={user_id}의 월간 목표 금액을 가져오는 데 실패했습니다.", exc_info=True)
        print(json.dumps({'error': 'Failed to fetch monthly goal'}, ensure_ascii=False))
        return

    # 총 지출 금액 계산
    total_spending = df['amount'].sum()
    logging.info(f"총 지출 금액: {total_spending:.2f}원")

    # 상품명 카테고리 분류
    product_names = df['description'].tolist()
    category_mapping = categorize_products(product_names)
    df['카테고리'] = df['description'].map(category_mapping).fillna('미분류')

    # 지출 내역 형식화
    formatted_expenses, monthly_goal = format_expenses_for_analysis(df, monthly_goal)

    # 분석 프롬프트 생성
    prompt = create_analysis_prompt(formatted_expenses, monthly_goal)

    # 지출 내역 분석 수행
    analysis = analyze_expenses(prompt, total_spending)

    # 예산 대비 초과 또는 절약 금액 계산
    budget_difference = calculate_budget_difference(monthly_goal, Decimal(total_spending))
    analysis['월간 예산 대비 초과 또는 절약 금액'] = budget_difference

    # 분석 결과 출력
    if analysis:
        try:
            print(json.dumps(analysis, indent=4, ensure_ascii=False, cls=DecimalEncoder))
            logging.info("지출 분석 결과를 성공적으로 출력했습니다.")
        except TypeError as te:
            logging.error(f"JSON 직렬화 오류: {te}", exc_info=True)
            logging.debug(f"분석 결과 내용: {analysis}")
            print(json.dumps({'error': 'Failed to serialize analysis result'}, ensure_ascii=False))
    else:
        logging.error("지출 분석에 실패했습니다.")
        print(json.dumps({'error': 'Spending analysis failed'}, ensure_ascii=False))

if __name__ == "__main__":
    main()
