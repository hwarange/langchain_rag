from edges import app
from utils import config

if __name__ == "__main__":
    # 여기에 실행 코드 추가
    result = app.invoke({'messages': '서울시 강남역에서 1000미터 이내 전세 10억 매물을 추천해줘'}, config=config)
    print(result['answers'])