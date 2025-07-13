"""
데이터 분할 스크립트
==================
conversation_example.json 파일을 5개 파일로 분할합니다.
10개/10개/10개/10개/6개로 나누어 1.json, 2.json, 3.json, 4.json, 5.json으로 저장합니다.
"""

import json
import os

def split_data():
    """원본 데이터를 5개 파일로 분할"""
    
    # 원본 데이터 로드
    input_file = "rag_benchmark/data/conversation_example.json"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"총 데이터 수: {len(data)}")
    
    # 분할 크기 정의
    split_sizes = [10, 10, 10, 10, 6]  # 46개를 5개 파일로 분할
    
    # 분할 실행
    current_index = 0
    
    for i, size in enumerate(split_sizes, 1):
        # 현재 청크 추출
        chunk = data[current_index:current_index + size]
        
        # 파일명 생성
        output_file = f"rag_benchmark/data/{i}.json"
        
        # 파일 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)
        
        print(f"파일 {i}.json 생성 완료: {len(chunk)}개 데이터 (인덱스 {current_index}-{current_index + size - 1})")
        
        current_index += size
    
    print("모든 파일 분할 완료!")

if __name__ == "__main__":
    split_data() 