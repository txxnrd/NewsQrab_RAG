"""
RAG 대화 품질 평가 벤치마크 시스템
=================================
Original 대화와 RagModified 대화를 비교하여 LLM-as-a-judge로 평가하는 시스템
"""

import json
import os
import random
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import logging

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationBenchmark:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)
        self.evaluation_criteria = {
            "information_richness": {
                "name": "정보 풍부함",
                "description": "뉴스 기사의 다양한 정보와 맥락이 얼마나 풍부하게 포함되었는가"
            },
            "factual_accuracy": {
                "name": "사실 정확성", 
                "description": "제공된 정보가 얼마나 정확하고 신뢰할 수 있는가"
            },
            "contextual_relevance": {
                "name": "맥락 적절성",
                "description": "뉴스 기사의 맥락과 배경 정보가 얼마나 잘 반영되었는가"
            },
            "educational_value": {
                "name": "교육적 가치",
                "description": "대화를 통해 얻을 수 있는 학습 효과와 이해도가 얼마나 높은가"
            },
            "overall_informativeness": {
                "name": "전반적 정보성",
                "description": "전체적으로 대화가 얼마나 유익하고 정보전달력이 높은가"
            }
        }
    
    def format_script(self, script: List[Dict[str, str]]) -> str:
        """스크립트 배열을 문자열로 변환"""
        formatted = []
        for turn in script:
            for speaker, content in turn.items():
                formatted.append(f"{speaker}: {content}")
        return "\n".join(formatted)
    
    def create_evaluation_prompt(self, news_content: str, script_a: str, script_b: str) -> str:
        """평가용 프롬프트 생성 (마스킹 적용)"""
        return f"""
당신은 뉴스 기사를 기반으로 한 대화 스크립트의 품질을 평가하는 전문가입니다.

아래 뉴스 기사를 기반으로 작성된 두 가지 대화 스크립트를 비교하여 평가해주세요.

**뉴스 기사:**
{news_content}

**스크립트 A:**
{script_a}

**스크립트 B:**
{script_b}

다음 5가지 기준으로 각각 1-5점(5점이 최고)으로 평가해주세요:

1. **정보 정확성 (Information Accuracy)**: 뉴스 기사의 내용이 얼마나 정확하게 반영되었는가
2. **대화 자연스러움 (Conversational Flow)**: 대화가 얼마나 자연스럽고 일관성 있게 흘러가는가  
3. **캐릭터 일관성 (Character Consistency)**: 캐릭터의 말투와 성격이 일관되게 유지되는가
4. **내용 완전성 (Content Completeness)**: 뉴스 기사의 핵심 내용이 충분히 다뤄졌는가
5. **전반적 품질 (Overall Quality)**: 전체적으로 대화의 품질이 얼마나 높은가

**반드시 아래 JSON 형식으로 답변해주세요:**

```json
{{
  "script_a_scores": {{
    "information_accuracy": [1-5 점수],
    "conversational_flow": [1-5 점수],
    "character_consistency": [1-5 점수], 
    "content_completeness": [1-5 점수],
    "overall_quality": [1-5 점수]
  }},
  "script_b_scores": {{
    "information_accuracy": [1-5 점수],
    "conversational_flow": [1-5 점수],
    "character_consistency": [1-5 점수],
    "content_completeness": [1-5 점수], 
    "overall_quality": [1-5 점수]
  }},
  "comparison": {{
    "winner": "script_a" 또는 "script_b" 또는 "tie",
    "reasoning": "판단 근거 설명"
  }}
}}
```

JSON 외의 다른 내용은 포함하지 마세요.
"""

    def parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """GPT 응답에서 JSON 평가 결과 파싱"""
        try:
            # JSON 코드 블록에서 JSON 추출
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"JSON 파싱 오류: {e}")
            logger.error(f"응답 내용: {response}")
            return None

    def evaluate_conversation(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """단일 대화 항목 평가 (마스킹 적용)"""
        logger.info(f"평가 중: {item.get('url', 'N/A')}")
        
        # 새로운 구조에서 스크립트 추출
        conversations = item.get("conversations", [])
        
        # original과 rag-modified 스크립트 찾기
        original_script = None
        rag_script = None
        
        for conv in conversations:
            if conv["type"] == "original":
                original_script = self.format_script(conv["script"])
            elif conv["type"] == "rag-modified":
                rag_script = self.format_script(conv["script"])
        
        if original_script is None or rag_script is None:
            logger.error("Original 또는 RAG 스크립트를 찾을 수 없습니다.")
            return None
        
        # 랜덤하게 순서 결정 (편향 방지)
        is_original_first = random.choice([True, False])
        
        if is_original_first:
            script_a = original_script
            script_b = rag_script
            a_is_original = True
        else:
            script_a = rag_script
            script_b = original_script
            a_is_original = False
        
        # 평가 프롬프트 생성 (마스킹됨)
        prompt = self.create_evaluation_prompt(
            item.get("content", ""),
            script_a,
            script_b
        )
        
        # GPT 평가 실행
        try:
            response = self.llm.invoke(prompt)
            masked_evaluation = self.parse_evaluation_response(response.content)
            
            if masked_evaluation is None:
                return None
            
            # 마스킹된 결과를 원래 라벨로 복원
            if a_is_original:
                original_scores = masked_evaluation["script_a_scores"]
                rag_scores = masked_evaluation["script_b_scores"]
                winner_mapping = {"script_a": "original", "script_b": "rag_modified", "tie": "tie"}
            else:
                original_scores = masked_evaluation["script_b_scores"]
                rag_scores = masked_evaluation["script_a_scores"]
                winner_mapping = {"script_a": "rag_modified", "script_b": "original", "tie": "tie"}
            
            # 최종 평가 결과 구성
            evaluation = {
                "original_scores": original_scores,
                "rag_modified_scores": rag_scores,
                "comparison": {
                    "winner": winner_mapping[masked_evaluation["comparison"]["winner"]],
                    "reasoning": masked_evaluation["comparison"]["reasoning"]
                },
                "metadata": {
                    "url": item.get("url", ""),
                    "id": item.get("_id", {}).get("$oid", "") if item.get("_id") else "",
                    "evaluation_timestamp": datetime.now().isoformat(),
                    "original_was_script_a": a_is_original  # 디버깅용
                }
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"평가 오류: {e}")
            return None

    def run_benchmark(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """전체 벤치마크 실행"""
        logger.info(f"벤치마크 시작: {input_file}")
        
        # 데이터 로드
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        successful_evaluations = 0
        
        # 각 항목 평가
        for idx, item in enumerate(data):
            logger.info(f"평가 진행: {idx + 1}/{len(data)}")
            
            evaluation = self.evaluate_conversation(item)
            if evaluation:
                results.append(evaluation)
                successful_evaluations += 1
            else:
                logger.warning(f"평가 실패: 항목 {idx + 1}")
        
        # 통계 계산
        stats = self.calculate_statistics(results)
        
        # 결과 저장
        final_results = {
            "benchmark_info": {
                "total_items": len(data),
                "successful_evaluations": successful_evaluations,
                "success_rate": successful_evaluations / len(data) * 100,
                "timestamp": datetime.now().isoformat(),
                "input_file": input_file
            },
            "statistics": stats,
            "detailed_results": results
        }
        
        # 결과 파일 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"벤치마크 완료: {output_file}")
        return final_results

    def calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """평가 결과 통계 계산"""
        if not results:
            return {}
        
        criteria = list(self.evaluation_criteria.keys())
        
        # 각 기준별 평균 점수 계산
        original_avg = {}
        rag_avg = {}
        
        for criterion in criteria:
            original_scores = [r["original_scores"][criterion] for r in results]
            rag_scores = [r["rag_modified_scores"][criterion] for r in results]
            
            original_avg[criterion] = sum(original_scores) / len(original_scores)
            rag_avg[criterion] = sum(rag_scores) / len(rag_scores)
        
        # 승패 통계
        winners = [r["comparison"]["winner"] for r in results]
        winner_stats = {
            "original_wins": winners.count("original"),
            "rag_modified_wins": winners.count("rag_modified"),
            "ties": winners.count("tie")
        }
        
        return {
            "original_average_scores": original_avg,
            "rag_modified_average_scores": rag_avg,
            "winner_statistics": winner_stats,
            "rag_improvement": {
                criterion: rag_avg[criterion] - original_avg[criterion]
                for criterion in criteria
            }
        }

    def print_summary(self, results: Dict[str, Any]):
        """결과 요약 출력"""
        print("\n" + "="*50)
        print("RAG 대화 품질 평가 결과")
        print("="*50)
        
        info = results["benchmark_info"]
        print(f"총 항목: {info['total_items']}")
        print(f"성공적 평가: {info['successful_evaluations']}")
        print(f"성공률: {info['success_rate']:.1f}%")
        
        if "statistics" in results:
            stats = results["statistics"]
            
            print("\n[평균 점수 비교]")
            for criterion in self.evaluation_criteria:
                original = stats["original_average_scores"][criterion]
                rag = stats["rag_modified_average_scores"][criterion]
                improvement = stats["rag_improvement"][criterion]
                
                print(f"{self.evaluation_criteria[criterion]['name']}: "
                      f"Original {original:.2f} vs RAG {rag:.2f} "
                      f"(차이: {improvement:+.2f})")
            
            print("\n[승패 통계]")
            winner_stats = stats["winner_statistics"]
            print(f"Original 승리: {winner_stats['original_wins']}")
            print(f"RAG Modified 승리: {winner_stats['rag_modified_wins']}")
            print(f"무승부: {winner_stats['ties']}")


def main():
    """메인 실행 함수"""
    benchmark = ConversationBenchmark()
    
    # 입력/출력 파일 경로
    input_file = "rag_benchmark/data/conversation_example_50.json"
    output_file = f"rag_benchmark/results/benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # 결과 디렉토리 생성
    os.makedirs("rag_benchmark/results", exist_ok=True)
    
    # 벤치마크 실행
    results = benchmark.run_benchmark(input_file, output_file)
    
    # 결과 요약 출력
    benchmark.print_summary(results)
    
    print(f"\n자세한 결과는 {output_file}에서 확인할 수 있습니다.")


if __name__ == "__main__":
    main()
