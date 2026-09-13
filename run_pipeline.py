import os
import sys
import json
import time
import argparse
import logging
from typing import List, Dict, Any

from src.config import GOLDEN_SET_PATH, TARGET_BRAND_HANDLE
from src.dataset import CustomerSupportDatasetLoader
from src.rag_store import GroundingRAGStore
from src.agent import AppleSupportAgent
from src.baselines import TrivialBaseline, SimpleBaseline
from src.evaluation import AutomatedEvaluator, LLMAsJudgeEvaluator
from src.judge_agreement import HumanJudgeAgreementAnalyzer
from src.llm_client import GrokLLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineRunner")

def run_evaluation_pipeline(fast_mode: bool = False, max_samples: int = 200) -> Dict[str, Any]:
    start_time = time.time()
    logger.info("=== Starting @AppleSupport AI Support Agent Pipeline ===")

    # 1. Load Golden Evaluation Set
    if not os.path.exists(GOLDEN_SET_PATH):
        logger.info("Golden evaluation set missing. Generating dataset...")
        from data.generate_golden_set import build_golden_dataset
        build_golden_dataset()

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    if max_samples and max_samples < len(golden_data):
        golden_data = golden_data[:max_samples]

    logger.info(f"Loaded {len(golden_data)} golden evaluation examples.")

    # 2. Build Grounding RAG Index from Historical Dialogues
    data_file = "data/sample.csv" if os.path.exists("data/sample.csv") else None
    historical_pairs = []
    if data_file:
        loader = CustomerSupportDatasetLoader(
            file_path=data_file,
            brand_handle=TARGET_BRAND_HANDLE
        )
        historical_pairs = loader.load_and_parse()

    rag_store = GroundingRAGStore()
    rag_store.build_index(historical_pairs)
    logger.info(f"Built Grounding RAG Store with {len(historical_pairs)} historical resolutions.")

    # 3. Initialize Models & Evaluators
    llm_client = GrokLLMClient()
    agent = AppleSupportAgent(llm_client=llm_client, rag_store=rag_store)
    trivial_bm = TrivialBaseline()
    simple_bm = SimpleBaseline()

    judge_evaluator = LLMAsJudgeEvaluator(llm_client=llm_client)

    models = {
        "Trivial Baseline (Static Rule)": trivial_bm,
        "Simple Baseline (Zero-Shot)": simple_bm,
        "Proposed Agent (@AppleSupport RAG)": agent
    }

    y_true_intent = [ex["true_intent"] for ex in golden_data]
    y_true_escalation = [ex["true_escalation"] for ex in golden_data]
    ref_replies = [ex["reference_reply"] for ex in golden_data]
    human_judge_scores = [ex["human_judge_score"] for ex in golden_data]

    benchmark_summary = {}

    for model_name, model_obj in models.items():
        logger.info(f"Evaluating model: {model_name}...")
        y_pred_intent = []
        y_pred_escalation = []
        gen_replies = []
        judge_scores = []

        for idx, ex in enumerate(golden_data):
            cust_text = ex["customer_text"]
            res = model_obj.process_message(cust_text)

            pred_intent = res["intent"]
            pred_esc = res["escalation_decision"]
            draft_reply = res["draft_reply"]

            y_pred_intent.append(pred_intent)
            y_pred_escalation.append(pred_esc)
            gen_replies.append(draft_reply)

            # LLM Judge Evaluation (Subsampled in fast mode for speed)
            if fast_mode and idx >= 30:
                # Use heuristic judge for fast evaluation
                j_res = judge_evaluator._heuristic_judge(cust_text, draft_reply, ex["reference_reply"])
            else:
                j_res = judge_evaluator.judge_reply(cust_text, draft_reply, ex["reference_reply"])
            
            judge_scores.append(j_res["overall_score"])

        # Compute Automated Metrics
        intent_metrics = AutomatedEvaluator.calculate_intent_metrics(y_true_intent, y_pred_intent)
        esc_metrics = AutomatedEvaluator.calculate_escalation_metrics(y_true_escalation, y_pred_escalation)

        # Compute Reply Quality Metrics
        rouge_scores = [AutomatedEvaluator.calculate_reply_rouge_l(g, r) for g, r in zip(gen_replies, ref_replies)]
        kw_scores = [AutomatedEvaluator.calculate_keyword_match(g) for g in gen_replies]

        avg_rouge = round(sum(rouge_scores) / len(rouge_scores), 4)
        avg_kw = round(sum(kw_scores) / len(kw_scores), 4)
        avg_judge_score = round(sum(judge_scores) / len(judge_scores), 2)

        # Human-Judge Agreement Metrics (for Proposed Agent)
        agreement_metrics = {}
        if model_name == "Proposed Agent (@AppleSupport RAG)":
            agreement_metrics = HumanJudgeAgreementAnalyzer.evaluate_agreement(human_judge_scores, judge_scores)

        benchmark_summary[model_name] = {
            "intent_metrics": intent_metrics,
            "escalation_metrics": esc_metrics,
            "reply_rouge_l": avg_rouge,
            "keyword_match_rate": avg_kw,
            "llm_judge_score": avg_judge_score,
            "judge_agreement": agreement_metrics
        }

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"Pipeline evaluation completed in {elapsed} seconds.")

    # 4. Print Benchmark Table to Console
    print("\n" + "=" * 90)
    print("                     @AppleSupport AI AGENT BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'Model / Architecture':<35} | {'Intent F1':<10} | {'Esc F1':<10} | {'ROUGE-L':<10} | {'Judge (1-5)':<10}")
    print("-" * 90)

    for m_name, m_res in benchmark_summary.items():
        intent_f1 = m_res["intent_metrics"]["macro_f1"]
        esc_f1 = m_res["escalation_metrics"]["f1_score"]
        rouge = m_res["reply_rouge_l"]
        j_score = m_res["llm_judge_score"]
        print(f"{m_name:<35} | {intent_f1:<10.4f} | {esc_f1:<10.4f} | {rouge:<10.4f} | {j_score:<10.2f}")

    print("=" * 90)

    # Print Agreement Summary
    agent_agr = benchmark_summary["Proposed Agent (@AppleSupport RAG)"]["judge_agreement"]
    print("\n--- LLM-as-a-Judge vs. Human Gold Agreement Evidence ---")
    print(f"Cohen's Kappa (Quadratic Weighted) : {agent_agr.get('cohen_kappa_quadratic')}")
    print(f"Exact Score Agreement               : {agent_agr.get('exact_agreement') * 100:.1f}%")
    print(f"Adjacent Agreement (+/- 1 score)     : {agent_agr.get('adjacent_agreement_pm1') * 100:.1f}%")
    print(f"Pearson Correlation                 : {agent_agr.get('pearson_correlation')}")
    print("=" * 90 + "\n")

    # Save to json file
    results_path = "data/benchmark_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)
    logger.info(f"Saved benchmark results to {results_path}")

    return benchmark_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run @AppleSupport AI Agent Pipeline Evaluation Harness")
    parser.add_argument("--fast", action="store_true", help="Run fast evaluation mode")
    parser.add_argument("--samples", type=int, default=200, help="Number of eval samples")
    args = parser.parse_args()

    run_evaluation_pipeline(fast_mode=args.fast, max_samples=args.samples)
