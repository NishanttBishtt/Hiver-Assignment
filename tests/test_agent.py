import unittest
from src.agent import AppleSupportAgent
from src.baselines import TrivialBaseline, SimpleBaseline
from src.rag_store import GroundingRAGStore
from src.evaluation import AutomatedEvaluator, LLMAsJudgeEvaluator
from src.judge_agreement import HumanJudgeAgreementAnalyzer

class TestAppleSupportAgent(unittest.TestCase):

    def setUp(self):
        self.rag_store = GroundingRAGStore()
        self.rag_store.build_index([
            {"customer_text": "Battery drains fast after iOS 11 update", "brand_reply": "DM us your iOS version in Settings > General > About."}
        ])
        self.agent = AppleSupportAgent(rag_store=self.rag_store)
        self.trivial = TrivialBaseline()
        self.simple = SimpleBaseline()

    def test_intent_classification(self):
        res = self.agent.process_message("@AppleSupport my battery is draining 10% every 5 minutes after iOS update!")
        self.assertEqual(res["intent"], "battery_drain")
        self.assertEqual(res["escalation_decision"], "AUTO_HANDLE")

    def test_escalation_routing(self):
        res = self.agent.process_message("@AppleSupport I dropped my iPhone and the screen is completely shattered!")
        self.assertEqual(res["intent"], "hardware_repair")
        self.assertEqual(res["escalation_decision"], "ESCALATE_TO_HUMAN")
        self.assertIn("repair", res["escalation_reason"].lower())

    def test_baselines(self):
        t_res = self.trivial.process_message("My battery is dead")
        self.assertEqual(t_res["intent"], "software_bugs")
        self.assertEqual(t_res["escalation_decision"], "AUTO_HANDLE")

        s_res = self.simple.process_message("I need a refund for my subscription")
        self.assertEqual(s_res["intent"], "billing_subscriptions")

    def test_metrics_evaluator(self):
        intent_m = AutomatedEvaluator.calculate_intent_metrics(["battery_drain", "software_bugs"], ["battery_drain", "software_bugs"])
        self.assertEqual(intent_m["accuracy"], 1.0)

        rouge = AutomatedEvaluator.calculate_reply_rouge_l("DM us your iOS version", "DM us your iOS version")
        self.assertEqual(rouge, 1.0)

    def test_judge_agreement(self):
        human = [5, 4, 3, 5, 2]
        judge = [5, 4, 3, 4, 2]
        res = HumanJudgeAgreementAnalyzer.evaluate_agreement(human, judge)
        self.assertEqual(res["exact_agreement"], 0.8)
        self.assertEqual(res["adjacent_agreement_pm1"], 1.0)

if __name__ == "__main__":
    unittest.main()
