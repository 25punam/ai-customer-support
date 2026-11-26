import time
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from app.ai.decide import decide
from app.ai.schemas import Decision
from evals.golden_emails import GOLDEN_EMAILS


@dataclass
class SectionMatches(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.output.section == ctx.expected_output["section"]


@dataclass
class ActionMatches(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.output.action == ctx.expected_output["action"]


def run_pipeline(email: str) -> Decision:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return decide(email)
        except Exception as e:
            if attempt < max_retries - 1 and any(
                term in str(e).lower()
                for term in ["connection error", "connect", "timeout", "rate limit", "503", "502", "529"]
            ):
                time.sleep(1.5 * (attempt + 1))
                continue
            raise


dataset = Dataset(
    name="golden-emails",
    cases=[
        Case(
            name=case["name"],
            inputs=case["email"],
            expected_output={"section": case["section"], "action": case["action"]},
        )
        for case in GOLDEN_EMAILS
    ],
    evaluators=[SectionMatches(), ActionMatches()],
)

if __name__ == "__main__":
    report = dataset.evaluate_sync(run_pipeline, max_concurrency=2)
    report.print(include_input=False, include_output=True, include_durations=False)