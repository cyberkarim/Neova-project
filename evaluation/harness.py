"""Socle d'évaluation : chaque cas rejoue une conversation contre un état d'API frais et isolé
(copie jetable du fixture, comme les tests), pour que les cas ne se polluent jamais entre eux."""
from __future__ import annotations

import logging
import shutil
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from fastapi.testclient import TestClient

from agent.graph import build_graph
from agent.session import Conversation
from agent.tools import build_api_tools
from api.main import app
from api.store import DataStore

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_DATA_PATH = REPO_ROOT / "data" / "neova_data.json"


@dataclass
class CheckResult:
    label: str
    passed: bool
    detail: str = ""


@dataclass
class EvalCase:
    id: str
    category: str
    description: str
    run: Callable[[Conversation, TestClient], list[CheckResult]]


@dataclass
class CaseResult:
    case: EvalCase
    checks: list[CheckResult] = field(default_factory=list)
    transcript: list[tuple[str, str]] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.checks and all(c.passed for c in self.checks)


def contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(n.lower() in lowered for n in needles)


def excludes_all(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return not any(n.lower() in lowered for n in needles)


def check(label: str, condition: bool, detail: str = "") -> CheckResult:
    return CheckResult(label, condition, detail)


def _fresh_client(tmp_path: Path) -> TestClient:
    base_copy = tmp_path / "neova_data.json"
    shutil.copy(BASE_DATA_PATH, base_copy)
    test_client = TestClient(app)
    test_client.__enter__()
    test_client.app.state.store = DataStore(base_path=base_copy, runtime_path=tmp_path / "runtime_state.json")
    return test_client


def run_case(case: EvalCase, chat_model, judge_model, search_tool) -> CaseResult:
    result = CaseResult(case=case)
    with tempfile.TemporaryDirectory(prefix=f"neova-eval-{case.id}-") as tmp:
        api = _fresh_client(Path(tmp))
        try:
            tools = build_api_tools(api, backoff_seconds=0)
            graph = build_graph(chat_model, judge_model, tools, search_tool)
            conv = Conversation(graph, thread_id=f"eval-{case.id}")
            recorder = _TranscriptConversation(conv, result.transcript)
            result.checks = case.run(recorder, api)
        except Exception:
            result.error = traceback.format_exc()
            logger.error("Cas %s en échec (exception): %s", case.id, result.error)
        finally:
            api.__exit__(None, None, None)
    return result


class _TranscriptConversation:
    """Enveloppe Conversation pour enregistrer chaque tour, utile au rapport d'erreurs."""

    def __init__(self, conv: Conversation, transcript: list[tuple[str, str]]):
        self._conv = conv
        self._transcript = transcript

    def send(self, text: str):
        self._transcript.append(("client", text))
        result = self._conv.send(text)
        self._transcript.append(("agent", result.reply))
        return result

    def state(self) -> dict:
        return self._conv.state()


def run_all(cases: list[EvalCase], chat_model, judge_model, search_tool) -> list[CaseResult]:
    results = []
    for case in cases:
        logger.info("Exécution du cas: %s", case.id)
        results.append(run_case(case, chat_model, judge_model, search_tool))
    return results


def render_report(results: list[CaseResult], usage: dict | None = None) -> str:
    total = len(results)
    passed = sum(r.passed for r in results)
    lines = [f"# Résultats d'évaluation\n", f"**Score global : {passed}/{total}**\n"]

    by_category: dict[str, list[CaseResult]] = {}
    for r in results:
        by_category.setdefault(r.case.category, []).append(r)

    lines.append("| Catégorie | Score |")
    lines.append("|---|---|")
    for cat, rs in sorted(by_category.items()):
        lines.append(f"| {cat} | {sum(r.passed for r in rs)}/{len(rs)} |")

    if usage:
        lines.append(
            f"\n**Consommation LLM** : {usage['calls']} appels, "
            f"{usage['input_tokens']} tokens en entrée, {usage['output_tokens']} tokens en sortie.\n"
        )

    failures = [r for r in results if not r.passed]
    if failures:
        lines.append("\n## Échecs\n")
        for r in failures:
            lines.append(f"### `{r.case.id}` ({r.case.category})")
            lines.append(r.case.description)
            if r.error:
                lines.append(f"\n**Exception :**\n```\n{r.error}\n```")
            else:
                for c in r.checks:
                    if not c.passed:
                        lines.append(f"- ❌ {c.label}" + (f" — {c.detail}" if c.detail else ""))
            lines.append("\n**Échange :**")
            for speaker, text in r.transcript:
                lines.append(f"- *{speaker}* : {text}")
            lines.append("")

    return "\n".join(lines)
