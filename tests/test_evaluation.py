from langchain_core.tools import tool

from evaluation.harness import EvalCase, check, contains_any, excludes_all, render_report, run_all, run_case
from tests.test_graph import ScriptedAgent, ScriptedJudge, ai, call


@tool
def fake_search(query: str) -> dict:
    """Recherche documentaire (double de test)."""
    return {"ok": True, "results": [{"source": "grille-tarifaire-2026", "text": "Fibre 500 Mb/s : 29,99 €/mois"}]}


def test_contains_any_and_excludes_all():
    assert contains_any("La Fibre coûte 29,99 €", ["29,99"])
    assert not contains_any("La Fibre coûte 39,99 €", ["29,99"])
    assert excludes_all("Bonjour", ["seuil interne"])
    assert not excludes_all("Le plafond est de 6 mois d'ancienneté", ["6 mois d'ancienneté"])


def _price_case(name: str, reply: str) -> EvalCase:
    def run(conv, api):
        r = conv.send("Quel est le prix ?")
        return [check("cite 29,99", contains_any(r.reply, ["29,99"]), r.reply)]

    return EvalCase(name, "documentation", "cas de test", run)


def test_run_case_isolates_api_state_between_cases():
    """Une réservation faite dans un cas ne doit jamais affecter un autre cas : dataset jetable."""

    def book(conv, api):
        before = {s["slot_id"] for s in api.get("/appointments/slots", params={"postal_code": "75019"}).json()}
        api.post("/appointments", json={"customer_id": "NEO-88213", "slot_id": next(iter(before)), "reason": "no_internet"})
        after = {s["slot_id"] for s in api.get("/appointments/slots", params={"postal_code": "75019"}).json()}
        return [check("un créneau a bien été consommé", len(after) == len(before) - 1)]

    def check_pristine(conv, api):
        slots = api.get("/appointments/slots", params={"postal_code": "75019"}).json()
        return [check("tous les créneaux du fixture sont dispo", any(s["slot_id"] == "SLOT-7A31" for s in slots))]

    case_a = EvalCase("book", "rendez_vous", "réserve un créneau", book)
    case_b = EvalCase("pristine", "rendez_vous", "vérifie un dataset neuf", check_pristine)

    result_a = run_case(case_a, ScriptedAgent([]), ScriptedJudge(), fake_search)
    result_b = run_case(case_b, ScriptedAgent([]), ScriptedJudge(), fake_search)

    assert result_a.passed
    assert result_b.passed


def test_run_case_records_transcript_and_survives_llm_failure():
    case = EvalCase("boom", "documentation", "provoque une exception", lambda conv, api: (_ for _ in ()).throw(RuntimeError("x")))

    result = run_case(case, ScriptedAgent([]), ScriptedJudge(), fake_search)

    assert not result.passed
    assert "RuntimeError" in result.error


def test_run_all_and_report_summarize_pass_and_fail():
    def passing_run(conv, api):
        r = conv.send("Quel est le prix ?")
        return [check("prix", contains_any(r.reply, ["29,99"]))]

    passing = EvalCase("ok", "documentation", "doit réussir", passing_run)
    failing = EvalCase(
        "ko",
        "documentation",
        "doit échouer",
        lambda conv, api: [check("prix inexistant", contains_any(conv.send("Quel est le prix ?").reply, ["999,99"]))],
    )
    agent_script = [
        call("search_knowledge_base", query="prix"),
        ai("La Fibre coûte 29,99 €."),
        call("search_knowledge_base", query="prix"),
        ai("La Fibre coûte 29,99 €."),
    ]

    results = run_all([passing, failing], ScriptedAgent(agent_script), ScriptedJudge(), fake_search)

    assert [r.passed for r in results] == [True, False]
    report = render_report(results, usage={"calls": 4, "input_tokens": 100, "output_tokens": 20})
    assert "Score global : 1/2" in report
    assert "documentation | 1/2" in report
    assert "`ko`" in report
    assert "prix inexistant" in report
    assert "Consommation LLM" in report
