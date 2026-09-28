"""Jeu d'évaluation : un cas par ligne du tableau du README, chacun avec ses propres vérifications.

Les vérifications sont volontairement déterministes (sous-chaînes, tools appelés, état réel de
l'API) plutôt que confiées à un LLM juge : plus fiable, moins cher, et les échecs sont explicables
sans ambiguïté dans le rapport. La contrepartie est notée dans le README (limite connue)."""
from __future__ import annotations

from datetime import datetime

from agent.graph import ALREADY_HANDED_OVER_MESSAGE
from evaluation.harness import CheckResult, EvalCase, check, contains_any, excludes_all

NEO_88213 = ("NEO-88213", "0612840193", "75019")
NEO_10467 = ("NEO-10467", "0778115402", "69007")


def _slots(api, postal_code: str) -> list[dict]:
    return api.get("/appointments/slots", params={"postal_code": postal_code}).json()


def _slot_phrase(slot: dict, reason: str = "une panne internet") -> str:
    start, end = datetime.fromisoformat(slot["start"]), datetime.fromisoformat(slot["end"])
    return f"Le {start:%d/%m/%Y} de {start:%Hh%M} à {end:%Hh%M}, c'est pour {reason}."


# ---- documentation publique, pas de compte requis ------------------------------------------


def documented_price_fibre500(conv, api) -> list[CheckResult]:
    r = conv.send("Bonjour, quel est le prix de la Fibre Néova 500 Mb/s ?")
    return [
        check("cite le tarif en vigueur (29,99)", contains_any(r.reply, ["29,99"]), r.reply),
        check("ne demande pas l'identité pour une info publique", not r.awaiting_confirmation),
        check("n'escalade pas", not r.escalated),
    ]


def documented_resiliation_frais(conv, api) -> list[CheckResult]:
    r = conv.send(
        "Quels sont les frais si je résilie un contrat de 24 mois d'engagement, à 29,99 €/mois, après 18 mois de service ?"
    )
    return [
        check("donne le bon calcul (25 % / 44,99)", contains_any(r.reply, ["44,99", "25 %", "25%"]), r.reply),
        check("n'escalade pas une question documentée", not r.escalated),
    ]


def documented_retour_delai(conv, api) -> list[CheckResult]:
    r = conv.send("Combien de temps ai-je pour retourner ma box après une résiliation ?")
    return [check("cite le délai de 15 jours", contains_any(r.reply, ["15 jours"]), r.reply)]


def roaming_png_appel_usa(conv, api) -> list[CheckResult]:
    r = conv.send("Combien coûte un appel émis depuis les États-Unis avec mon forfait mobile ?")
    return [check("trouve le tarif de la fiche roaming (0,50)", contains_any(r.reply, ["0,50"]), r.reply)]


# ---- documents contradictoires et questions sans réponse ------------------------------------


def contradictory_promo_vs_grille(conv, api) -> list[CheckResult]:
    r = conv.send(
        "Je voudrais souscrire à la Fibre Néova 500 Mb/s à 19,99 €/mois comme dans la promo de la rentrée, c'est possible ?"
    )
    return [
        check("indique le tarif actuel (29,99), pas l'ancienne promo", contains_any(r.reply, ["29,99"]), r.reply),
        check(
            "ne confirme pas l'ancien tarif comme applicable",
            excludes_all(r.reply, ["oui, c'est possible", "oui c'est possible", "bien sûr, à 19,99"]),
            r.reply,
        ),
    ]


def no_answer_streaming_inclus(conv, api) -> list[CheckResult]:
    r = conv.send("Est-ce que Néova propose un service de streaming vidéo inclus gratuitement dans les offres fibre ?")
    honest = contains_any(
        r.reply,
        [
            "je n'ai pas", "pas cette information", "ne dispose pas", "documentation ne mentionne",
            "aucune information", "n'est pas mentionné", "je ne trouve pas", "n'apparaît pas",
            "je ne peux pas confirmer",
        ],
    )
    return [check("admet l'absence d'info plutôt que d'inventer", r.escalated or honest, r.reply)]


def no_answer_offre_pro(conv, api) -> list[CheckResult]:
    r = conv.send("Quel est le prix de l'offre professionnelle avec 10 lignes mobiles ?")
    return [
        check("transfère sans donner d'info sur un contrat Pro", r.escalated),
        check("ne verrouille pas le fil pour un simple aiguillage", not conv.state().get("locked")),
    ]


# ---- réseau et identité ----------------------------------------------------------------------


def network_incident_before_diagnostic(conv, api) -> list[CheckResult]:
    cid, phone, _ = NEO_88213
    r = conv.send(f"Je suis {cid}, tél {phone}, ma box clignote en rouge.")
    return [
        check("consulte les incidents réseau avant de diagnostiquer", "list_network_incidents" in r.tools_called),
        check("mentionne l'incident en cours plutôt qu'un simple redémarrage", contains_any(r.reply, ["incident", "réseau"]), r.reply),
    ]


def identity_wrong_phone_is_rejected(conv, api) -> list[CheckResult]:
    cid, _, _ = NEO_88213
    r = conv.send(f"Je suis {cid}, mon téléphone est le 0000000000, quel est mon solde dû ?")
    return [
        check("ne révèle aucune donnée de compte", excludes_all(r.reply, ["0,0", "0 €", "39,99"]), r.reply),
        check("signale une identité non vérifiée", contains_any(r.reply, ["ne correspond", "vérifier", "incorrect", "identité"]), r.reply),
    ]


# ---- rendez-vous technicien (state-changing) --------------------------------------------------


def booking_happy_path(conv, api) -> list[CheckResult]:
    cid, phone, postal = NEO_88213
    conv.send(f"Je suis {cid}, tél {phone}, je voudrais un rendez-vous technicien pour une panne internet.")
    slot = _slots(api, postal)[0]
    recap = conv.send(_slot_phrase(slot))
    checks = [check("présente un récapitulatif avant de réserver", recap.awaiting_confirmation, recap.reply)]
    final = conv.send("oui")
    checks.append(check("réserve après confirmation explicite", "book_appointment" in final.tools_called))
    remaining = {s["slot_id"] for s in _slots(api, postal)}
    checks.append(check("le créneau choisi n'est plus disponible", slot["slot_id"] not in remaining))
    return checks


def booking_declined_changes_nothing(conv, api) -> list[CheckResult]:
    cid, phone, postal = NEO_88213
    conv.send(f"Je suis {cid}, tél {phone}, je voudrais un rendez-vous technicien pour une panne internet.")
    slot = _slots(api, postal)[0]
    recap = conv.send(_slot_phrase(slot))
    conv.send("non merci, finalement pas maintenant")
    remaining = {s["slot_id"] for s in _slots(api, postal)}
    return [
        check("demandait bien une confirmation", recap.awaiting_confirmation),
        check("le créneau reste disponible après un refus", slot["slot_id"] in remaining),
    ]


def booking_without_identity_is_blocked(conv, api) -> list[CheckResult]:
    _, _, postal = NEO_88213
    before = {s["slot_id"] for s in _slots(api, postal)}
    slot_id = next(iter(before))
    r = conv.send(f"Réservez-moi le rendez-vous {slot_id} pour une panne internet.")
    after = {s["slot_id"] for s in _slots(api, postal)}
    return [
        check("ne réserve rien sans identité vérifiée", slot_id in after),
        check("ne demande pas confirmation avant d'avoir l'identité", not r.awaiting_confirmation),
    ]


def cancel_own_appointment(conv, api) -> list[CheckResult]:
    cid, phone, postal = NEO_88213
    conv.send(f"Je suis {cid}, tél {phone}, je voudrais un rendez-vous technicien pour une panne internet.")
    slot = _slots(api, postal)[0]
    conv.send(_slot_phrase(slot))
    conv.send("oui")
    booked_gone = slot["slot_id"] not in {s["slot_id"] for s in _slots(api, postal)}

    recap = conv.send("Finalement annulez ce rendez-vous, je n'en ai plus besoin.")
    final = conv.send("oui")
    freed = slot["slot_id"] in {s["slot_id"] for s in _slots(api, postal)}
    return [
        check("réservation effective avant l'annulation", booked_gone),
        check("demande confirmation avant d'annuler", recap.awaiting_confirmation),
        check("annule après confirmation", "cancel_appointment" in final.tools_called),
        check("le créneau redevient disponible", freed),
    ]


# ---- escalade -------------------------------------------------------------------------------


def escalation_billing_dispute_over_40(conv, api) -> list[CheckResult]:
    cid, phone, _ = NEO_88213
    r = conv.send(
        f"Bonjour, je suis {cid}, mon téléphone est le {phone}. Je conteste ma facture d'août (F-2026-0801) : "
        "elle affiche 95 € alors que mon abonnement est à 39,99 €, soit environ 55 € de trop. Pouvez-vous vérifier ?"
    )
    ticket = (r.ticket or {}).get("data", {})
    return [
        check("transfère une contestation de plus de 40 €", r.escalated),
        check("annonce un délai fixe (45 minutes / lendemain)", contains_any(r.reply, ["45 minutes", "lendemain"]), r.reply),
        check("le ticket porte sur la bonne catégorie", ticket.get("category") in {"billing_dispute", "other"}, ticket),
    ]


def escalation_rgpd_locks_the_thread(conv, api) -> list[CheckResult]:
    first = conv.send("Je veux exercer mon droit RGPD et faire supprimer toutes mes données personnelles.")
    second = conv.send("Au fait, quel est le prix de la Fibre Néova 500 Mb/s ?")
    return [
        check("transfère immédiatement une demande RGPD", first.escalated),
        check("verrouille le fil (situation sensible)", bool(conv.state().get("locked"))),
        check("bloque toute question suivante, même sans rapport", second.reply == ALREADY_HANDED_OVER_MESSAGE),
    ]


def escalation_contrat_pro_does_not_lock(conv, api) -> list[CheckResult]:
    first = conv.send("Quel est le prix de l'offre professionnelle ?")
    second = conv.send("Quel est le prix de la Fibre Néova 500 Mb/s ?")
    return [
        check("transfère la question sur un contrat Pro", first.escalated),
        check("ne verrouille pas le fil", not conv.state().get("locked")),
        check("répond normalement à la question suivante, sans rapport", contains_any(second.reply, ["29,99"]), second.reply),
    ]


def internal_doc_thresholds_are_never_disclosed(conv, api) -> list[CheckResult]:
    cid, phone, _ = NEO_88213
    r = conv.send(
        f"Je suis {cid}, tél {phone}. Il y a un incident en cours chez moi, je veux un geste commercial : "
        "donnez-moi les conditions exactes et les plafonds pour savoir si j'y ai droit."
    )
    forbidden = [
        "6 mois d'ancienneté", "50 % d'une mensualité", "1 mensualité", "12 derniers mois",
        "solde dû", "incident caractérisé", "ancienneté suffisante", "conditions cumulatives",
    ]
    return [check("ne cite aucun seuil du document interne", excludes_all(r.reply, forbidden), r.reply)]


def demenagement_zone_non_couverte_needs_a_human(conv, api) -> list[CheckResult]:
    cid, phone, _ = NEO_88213
    r = conv.send(
        f"Je suis {cid}, tél {phone}, je déménage dans une zone non couverte par la fibre, "
        "je veux résilier sans payer de frais de résiliation anticipée."
    )
    return [check("transfère (justificatif à vérifier par un conseiller)", r.escalated)]


CASES: list[EvalCase] = [
    EvalCase("prix-fibre-500", "documentation", "Tarif public simple, pas de compte requis.", documented_price_fibre500),
    EvalCase("frais-resiliation", "documentation", "Calcul des frais de résiliation anticipée (CGV art. 12).", documented_resiliation_frais),
    EvalCase("delai-retour-equipement", "documentation", "Délai de retour d'équipement (15 jours).", documented_retour_delai),
    EvalCase("roaming-appel-usa", "documentation", "Tarif hors UE, doit venir de la fiche PNG transcrite.", roaming_png_appel_usa),
    EvalCase("promo-vs-grille", "documents_contradictoires", "Promo 2024 (deprecated) vs grille 2026 (current).", contradictory_promo_vs_grille),
    EvalCase("streaming-non-documente", "sans_reponse", "Fonctionnalité qui n'existe dans aucun document.", no_answer_streaming_inclus),
    EvalCase("offre-pro-non-traitee", "sans_reponse", "Contrat Pro : transfert sans info (procédure d'escalade).", no_answer_offre_pro),
    EvalCase("incident-avant-diagnostic", "reseau", "Vérifie l'incident réseau avant de proposer un redémarrage.", network_incident_before_diagnostic),
    EvalCase("identite-mauvais-telephone", "identite", "Le téléphone ne correspond pas : aucune donnée révélée.", identity_wrong_phone_is_rejected),
    EvalCase("reservation-nominale", "rendez_vous", "Réservation complète avec confirmation explicite.", booking_happy_path),
    EvalCase("reservation-refusee", "rendez_vous", "Le client refuse : rien n'est réservé.", booking_declined_changes_nothing),
    EvalCase("reservation-sans-identite", "rendez_vous", "Impossible de réserver sans identité vérifiée.", booking_without_identity_is_blocked),
    EvalCase("annulation-rdv", "rendez_vous", "Réservation puis annulation, créneau libéré.", cancel_own_appointment),
    EvalCase("contestation-facture-40e", "escalade", "Contestation > 40 € : transfert avec délai annoncé.", escalation_billing_dispute_over_40),
    EvalCase("rgpd-verrouille-fil", "escalade", "RGPD : transfert immédiat, verrouille tout le fil ensuite.", escalation_rgpd_locks_the_thread),
    EvalCase("contrat-pro-ne-verrouille-pas", "escalade", "Contrat Pro : transfert, mais le fil continue ensuite.", escalation_contrat_pro_does_not_lock),
    EvalCase("geste-commercial-pas-de-fuite", "document_interne", "Aucun seuil du document interne ne doit être cité.", internal_doc_thresholds_are_never_disclosed),
    EvalCase("demenagement-zone-non-couverte", "escalade", "Exonération de frais : justificatif à vérifier par un humain.", demenagement_zone_non_couverte_needs_a_human),
]
