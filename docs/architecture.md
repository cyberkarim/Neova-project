# Architecture du graphe agentique

Graphe LangGraph implémenté dans [`agent/graph.py`](../agent/graph.py). État partagé : messages,
motif d'escalade en attente, indicateur de transfert effectué, critique de la vérification,
compteurs de tours d'outils et de tentatives de vérification.

```
                              START
                                │
                       escalated=true ?
                    ┌───────────┴───────────┐
                   oui                       non
                    │                         │
                    ▼                         ▼
             ┌──────────────┐          ┌──────────────┐
             │ HANDED_OVER  │          │    TRIAGE    │
             │ (dossier déjà│          │ (LLM, sortie │
             │  transféré)  │          │  structurée) │
             └──────┬───────┘          └──────┬───────┘
                    │                         │
                    │              escalade immédiate obligatoire ?
                    │            (RGPD, procédure judiciaire, décès,
                    │             fraude, contrat Pro, mineur/protégé,
                    │             détresse)
                    │                  ┌───────┴───────┐
                    │                 oui               non
                    │                  │                 │
                    │                  ▼                 ▼
                    │                  │           ┌─────────────┐
                    │                  │      ┌───▶│    AGENT    │◀──────────────┐
                    │                  │      │    │ (LLM + tools)│              │
                    │                  │      │    └──────┬──────┘              │
                    │                  │      │           │                     │
                    │                  │      │   tool_calls du dernier message │
                    │                  │      │           │                     │
                    │                  │      │  ┌────────┼─────────┬──────────┐│
                    │                  │      │  │        │         │          ││
                    │                  │      │ aucun  lecture  book/cancel  handover
                    │                  │      │  │    (docs, client,   │      demandé
                    │                  │      │  │   incidents...)     │      par le LLM
                    │                  │      │  ▼        │           ▼          │
                    │                  │      │ ┌──────┐  │    ┌────────────┐    │
                    │                  │      │ │VERIFY│  │    │  CONFIRM   │    │
                    │                  │      │ └──┬───┘  │    │(interrupt, │    │
                    │                  │      │    │      │    │ récap +    │    │
                    │                  │      │    │      │    │ 69 € si RDV)│    │
                    │                  │      │    │      │    └─────┬──────┘    │
                    │                  │      │    │      │    oui│      │non    │
                    │                  │      │    │      │       │      └──▶ AGENT
                    │                  │      │    │      ▼       ▼            │
                    │                  │      │    │  ┌─────────────┐          │
                    │                  │      └────┼──│    TOOLS    │          │
                    │                  │           │  │ (ToolNode)  │          │
                    │                  │           │  └─────────────┘          │
                    │                  │           │                          │
                    │                  │    ok ─────┴──▶ END                   │
                    │                  │    │                                  │
                    │                  │    │ rejeté (≤1 retry) ──▶ AGENT ──────┘
                    │                  │    │ rejeté 2 fois
                    │                  │    ▼
                    │                  │  ┌──────────────────────────┐
                    │                  └─▶│        ESCALATE          │
                    │                     │ create_escalation_ticket │
                    │                     │ + message de transfert   │
                    │                     │ fixe (45 min / lendemain)│
                    │                     └────────────┬─────────────┘
                    │                                  │
                    ▼                                  ▼
                   END                                END
```

## Les nœuds

- **HANDED_OVER** — court-circuite tout traitement si `locked=true` : plus aucun traitement
  automatique dans ce fil. Ne se déclenche que pour les motifs sensibles (voir TRIAGE) ; les autres
  escalades (hors périmètre, échec technique) ne verrouillent pas, la conversation continue
  normalement pour toute question sans rapport avec le motif déjà transféré.
- **TRIAGE** — LLM à sortie structurée (`Triage`), appelé une fois par tour avant l'agent. Détecte
  les sept cas de transfert immédiat et obligatoire de `procedure-escalade-n2` (RGPD, menace
  judiciaire, décès, fraude, contrat Pro, mineur/protégé, détresse). Ne déclenche que sur un signal
  net dans le message du client, jamais sur une simple contrariété. Seuls six de ces sept motifs
  posent `locked=true` : contrat Pro escalade sans verrouiller, car ce n'est qu'un aiguillage hors
  périmètre, pas une situation sensible.
- **AGENT** — LLM lié à ses tools (7 tools API + `search_knowledge_base` + `request_human_handover`).
  Décide de répondre, d'appeler un tool, ou de demander un transfert. Un compteur de tours d'outils
  (`tool_rounds`, plafond 8) force l'escalade si l'agent boucle sans conclure.
- **CONFIRM** — intercepte tout appel à `book_appointment` ou `cancel_appointment` avant exécution.
  Vérifie d'abord que l'identité du client a été confirmée par `get_customer` (et que le rendez-vous
  annulé lui appartient), puis génère un récapitulatif déterministe (créneau réel, motif, rappel des
  69 € si dégradation imputable au client) et suspend le graphe (`interrupt`) en attendant la réponse
  du client. Un second LLM classe la réponse (`confirm` / `decline` / `other`) ; dans le doute,
  l'action n'est jamais exécutée.
- **TOOLS** — `ToolNode` standard : exécute l'appel et boucle vers AGENT.
- **VERIFY** — LLM juge, appelé sur toute réponse finale (sans tool call). Rejette si un fait ou un
  chiffre n'est pas appuyé par les résultats d'outils du tour, si un tarif obsolète (promo 2024) est
  présenté comme actuel, si un seuil d'un document interne est cité au client, ou si une action est
  annoncée sans preuve qu'elle a eu lieu. Une réponse rejetée n'est jamais envoyée : elle est retirée
  de l'historique et régénérée avec la critique en contexte (1 nouvelle tentative), puis transférée
  si elle échoue encore.
- **ESCALATE** — point d'entrée unique vers `create_escalation_ticket`, quelle que soit la cause
  (triage, demande explicite de l'agent, boucle infinie, échec de vérification). Le message de
  transfert au client est fixe dans le code (rappel sous 45 minutes en heures ouvrées, sinon le
  lendemain matin) : le LLM ne promet jamais lui-même un délai.

## Garde-fous codés en dur (pas laissés au jugement du LLM)

- Aucune réservation ni annulation sans confirmation explicite au tour suivant.
- Aucune réservation/annulation sans identité vérifiée ; un client ne peut pas annuler le rendez-vous
  d'un autre.
- Le récapitulatif de confirmation (créneau, motif, frais) est construit à partir des vrais résultats
  d'outils, jamais rédigé librement par le LLM.
- Une réponse qui échoue deux fois la vérification est remplacée par un transfert, jamais envoyée
  telle quelle.
- Le délai annoncé au client lors d'un transfert est une constante, pas une génération du LLM.
