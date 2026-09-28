# Néova Télécom — agent de relation client

Agent conversationnel (LangGraph) pour le service client résidentiel de Néova Télécom (FAI/mobile
fictif). Répond aux questions courantes à partir de la documentation interne, consulte les données
client via une API FastAPI dédiée, peut réserver un rendez-vous technicien, et sait transférer un
dossier à un conseiller humain quand la situation le demande.

Construit pour le take-home *AI Engineer, Customer Relations Agent*.

---

## 1. Installation locale

Prérequis : **Python 3.11+**, un terminal (PowerShell sous Windows), une clé API OpenRouter valide.

```powershell
# 1. Cloner et se placer dans le dossier du projet
git clone <url-du-repo>
cd neova

# 2. Créer et activer un environnement virtuel
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux : source .venv/bin/activate

# 3. Installer le projet (dépendances runtime + dev : pytest, httpx)
pip install -e ".[dev]"

# 4. Configurer les secrets et les modèles
# .env.example est le modèle versionné dans le repo : ne jamais le renommer ni le modifier.
# On en fait une copie nommée .env (ignorée par git), et c'est DANS .env qu'on configure tout.
Copy-Item .env.example .env         # macOS/Linux : cp .env.example .env
# Ouvrir .env et y renseigner directement : OPENROUTER_API_KEY, CHAT_MODEL, EMBEDDING_MODEL (voir §7)
```

### Lancer le service

Deux processus séparés, dans deux terminaux (le premier lancement transcrit l'image du corpus et
construit l'index vectoriel : quelques secondes ; les suivants réutilisent le cache dans `.cache/`).

```powershell
# Terminal 1 — API métier (clients, incidents, créneaux, tickets)
.venv\Scripts\python.exe -m uvicorn api.main:app

# Terminal 2 — agent conversationnel en ligne de commande
.venv\Scripts\python.exe -m agent.cli
```

La CLI affiche une invite `Vous >` ; chaque réponse indique les tools appelés et signale un
transfert vers un conseiller. `Ctrl+C` pour quitter et voir la consommation de tokens.

### Lancer les tests

```powershell
.venv\Scripts\python.exe -m pytest tests\ -v
```

60 tests (API, ingestion du corpus, retrieval, graphe de l'agent, socle d'évaluation), tous hors
ligne — aucun appel réseau, un LLM scripté remplace les vrais modèles.

### Lancer l'évaluation

```powershell
.venv\Scripts\python.exe -m evaluation.run_eval
```

Rejoue 18 scénarios contre les vrais modèles OpenRouter (coût estimé : quelques centimes) et écrit
le détail dans `evaluation/results.md`.

---

## 2. Architecture du graphe

Graphe LangGraph : **TRIAGE** (détecte les 7 situations à transfert immédiat de la procédure
d'escalade) → **AGENT** (LLM + tools : documentation, données client, réseau, créneaux) →
**CONFIRM** (interrompt le graphe avant toute réservation/annulation, récapitulatif déterministe) →
**TOOLS** → **VERIFY** (rejette toute réponse non appuyée par un résultat d'outil, tarif obsolète
présenté comme actuel, ou fuite d'un document interne) → réponse, ou **ESCALATE** (ticket créé,
délai de rappel fixe annoncé). Schéma complet et détail de chaque nœud dans
[`docs/architecture.md`](docs/architecture.md).

Garde-fous codés en dur plutôt que confiés au LLM : identité vérifiée avant toute donnée ou action
de compte, confirmation explicite avant réservation/annulation, réponse rejetée deux fois =
transfert (jamais envoyée), délai de rappel constant, verrouillage du fil réservé aux seules
situations sensibles (RGPD, fraude, décès, procédure judiciaire, mineur/protégé, détresse).

---

## 3. Résultats d'évaluation

*(à compléter après un run `python -m evaluation.run_eval` avec une clé OpenRouter valide — voir
`evaluation/results.md` pour le détail par catégorie et l'analyse des échecs. Non exécuté à ce
stade : la clé fournie pour ce take-home était invalide/expirée, en attente de résolution.)*

| Catégorie | Score |
|---|---|
| documentation | — |
| documents_contradictoires | — |
| sans_reponse | — |
| reseau | — |
| identite | — |
| rendez_vous | — |
| escalade | — |
| document_interne | — |
| **Total** | **—/18** |

Les vérifications sont déterministes (sous-chaînes attendues, tools réellement appelés, état de
l'API après coup) plutôt que confiées à un LLM juge : plus fiable et moins cher, au prix d'une
tolérance plus faible aux reformulations imprévues du modèle (voir §5).

---

## 4. Trois décisions de conception

**1. Vector store en mémoire, pas de base vectorielle dédiée.**
`InMemoryVectorStore` de LangChain, sauvegardé sur disque en JSON. Le corpus ne fait que 55-60
chunks : une recherche exhaustive est instantanée, et ça évite tout service supplémentaire pour la
commande de lancement.
*Sacrifié* : passage à l'échelle (des dizaines de milliers de chunks) et partage de l'index entre
plusieurs processus. `KnowledgeBase.search()` masque le store, donc migrer vers Chroma/Qdrant ne
toucherait que `build_knowledge_base`.

**2. Un LLM et un SLM séparés (`CHAT_MODEL` / `JUDGE_MODEL`).**
Le dialogue (nœud AGENT) utilise un modèle plus capable ; les décisions de classification (TRIAGE,
VERIFY, lecture de la confirmation du client) tournent sur un modèle plus petit et moins cher, sans
changement de code — juste une variable d'environnement différente.
*Sacrifié* : un SLM classifie moins finement qu'un LLM sur des cas ambigus ; à surveiller dans
l'analyse d'erreurs de l'évaluation.

**3. Verrouillage sélectif après escalade, pas un blocage systématique.**
Seuls les motifs réellement sensibles (RGPD, fraude, décès, procédure judiciaire, mineur/protégé,
détresse) bloquent tout traitement automatique du fil. Un aiguillage hors périmètre (contrat Pro)
ou un échec technique n'empêche pas de répondre à une question suivante sans rapport.
*Sacrifié* : plus simple (et plus prudent) de tout bloquer après la moindre escalade ; ce choix
demande de maintenir correctement la liste des situations sensibles dans le prompt de TRIAGE.

---

## 5. Avec deux jours de plus

1. Dockerfile + `docker-compose.yml` pour la commande unique demandée par l'énoncé.
2. Calibrer `min_score` avec un jeu de questions hors corpus, une fois les vrais embeddings mesurés.
3. LLM-as-judge en complément des vérifications déterministes de l'évaluation, sur les cas
   qualitatifs (groundedness fine, refus bien formulé, résistance à l'injection via le corpus).
4. Élargir le jeu d'évaluation (18 cas → plus de variantes par catégorie, notamment les cas
   d'injection via un document ou une réponse d'outil).

---

## 6. Modèles utilisés

| Rôle | Modèle | Pourquoi |
|---|---|---|
| `CHAT_MODEL` | `openai/gpt-4.1-mini` | Tool calling fiable, bon français, gère les images (transcription de la fiche roaming scannée). |
| `EMBEDDING_MODEL` | `baai/bge-m3` | Multilingue (le corpus est entièrement en français). |
| `JUDGE_MODEL` | *(à définir, ex. `mistralai/ministral-8b-2512`)* | Classification structurée (triage, vérification) : un modèle plus petit suffit et réduit le coût du juge, appelé plusieurs fois par tour. |

Prix et disponibilité vérifiés sur `https://openrouter.ai/api/v1/models` au moment du choix. Suivi
de la consommation via `UsageTracker` (affiché en fin de session CLI et dans le rapport
d'évaluation) et via :

```bash
curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  | jq '.data | {usage, limit, limit_remaining}'
```


