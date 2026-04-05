# Py.pou

**Bibliothèque Python minimaliste pour intégrer les APIs IA en une ligne.**

Interface unifiée pour **OpenAI**, **Anthropic** et **Mistral** — sans changer votre code quand vous changez de provider.

---

## Installation

```bash
pip install aipou
# ou depuis les sources :
pip install -e .
```

---

## Démarrage rapide

```python
from aipou import AIClient

# --- OpenAI ---
client = AIClient("openai", api_key="sk-...")
response = client.chat("Explique-moi les LLMs en 3 phrases.")
print(response)           # AIResponse se convertit automatiquement en str

# --- Anthropic Claude ---
client = AIClient("anthropic", api_key="sk-ant-...")
response = client.chat("Explique-moi les LLMs en 3 phrases.")
print(response.content)   # accès direct au texte

# --- Mistral ---
client = AIClient("mistral", api_key="...")
response = client.chat("Explique-moi les LLMs en 3 phrases.")
print(response.usage)     # TokenUsage(prompt=12, completion=87, total=99)
```

---

## Fonctionnalités

### Chat simple

```python
response = client.chat("Bonjour !")
print(response.content)       # texte de la réponse
print(response.model)         # ex. "gpt-4o"
print(response.provider)      # ex. "openai"
print(response.usage)         # TokenUsage
print(response.finish_reason) # "stop", "length", …
```

### Streaming

```python
# Itération sur des StreamChunk
for chunk in client.stream("Raconte une histoire courte."):
    print(chunk.delta, end="", flush=True)

# Ou directement sur des strings
for text in client.stream_text("Raconte une histoire courte."):
    print(text, end="", flush=True)
```

### Conversations multi-tours

```python
from aipou import ChatMessage

history = [
    ChatMessage.user("Qui a créé Python ?"),
    ChatMessage.assistant("Python a été créé par Guido van Rossum en 1991."),
]
response = client.chat("Et quel est son langage favori ?", history=history)
print(response)
```

### Message système

```python
response = client.chat(
    "Raconte une blague.",
    system="Tu es un assistant pince-sans-rire qui répond toujours en vers.",
)
print(response)
```

### Choisir un modèle spécifique

```python
# Au niveau du client (par défaut pour toutes les requêtes)
client = AIClient("openai", api_key="sk-...", model="gpt-4-turbo")

# Ou par requête
response = client.chat("Bonjour", model="gpt-3.5-turbo")
```

### Suivi des tokens

```python
client.chat("Requête 1")
client.chat("Requête 2")
print(client.session_usage)
# TokenUsage(prompt=45, completion=123, total=168)
```

### Retry automatique

Par défaut, `AIClient` réessaie jusqu'à **3 fois** en cas de rate limit (HTTP 429),
avec backoff exponentiel.

```python
client = AIClient("openai", api_key="sk-...", max_retries=5, retry_delay=2.0)
```

### Décorateurs utilitaires (aipou.utils)

```python
from aipou.utils import cache_response, retry_on_failure, log_call

@cache_response
def appel_couteux(**kwargs):
    return client.chat(**kwargs)

@retry_on_failure(max_retries=3, delay=0.5)
def appel_fragile():
    ...

@log_call
def mon_pipeline():
    ...
```

---

## Structure du projet

```
aipou/
├── __init__.py          # Exports publics
├── client.py            # AIClient — point d'entrée principal
├── models.py            # ChatMessage, AIResponse, StreamChunk, TokenUsage
├── exceptions.py        # Hiérarchie d'exceptions
├── utils.py             # Décorateurs : cache, retry, log
└── providers/
    ├── base.py          # BaseProvider (ABC)
    ├── openai.py        # OpenAIProvider
    ├── anthropic.py     # AnthropicProvider
    └── mistral.py       # MistralProvider
```

---

## Ajouter un nouveau provider

```python
# aipou/providers/mon_provider.py
from aipou.providers.base import BaseProvider
from aipou.models import ChatMessage, AIResponse, StreamChunk

class MonProvider(BaseProvider):
    def _default_model(self) -> str:
        return "mon-modele-v1"

    def chat(self, messages, *, model=None, **kwargs) -> AIResponse:
        ...  # Votre implémentation

    def stream(self, messages, *, model=None, **kwargs):
        ...  # Votre implémentation
```

Puis enregistrez-le dans `client.py` :
```python
from aipou.providers.mon_provider import MonProvider
_PROVIDER_REGISTRY["monprovider"] = MonProvider
```

---

## Gestion des erreurs

```python
from aipou import AIClient
from aipou.exceptions import AuthenticationError, RateLimitError, APIError

client = AIClient("openai", api_key="clé-invalide")

try:
    response = client.chat("Bonjour")
except AuthenticationError:
    print("Clé API invalide !")
except RateLimitError:
    print("Rate limit atteint — les retries ont été épuisés.")
except APIError as e:
    print(f"Erreur API : {e}")
```

---
