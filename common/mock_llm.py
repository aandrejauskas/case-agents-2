"""Mocks para simular custo e latência de chamadas a LLMs e execução de tools.

Usado tanto pelo caminho FAST_PATH (resposta local, sem custo de LLM) quanto pelo
caminho AGENT (seleção de tools + execução de tool + 1 chamada de LLM "cara").

Os valores de custo/latência são ilustrativos, só para o harness conseguir comparar
o pipeline "inteligente" (router + seleção de tools) contra o baseline de mandar tudo
para um LLM caro com todas as tools no prompt.
"""
import random
import time

random.seed(42)

# Custos aproximados em USD por operação
COST_ROUTER_USD = 0.0000005          # classificador local, custo desprezível
COST_RETRIEVAL_USD = 0.0000010       # seleção de tools local, custo desprezível
COST_FAST_PATH_USD = 0.0             # resposta local, sem LLM
COST_AGENT_LLM_CALL_USD = 0.010      # 1 chamada a LLM já com só top-k tools no contexto
COST_BASELINE_LLM_CALL_USD = 0.030   # chamada enviando TODAS as tools + prompt maior

# Latências simuladas (segundos) — pequenas para o harness rodar rápido
_LATENCY_AGENT_LLM = (0.030, 0.070)
_LATENCY_BASELINE_LLM = (0.060, 0.120)

_FAQ_ANSWERS = {
    "horario": "Nosso atendimento funciona de segunda a sábado, das 8h às 20h.",
    "taxa": "A manutenção da conta é gratuita para clientes com movimentação mensal.",
    "cashback": "O cashback devolve até 1% do valor das compras no cartão de crédito.",
    "document": "Para abrir conta você precisa de CPF, RG ou CNH e comprovante de endereço.",
    "telefone": "Nosso SAC pode ser encontrado no verso do seu cartão ou no app.",
    "conta": "Você pode abrir uma conta digital gratuitamente pelo nosso aplicativo.",
}


def fast_path_answer(query: str) -> str:
    """Retorna uma resposta 'de prateleira' sem chamar nenhum LLM."""
    q = query.lower()
    for keyword, answer in _FAQ_ANSWERS.items():
        if keyword in q:
            return answer
    if any(g in q for g in ("bom dia", "boa tarde", "boa noite", "oi", "olá", "obrigad", "tchau")):
        return "Olá! Como posso te ajudar hoje?"
    return "Obrigado pelo contato! Em breve retornaremos com mais detalhes."


def mock_tool_execution(tool_name: str, query: str) -> dict:
    """Simula a execução de uma tool (sem custo de LLM associado à execução)."""
    return {
        "tool": tool_name,
        "status": "success",
        "result": f"[MOCK] '{tool_name}' executada com sucesso para a query: {query!r}",
    }


def simulate_agent_llm_call(query: str, tool_name: str) -> dict:
    """Simula 1 chamada de LLM 'cara' já com o contexto reduzido (top-k tools)."""
    low, high = _LATENCY_AGENT_LLM
    time.sleep(random.uniform(low, high))
    return {
        "cost_usd": COST_AGENT_LLM_CALL_USD,
        "response": f"[LLM] Executando '{tool_name}' para responder: {query!r}",
    }


def simulate_baseline_llm_call(query: str) -> dict:
    """Simula mandar a query direto para o LLM caro, com TODAS as tools no prompt."""
    low, high = _LATENCY_BASELINE_LLM
    time.sleep(random.uniform(low, high))
    return {
        "cost_usd": COST_BASELINE_LLM_CALL_USD,
        "response": f"[LLM-BASELINE] Resposta completa para: {query!r}",
    }
