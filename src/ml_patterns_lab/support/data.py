"""Build a small, deterministic Portuguese-language support-ticket dataset."""

from __future__ import annotations

import pandas as pd


INTENT_PHRASES = {
    "cobranca": [
        "minha fatura mostra uma cobrança duplicada",
        "o pagamento foi feito e continua pendente",
        "quero contestar um valor lançado na conta",
        "não reconheço esta cobrança no meu cartão",
        "preciso da segunda via do boleto da assinatura",
    ],
    "tecnico": [
        "o aplicativo fecha quando tento fazer o pagamento",
        "não consigo acessar o painel porque aparece um erro",
        "a conexão cai sempre que abro a plataforma",
        "a tela fica carregando e não conclui a operação",
        "a última atualização deixou o aplicativo muito lento",
    ],
    "conta": [
        "preciso atualizar o email cadastrado no perfil",
        "quero recuperar o acesso à minha conta",
        "meu perfil foi bloqueado e não consigo entrar",
        "preciso alterar o telefone associado ao cadastro",
        "não recebi o código para confirmar minha identidade",
    ],
    "entrega": [
        "meu pedido ainda não chegou no endereço informado",
        "o pacote aparece como entregue mas não recebi",
        "preciso mudar o endereço de entrega do pedido",
        "quero saber quando a encomenda será enviada",
        "o rastreamento não mostra nenhuma atualização",
    ],
}

CONTEXTS = [
    "desde a semana passada",
    "ao tentar resolver pelo celular",
    "depois da última atualização",
    "e preciso de uma resposta",
]
ENDINGS = [
    "podem verificar por favor",
    "preciso de orientação para resolver",
    "não encontrei a solução no aplicativo",
]


def make_dataset() -> pd.DataFrame:
    rows = []
    for intent, phrases in INTENT_PHRASES.items():
        for phrase_index, phrase in enumerate(phrases):
            for context_index, context in enumerate(CONTEXTS):
                for ending_index, ending in enumerate(ENDINGS):
                    rows.append(
                        {
                            "text": f"{phrase} {context} {ending}",
                            "intent": intent,
                            "example_id": f"{intent}-{phrase_index}-{context_index}-{ending_index}",
                        }
                    )
    frame = pd.DataFrame(rows)
    return frame.sample(frac=1.0, random_state=41).reset_index(drop=True)
