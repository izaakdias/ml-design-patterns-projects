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
        "o boleto da mensalidade veio com o valor errado",
        "a fatura do cartão tem uma cobrança que não autorizei",
        "solicito reembolso do pagamento cobrado em duplicidade",
        "preciso corrigir a data de vencimento do boleto",
        "o valor da assinatura foi debitado duas vezes",
        "não recebi a confirmação do pagamento da fatura",
        "quero cancelar a cobrança adicional na mensalidade",
        "o desconto prometido não apareceu na minha fatura",
        "o boleto já foi pago mas consta outra cobrança",
        "solicito o comprovante de quitação da mensalidade",
        "a cobrança do cartão continuou depois do cancelamento",
        "preciso parcelar o valor pendente da fatura",
    ],
    "tecnico": [
        "o aplicativo fecha quando tento fazer o pagamento",
        "não consigo acessar o painel porque aparece um erro",
        "a conexão cai sempre que abro a plataforma",
        "a tela fica carregando e não conclui a operação",
        "a última atualização deixou o aplicativo muito lento",
        "o aplicativo mostra erro ao abrir a tela principal",
        "a plataforma está lenta e trava no carregamento",
        "há um erro de conexão no sistema",
        "o aplicativo para de funcionar durante a operação",
        "a tela de configurações não carrega na plataforma",
        "o sistema apresenta uma falha depois da atualização",
        "não consigo instalar a atualização do aplicativo",
        "a conexão com a plataforma fica instável",
        "o aplicativo exibe uma mensagem de erro inesperado",
        "a tela do sistema congela e preciso reiniciar",
        "a atualização causou uma falha no carregamento",
        "o aplicativo perde a conexão durante o uso",
    ],
    "conta": [
        "preciso atualizar o email cadastrado no perfil",
        "quero recuperar o acesso à minha conta",
        "meu perfil foi bloqueado e não consigo entrar",
        "preciso alterar o telefone associado ao cadastro",
        "não recebi o código para confirmar minha identidade",
        "quero redefinir a senha de acesso ao perfil",
        "preciso trocar o telefone no cadastro da conta",
        "meu email de cadastro está incorreto",
        "não consigo recuperar a senha da conta",
        "quero atualizar os dados pessoais do perfil",
        "o código de verificação da identidade expirou",
        "preciso desbloquear meu cadastro para entrar na conta",
        "o email de recuperação da senha não chegou",
        "quero mudar o nome cadastrado no meu perfil",
        "o telefone da conta já não é meu",
        "preciso confirmar minha identidade para recuperar o perfil",
        "a senha do cadastro não permite o acesso à conta",
    ],
    "entrega": [
        "meu pedido ainda não chegou no endereço informado",
        "o pacote aparece como entregue mas não recebi",
        "preciso mudar o endereço de entrega do pedido",
        "quero saber quando a encomenda será enviada",
        "o rastreamento não mostra nenhuma atualização",
        "a entrega da encomenda passou do prazo previsto",
        "o endereço do pedido precisa ser corrigido",
        "o pacote foi enviado para o endereço errado",
        "quero acompanhar o rastreamento da encomenda",
        "a transportadora não entregou o meu pedido",
        "preciso reagendar a entrega do pacote",
        "a encomenda consta parada no rastreamento",
        "o prazo de envio do pedido terminou",
        "o pacote não foi entregue pela transportadora",
        "preciso informar um novo endereço para a encomenda",
        "a entrega do pedido foi marcada sem que o pacote chegasse",
        "quero confirmar a data de chegada da encomenda",
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
                            "phrase_group": f"{intent}-{phrase_index}",
                            "example_id": f"{intent}-{phrase_index}-{context_index}-{ending_index}",
                        }
                    )
    frame = pd.DataFrame(rows)
    return frame.sample(frac=1.0, random_state=41).reset_index(drop=True)
