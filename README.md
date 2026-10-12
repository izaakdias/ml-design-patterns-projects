# Machine Learning Design Patterns — projetos práticos

Três projetos originais, pequenos e executáveis, que combinam padrões de ML ao longo do ciclo de vida: preparação dos dados, treinamento, avaliação e inferência. Todos os dados são sintéticos e gerados localmente.

O capítulo final do livro conecta padrões ao ciclo de vida de projetos de ML, mas não traz uma lista de exercícios. Este repositório transforma essa conexão em três laboratórios autorais; não reproduz os notebooks nem o código do livro.

## Projetos

| Projeto | Problema | Padrões praticados |
| --- | --- | --- |
| **churn** | Priorizar clientes para uma revisão humana de retenção | transformações reproduzíveis, divisão estratificada, benchmark heurístico, reequilíbrio, avaliação por grupos, explicação global e serviço sem estado |
| **demand** | Prever demanda horária e atender previsões em janelas | features temporais sem vazamento, divisão cronológica, benchmark sazonal, inferência em janela, avaliação contínua e monitoramento de drift |
| **support** | Encaminhar solicitações escritas para uma equipe | benchmark simples, cascata de modelos, classe neutra para revisão humana, controle de confiança e serviço sem estado |

As métricas dos dados sintéticos servem para demonstrar o fluxo, não para estimar desempenho em produção. O projeto de churn é um exemplo educacional de priorização: não deve ser usado para negar serviços ou automatizar decisões sobre pessoas. Métricas por grupo ajudam a investigar diferenças, mas não comprovam que um modelo seja justo.

## Requisitos e instalação

Requer Python 3.11 ou mais recente. Crie um ambiente virtual e instale o pacote:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[api]"

O extra api instala FastAPI e Uvicorn. Os modelos e metadados são gravados em artifacts/, ignorada pelo Git.

## Treinar os projetos

Execute a partir da raiz do repositório:

    python -m ml_patterns_lab.churn.train
    python -m ml_patterns_lab.demand.train
    python -m ml_patterns_lab.support.train

Cada execução cria uma versão de modelo com metadados e atualiza o ponteiro da versão atual em artifacts/<projeto>/latest.json.

## Servir previsões

Treine o projeto desejado antes de iniciar sua API:

    uvicorn ml_patterns_lab.churn.api:app --host 127.0.0.1 --port 8001
    uvicorn ml_patterns_lab.demand.api:app --host 127.0.0.1 --port 8002
    uvicorn ml_patterns_lab.support.api:app --host 127.0.0.1 --port 8003

Cada API oferece GET /health e POST /predict. A documentação interativa fica em /docs.

As requisições usam JSON. O endpoint de churn recebe tenure_months, monthly_spend, support_tickets_90d, plan e region. O de demand recebe uma lista history com pelo menos 24 valores horários, last_timestamp em ISO 8601 e um horizon de 1 a 48 horas. O de support recebe text; por exemplo:

    {"text": "Minha fatura mostra uma cobrança duplicada. Podem verificar?"}

A página /docs mostra os esquemas completos e permite enviar exemplos.

Para comparar a distribuição recente de demanda com a referência sintética:

    python -m ml_patterns_lab.demand.monitor

O índice PSI é usado apenas como sinal de triagem; o limite configurado é heurístico e precisa ser calibrado para dados e custos reais.

## Estrutura

    src/ml_patterns_lab/
      churn/       classificação tabular e avaliação por grupos
      demand/      previsão temporal, inferência em janela e drift
      support/     classificação de texto e encaminhamento em cascata
      shared.py    artefatos versionados e carregamento da versão atual

## Referências

Ideias organizadas a partir de *Machine Learning Design Patterns*, de Valliappa Lakshmanan, Sara Robinson e Michael Munn, O'Reilly, 2020. O código deste repositório foi escrito para estes laboratórios e é independente do código que acompanha o livro.

O material de referência e os exemplos oficiais estão em https://github.com/GoogleCloudPlatform/ml-design-patterns.

## Licença

Este repositório está sob a licença MIT. O livro e o repositório oficial são obras de seus respectivos autores e titulares.
