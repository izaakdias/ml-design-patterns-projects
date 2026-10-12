# Machine Learning Design Patterns — projetos práticos

[![Quality](https://github.com/izaakdias/ml-design-patterns-projects/actions/workflows/quality.yml/badge.svg)](https://github.com/izaakdias/ml-design-patterns-projects/actions/workflows/quality.yml)

Oito projetos autorais e executáveis que combinam padrões de ML ao longo do ciclo de vida: preparação dos dados, treinamento, avaliação e inferência. Todos usam dados sintéticos gerados localmente.

O capítulo final do livro conecta padrões ao ciclo de vida de projetos de ML, mas não traz uma lista de exercícios. Este repositório transforma essa conexão em laboratórios independentes. Os projetos são originais e não reproduzem os notebooks nem o código do livro.

## Projetos

| Projeto | Problema | Padrões praticados |
| --- | --- | --- |
| **churn** | Priorizar clientes para uma revisão humana de retenção | transformações reproduzíveis, divisão estratificada, benchmark heurístico, reequilíbrio, avaliação por grupos, explicação global e serviço sem estado |
| **demand** | Prever demanda horária e atender previsões em janelas | features temporais sem vazamento, divisão cronológica, benchmark sazonal, inferência em janela, avaliação contínua e monitoramento de drift |
| **support** | Encaminhar solicitações escritas para uma equipe | benchmark simples, cascata de modelos, classe neutra para revisão humana, controle de confiança e serviço sem estado |
| **fraud** | Priorizar transações para análise de risco | classificação de evento raro, benchmark heurístico, limiares calibrados em validação, filas em dois níveis e revisão humana |
| **vision** | Inspecionar imagens sintéticas de peças com defeito | pré-treinamento, adaptação entre domínios, checkpoints, quantização, cascata de modelos e revisão manual |
| **recommend** | Encontrar e recomendar conteúdo de um catálogo | embeddings de interações, busca TF-IDF, recomendação híbrida, cruzamento de features, resultados identificados por usuário e avaliação HitRate@10 |
| **sensors** | Detectar anomalias em janelas de telemetria | detecção não supervisionada, limiar calibrado em dados normais, features por janela, inferência com estado e explicações por desvio |
| **mlops** | Reproduzir o ciclo de vida de um modelo de manutenção | validação de dados, divisão repetível, transformação e modelo em pipeline, seleção de limiar, métricas por planta, explicação global e registro de versões |

As métricas dos dados sintéticos demonstram o fluxo, não estimam desempenho em produção. O projeto de churn é um exemplo educacional de priorização: não deve ser usado para negar serviços ou automatizar decisões sobre pessoas. Métricas por grupo ajudam a investigar diferenças, mas não comprovam que um modelo seja justo.

## Requisitos e instalação

Requer Python 3.11 ou mais recente. Crie um ambiente virtual e instale o pacote:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[api]"

O extra api instala FastAPI e Uvicorn. Para desenvolvimento e testes, use o extra dev junto com api. Os modelos e metadados são gravados em artifacts/, ignorada pelo Git.

Para reproduzir as dependências registradas em uv.lock, instale [uv](https://docs.astral.sh/uv/getting-started/installation/) e execute:

    uv sync --locked --extra api --extra dev

Se uv foi instalado dentro da própria .venv, acrescente --inexact para preservar essa ferramenta no ambiente. A instalação com pip aceita as faixas do pyproject; a instalação com uv usa as versões do lockfile para a versão de Python escolhida.

## Arquitetura e qualidade

O [guia de arquitetura](docs/architecture.md) explica os limites entre domínio, treino, HTTP e registro, junto com as correções de avaliação. O [guia de projetos](docs/projects.md) traz os corpos das requisições e as métricas de cada laboratório. A [execução de validação](docs/validation.md) registra a evidência local.

Depois de instalar os extras api e dev, execute:

    make check

A suíte treina os oito projetos em um registro temporário, testa suas APIs e verifica invariantes de dados, políticas e publicação de modelos. Ruff verifica estilo e problemas estáticos. O workflow Quality repete os checks em Python 3.11 e 3.14 e constrói os pacotes.

## Treinar os projetos

Execute a partir da raiz do repositório. A CLI central permite um projeto ou todos:

    python -m ml_patterns_lab train all
    python -m ml_patterns_lab train fraud

Os comandos individuais também continuam disponíveis:

    python -m ml_patterns_lab.churn.train
    python -m ml_patterns_lab.demand.train
    python -m ml_patterns_lab.support.train
    python -m ml_patterns_lab.fraud.train
    python -m ml_patterns_lab.vision.train
    python -m ml_patterns_lab.recommend.train
    python -m ml_patterns_lab.sensors.train
    python -m ml_patterns_lab.mlops.run

Cada execução cria uma versão completa com modelo, metadados e manifesto SHA-256; então atualiza atomicamente artifacts/<projeto>/latest.json. Checkpoints de visão não são promovidos automaticamente. Modelos tabulares calibram probabilidades dentro do treino e escolhem limiares numa validação separada.

É possível listar versões, selecionar uma ao servir e promover uma versão anterior:

    python -m ml_patterns_lab list churn
    python -m ml_patterns_lab serve churn --port 8001 --model-version RUN_ID
    python -m ml_patterns_lab promote churn RUN_ID

Use MLDP_ARTIFACTS_DIR ou --artifacts-dir antes do subcomando para escolher outro registro. MLDP_MODEL_VERSION fixa uma versão para as APIs iniciadas diretamente com Uvicorn. Uma API já iniciada mantém o modelo carregado até ser reiniciada. Artefatos da versão anterior sem manifesto precisam ser retreinados.

## Servir previsões

Treine o projeto desejado antes de iniciar sua API. Pela CLI central:

    python -m ml_patterns_lab serve fraud --port 8004

Ou inicie diretamente com Uvicorn:

    uvicorn ml_patterns_lab.churn.api:app --host 127.0.0.1 --port 8001
    uvicorn ml_patterns_lab.demand.api:app --host 127.0.0.1 --port 8002
    uvicorn ml_patterns_lab.support.api:app --host 127.0.0.1 --port 8003
    uvicorn ml_patterns_lab.fraud.api:app --host 127.0.0.1 --port 8004
    uvicorn ml_patterns_lab.vision.api:app --host 127.0.0.1 --port 8005
    uvicorn ml_patterns_lab.recommend.api:app --host 127.0.0.1 --port 8006
    uvicorn ml_patterns_lab.sensors.api:app --host 127.0.0.1 --port 8007
    uvicorn ml_patterns_lab.mlops.api:app --host 127.0.0.1 --port 8008

Cada API oferece GET /health e documentação interativa em /docs. As APIs de classificação e previsão oferecem POST /predict. A API recommend oferece POST /recommend, POST /recommend/batch e POST /search.

As requisições usam JSON. O endpoint de churn recebe tenure_months, monthly_spend, support_tickets_90d, plan e region. O de demand recebe uma lista history com pelo menos 24 valores horários, last_timestamp em ISO 8601 e um horizon de 1 a 48 horas. O de support recebe text. O de fraud recebe transaction_id, amount, transactions_last_hour, merchant_risk_score, distance_from_usual_km, is_new_device e is_cross_border. O de vision recebe pixels, uma lista de 576 valores de escala de cinza entre 0 e 1. O de sensors recebe readings, uma lista de observações com temperature_c, vibration_mm_s e pressure_kpa. O de mlops recebe hours_since_service, temperature_c, vibration_mm_s, load_percent e machine_type.

Exemplo de solicitação para support:

    {"text": "Minha fatura mostra uma cobrança duplicada. Podem verificar?"}

Exemplo de recomendação:

    {"user_id": "user-001", "limit": 10}

A página /docs mostra os esquemas completos e permite enviar exemplos.

Para comparar a distribuição recente de demanda com a referência sintética:

    python -m ml_patterns_lab.demand.monitor

Para reproduzir um fluxo simples de telemetria com janelas móveis:

    python -m ml_patterns_lab.sensors.monitor

O índice PSI de demand é apenas um sinal de triagem; o limite configurado é heurístico e precisa ser calibrado para dados e custos reais. Fraud, vision e support oferecem encaminhamento para revisão humana. Sensors informa alertas e desvios para investigação.

## Estrutura

    src/ml_patterns_lab/
      churn/       classificação tabular e avaliação por grupos
      demand/      previsão temporal, inferência em janela e drift
      support/     classificação de texto e encaminhamento em cascata
      fraud/       evento raro e triagem em filas de revisão
      vision/      adaptação de domínio, quantização e inspeção em cascata
      recommend/   recomendação híbrida, busca e features compartilhadas
      sensors/     detecção de anomalias em janelas de telemetria
      mlops/       fluxo repetível de treino, avaliação e registro
      shared.py    registro, integridade e promoção de versões
      serving.py   ciclo de vida HTTP e contratos compartilhados
      tabular.py   classificadores calibrados e coeficientes globais
      evaluation.py métricas e políticas de limiar
      __main__.py  CLI para treino, serviço, monitoramento e versões
    tests/         regressões e integração com modelos reais
    docs/          arquitetura, projetos e validação
    uv.lock        dependências resolvidas por versão de Python

## Referências

Ideias organizadas a partir de *Machine Learning Design Patterns*, de Valliappa Lakshmanan, Sara Robinson e Michael Munn, O'Reilly, 2020. O código deste repositório foi escrito para estes laboratórios e é independente do código que acompanha o livro.

O material de referência e os exemplos oficiais estão em https://github.com/GoogleCloudPlatform/ml-design-patterns.

## Licença

Este repositório está sob a licença MIT. O livro e o repositório oficial são obras de seus respectivos autores e titulares.
