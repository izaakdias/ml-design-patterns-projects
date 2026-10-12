# Guia dos laboratórios

Os comandos abaixo partem de um ambiente instalado na raiz do repositório. Todas as APIs oferecem /health e /docs. O comando serve usa a porta 8000 por padrão; altere --port para manter várias APIs abertas.

| Projeto | Comando de treino | Endpoint principal | Métricas registradas |
| --- | --- | --- | --- |
| churn | python -m ml_patterns_lab train churn | POST /predict | ROC-AUC, PR-AUC, Brier, precisão, recall, F1 e recortes por região |
| demand | python -m ml_patterns_lab train demand | POST /predict | MAE de uma hora, R² e backtest recursivo de 24 horas contra baseline sazonal |
| support | python -m ml_patterns_lab train support | POST /predict | Baseline, cobertura, acurácia dos encaminhados e parcelas por estágio |
| fraud | python -m ml_patterns_lab train fraud | POST /predict | PR-AUC, Brier, precisão, recall e volume das filas de revisão |
| vision | python -m ml_patterns_lab train vision | POST /predict | Acurácia com pixels quantizados e originais, cobertura e acurácia da cascata |
| recommend | python -m ml_patterns_lab train recommend | POST /recommend | HitRate@10, NDCG@10 e baseline de popularidade |
| sensors | python -m ml_patterns_lab train sensors | POST /predict | PR-AUC, precisão, recall, F1 e alvo de falsos alertas na validação |
| mlops | python -m ml_patterns_lab train mlops | POST /predict | ROC-AUC, PR-AUC, Brier, F1, recortes por planta e benchmark heurístico |

## Churn

```json
{"tenure_months":12,"monthly_spend":90,"support_tickets_90d":2,"plan":"premium","region":"sul"}
```

A resposta inclui probabilidade calibrada, limiar e recomendação de revisão para retenção. Planos e regiões têm categorias explícitas. A prioridade continua sendo uma indicação para avaliação humana.

## Demanda

O corpo recebe history com pelo menos 24 valores não negativos, last_timestamp em ISO 8601 e horizon de 1 a 48. O exemplo abaixo usa 24 horas de histórico:

```json
{"history":[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100],"last_timestamp":"2026-01-01T00:00:00","horizon":24}
```

O serviço pressupõe uma leitura por hora, em ordem, sem lacunas. A lista de valores não contém timestamps individuais, portanto não permite conferir essa frequência. O monitor local calcula PSI entre janelas; ele é um sinal de triagem com limiar heurístico.

## Suporte

```json
{"text":"Minha fatura mostra uma cobrança duplicada. Podem verificar?"}
```

As intenções são cobranca, tecnico, conta e entrega; revisao_humana é a abstinência. O conjunto pequeno limita a evidência de generalização. A separação por família de frase torna a avaliação mais exigente que uma divisão aleatória das variações.

## Fraude

```json
{"transaction_id":"txn-demo","amount":200,"transactions_last_hour":3,"merchant_risk_score":0.4,"distance_from_usual_km":40,"is_new_device":true,"is_cross_border":false}
```

A resposta mantém transaction_id e classifica a prioridade entre standard_queue, human_review e urgent_human_review. Se a validação não sustenta a precisão e o número mínimo de alertas, a faixa urgente fica desativada. Nenhuma faixa bloqueia pagamentos automaticamente.

## Visão

O corpo é um objeto com pixels: exatamente 576 valores entre 0 e 1, em ordem de linhas, para uma imagem 24×24. Em /docs é possível editar o esquema. O gerador make_images de vision/data cria amostras reproduzíveis dos dois domínios. O retorno informa clear, defect ou manual_review, a confiança e o estágio usado.

## Recomendação

```json
{"user_id":"user-001","limit":10}
```

POST /recommend/batch recebe user_ids distintos e um limit. Os IDs válidos ficam em predictions_by_user_id; os desconhecidos ficam em errors_by_user_id. POST /search recebe query e limit. As features do catálogo são compartilhadas na mesma versão; não há uma infraestrutura externa de feature store.

```json
{"query":"programação e dados","limit":5}
```

## Sensores

O corpo recebe readings com pelo menos 24 objetos, cada um contendo temperature_c, vibration_mm_s e pressure_kpa. A inferência usa as últimas 24 leituras. O exemplo pressupõe ordem temporal e intervalo constante, com a mesma unidade do treino. Uma lista sem timestamps não permite conferir frequência ou ordenação.

A resposta informa anomaly_score, anomaly_flag, threshold e três features com maiores desvios em relação à referência normal. Esses desvios são pistas de investigação, não uma explicação causal nem uma atribuição do IsolationForest.

```bash
python -m ml_patterns_lab monitor sensors
```

## MLOps

```json
{"hours_since_service":900,"temperature_c":70,"vibration_mm_s":3.5,"load_percent":60,"machine_type":"pump"}
```

O fluxo valida os dados e registra workflow.json na mesma transação da versão do modelo. machine_type aceita pump, compressor ou conveyor. A recomendação de manutenção depende da probabilidade e do limiar definido na validação.
