# Evidência de validação

Execução local da versão 0.2.0 em macOS ARM64. A suíte executou 71 testes com sucesso, treinando os oito modelos em um registro temporário e usando esses modelos nas APIs. Ruff passou nos checks de lint e formatação. Wheel e sdist foram construídos; o lockfile foi conferido e o ambiente não apresentou dependências quebradas.

Esses checks demonstram os fluxos e os invariantes exercitados. Eles não garantem ausência de todos os bugs nem desempenho em produção. O CI Quality usa as dependências do lockfile em Python 3.11 e 3.14.

## Ambiente da execução

| Componente | Versão |
| --- | --- |
| python | 3.14.7 |
| numpy | 2.5.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.9.1 |
| joblib | 1.6.0 |

## Resultados nos dados sintéticos

Valores abaixo foram extraídos dos metadados produzidos pelo treino da suíte. As métricas não usam o conjunto de validação que escolheu os limiares. A tabela apresenta um recorte; os metadados locais preservam as métricas completas.

| Projeto | Resultado no teste | Comparação ou cobertura |
| --- | --- | --- |
| churn | ROC-AUC 0.8001; F1 0.4989; Brier 0.1215 | Recall 0.7338; heurística 0.1169 |
| demand | MAE de 1 hora 4.676 kWh; recursivo de 24 horas 5.169 kWh | MAE sazonal 5.207 kWh |
| support | Acurácia entre encaminhados 0.9167; cobertura 1.0 | Baseline 0.25 |
| fraud | PR-AUC 0.0634; precisão 0.0317; recall 0.7179 | Prevalência 0.0217; fila de revisão 0.4906 |
| vision | Acurácia da cascata 1.0; cobertura 1.0 | Pixels quantizados 1.0; originais 1.0 |
| recommend | HitRate@10 0.4; NDCG@10 0.1591 | HitRate de popularidade 0.0667 |
| sensors | Precisão 0.9828; recall 1.0; F1 0.9913 | 877 janelas de teste |
| mlops | ROC-AUC 0.7963; F1 0.5216; Brier 0.1392 | Recall 0.6856; heurística 0.476 |

## Interpretação e limites

Fraude tem sinais fracos e uma classe rara neste gerador. A política recupera mais fraudes que a heurística, ao custo de encaminhar aproximadamente metade das transações, com baixa precisão. A validação não sustentou a faixa urgente, que ficou desativada. Esse resultado exige discutir custos e capacidade de revisão; não representa um detector pronto para bloquear pagamentos.

Suporte usa 17 famílias por intenção, com variações sintéticas. Treino, validação e teste preservam famílias distintas. O caso de teste com vocabulário desconhecido abstém; os tickets do teste sintético foram encaminhados pelo estágio rápido. Não foi medido ganho de latência entre estágios.

Os defeitos visuais e as anomalias de sensores são simples e fortes no gerador, produzindo resultados altos. As janelas de sensores são correlacionadas dentro de uma partição: a métrica mede janelas, não eventos independentes. Nenhum desses resultados prevê a dificuldade de imagens ou equipamentos reais.

O limiar de recall é escolhido na validação e pode apresentar recall menor no teste, como ocorreu em manutenção. A calibração não garante o mesmo comportamento sob mudança de distribuição. Pequenas diferenças entre versões de Python, bibliotecas e plataformas são possíveis.

## Reproduzir

```bash
uv sync --locked --extra api --extra dev
make check
uv build
python -m ml_patterns_lab train all
```

Os treinos também foram executados pela CLI central no registro padrão artifacts/, deixando os oito modelos disponíveis localmente. Modelos, relatórios temporários e ambientes não entram no Git.
