# Tech Challenge FIAP — Fase 3

Classificador educacional de cinco categorias de condições médicas. Entrega
TF-IDF + regressão logística, FastAPI, ONNX Runtime, Docker, Prometheus/Grafana,
Airflow e GitHub Actions. Execução local; **nenhuma nuvem foi provisionada**.

> Exclusivamente educacional. Não realiza diagnóstico, triagem, avaliação de
> urgência ou recomendação de tratamento. A confiança não é uma probabilidade
> clínica calibrada. Não envie dados reais de pacientes.

## Comece por aqui

- [EDA: conclusões e decisões](docs/eda.md)
- [Notebook executado: 18 células e 7 gráficos](notebooks/01_eda_medical_abstracts.ipynb)
- [Requisitos](docs/requirements/MLET%20-%20Tech%20Challenge%20Fase%203.md)
- [Issues](https://github.com/ygormartinelli/tech-challenge-fase-3/issues),
  [Kanban](https://github.com/users/ygormartinelli/projects/1) e
  [CI](https://github.com/ygormartinelli/tech-challenge-fase-3/actions)

## O que a EDA mudou

São **11.550 linhas de treino e 2.888 de teste**, sem nulos. Porém,
**1.010 linhas de teste (34,97%) repetem textos de treino**, e 2.929 grupos
do corpus combinado possuem categorias diferentes para o mesmo texto.
Não removemos esses conflitos nem escolhemos rótulos arbitrariamente.

A análise inclui classes, comprimentos, duplicatas, conflitos, sobreposição,
vocabulário, similaridade lexical, baseline majoritário, matriz de confusão e
confiança. A avaliação separa explicitamente as populações:

| População | Linhas | Acurácia | Macro-F1 |
|---|---:|---:|---:|
| Majoritário na validação | 2.310 | 0,3329 | 0,0999 |
| Modelo na validação por grupos | 2.310 | 0,6623 | 0,6650 |
| Teste completo | 2.888 | 0,5945 | 0,5982 |
| Teste: textos inéditos | 1.878 | 0,7604 | 0,7601 |
| Teste: textos vistos | 1.010 | 0,2861 | 0,3112 |

São populações distintas, não uma melhoria causal. O teste já havia sido
avaliado antes da revisão, portanto não é um holdout cego. Evidências:
[validação](reports/validation.json), [avaliação](reports/evaluation.json)
e [qualidade dos dados](reports/eda.json).

## Instalação e execução

Pré-requisitos: Python 3.11, [uv](https://docs.astral.sh/uv/), Docker Desktop
em modo Linux e Docker Compose. GNU Make oferece os atalhos abaixo; cada etapa
também pode ser executada com
`uv run --frozen python -m techchallenge_fase3.pipelines.NOME`.

Mantenha em `data/raw/`: `medical_tc_train.csv`, `medical_tc_test.csv` e
`medical_tc_labels.csv`. Os CSVs não são baixados automaticamente nem enviados
ao Git. Referência: [corpus de Schopf, Braun e Matthes](https://github.com/sebischair/Medical-Abstracts-TC-Corpus),
com licença publicada CC BY-SA 3.0. O relatório de EDA registra os hashes dos
arquivos efetivamente analisados; não redistribuímos abstracts.

Na raiz do projeto, em PowerShell:

~~~powershell
# Apenas na primeira configuração; preserve um .env existente.
Copy-Item .env.example .env
uv sync --frozen --all-groups
uv run pre-commit install
make validate
make eda
make notebook
make pipeline
make compose
~~~

Em Linux/macOS, use `cp .env.example .env` para a cópia.
API, Prometheus, Grafana e Airflow são instalados **pelas imagens Docker**.
Não instale Airflow no Windows. Dependências Python de execução usam o mesmo
`uv.lock`; a imagem da API não instala as dependências do notebook.
Airflow executa o projeto em ambiente separado do seu próprio ambiente.

`make pipeline` executa `validate → train → optimize → evaluate → benchmark → publish`.
Se uma etapa falhar, a sequência para e a release anterior permanece publicada.

## API e contrato

| Rota | Comportamento |
|---|---|
| `POST /predict` | Categoria, nome, confiança, variante e aviso educacional |
| `GET /health` | Readiness: 200 quando o modelo carrega; 503 se indisponível |
| `GET /metrics` | Métricas Prometheus sem textos recebidos |
| `GET /docs` | Documentação interativa OpenAPI |

~~~powershell
$payload = @{ medical_abstract = "Tumor cells were investigated in cancer treatment." } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -Body $payload
~~~

Resposta: `condition_label`, `condition_name`, `confidence`, `model_variant`
e `disclaimer`. Não há campo de urgência.

| Rótulo | Nome retornado |
|---|---|
| 1 | neoplasms |
| 2 | digestive system diseases |
| 3 | nervous system diseases |
| 4 | cardiovascular diseases |
| 5 | general pathological conditions |

Vazios, valores não textuais, textos menores que 10 ou maiores que 20.000
caracteres e campos extras recebem 422. Erros de modelo recebem 503 genérico,
sem detalhes internos. `make run` inicia somente a API Python; não use a mesma
porta simultaneamente com a API Docker.

## Modelo, otimização e publicação segura

Modelo final: até 30.000 unigramas, tokenização ASCII explícita de dois ou mais
caracteres, TF linear e regressão logística balanceada, seed 42. Validação:
primeiro fold de `StratifiedGroupKFold(5)` com grupos disjuntos, aproximadamente
80/20; não é média de cinco folds. Depois, ajuste nas 11.550 linhas de treino.

A revisão detectou diferenças na conversão do tokenizer, da frequência sublinear
e de alguns bigramas. A configuração final evita essas divergências sem
conversor customizado. Veja o [diagnóstico preservado](reports/onnx_pre_fix.json)
e a [documentação do conversor](https://onnx.ai/sklearn-onnx/_modules/skl2onnx/operator_converters/text_vectoriser.html).

Factory seleciona a estratégia original ou ONNX; ambas retornam rótulos e
probabilidades em uma única passagem. ONNX exige aprovação vinculada aos hashes
da release. Sem aprovação, a seleção usa o original; arquivo ausente ou
corrompido falha fechado com 503.

Candidatos ficam em `models/candidate/`. A publicação cria uma release em
`models/releases/` e troca atomicamente `models/current.json`, sem apagar
releases anteriores. **Carregue joblib somente de origem confiável.**
A API mantém a release em memória; para adotar uma publicação posterior:

~~~powershell
docker compose --env-file .env -f docker/docker-compose.yml restart api
~~~

Não execute dois pipelines CLI simultaneamente sobre o mesmo candidato.
Airflow usa outro candidato, `max_active_runs=1` e relatórios em
`reports/runtime/airflow/`, separados da evidência versionada desta entrega.

### Benchmark

`make benchmark` usa 200 textos variados, seed 42, 20 chamadas de aquecimento
e três rodadas com ordem alternada. O contrato medido inclui rótulos e
probabilidades, sem HTTP. Aprovação exige:

- Rótulos idênticos em todo o teste e seis casos de borda: 2.894 textos.
- Diferença absoluta máxima nas probabilidades ≤ `1e-5`.
- p50 ONNX menor em todas as três rodadas.

Resultado local Windows/CPU de 15/09/2026:

| Variante | Média (ms) | p50 (ms) | p95 (ms) | Predições/s |
|---|---:|---:|---:|---:|
| Original | 1,1225 | 1,0617 | 1,6354 | 890,84 |
| ONNX | 0,3290 | 0,2804 | 0,6177 | 3.039,86 |

Zero divergências de rótulo; maior diferença de probabilidade: `2,06e-7`.
[Relatório, hashes e rodadas](reports/latency_benchmark.json).
O grafo usa `LinearClassifier`, sem operadores elegíveis identificados para
quantização dinâmica padrão; não aplicamos quantização sem ganho demonstrado.

`make api-benchmark` mede a variante realmente em execução, consumindo a
resposta HTTP inteira: 200 textos fixos e 20 chamadas de aquecimento.
Altere `MODEL_VARIANT` no `.env`, recrie apenas a API com o comando abaixo e
execute uma vez para cada variante. Deixe `optimized` ao final.

~~~powershell
docker compose --env-file .env -f docker/docker-compose.yml up -d --no-deps --force-recreate --wait api
make api-benchmark
~~~

| HTTP em Docker | Média (ms) | p50 (ms) | p95 (ms) | Requisições/s |
|---|---:|---:|---:|---:|
| Original | 8,4444 | 6,0369 | 24,1267 | 118,42 |
| ONNX | 7,3982 | 4,8242 | 25,9381 | 135,17 |

[HTTP original](reports/http_original.json) e [HTTP ONNX](reports/http_optimized.json).
O p50 melhorou, mas o p95 HTTP piorou nessa medição. Não alegamos melhoria
em todos os percentis. Throughput é sequencial, não teste de saturação,
SLA de produção ou garantia de resultado em outra máquina.

## Monitoramento e Airflow

~~~powershell
make compose
make smoke
make airflow
docker compose -f docker/docker-compose.yml exec -T airflow airflow dags trigger retrain_medical_classifier
docker compose -f docker/docker-compose.yml exec -T airflow airflow dags list-runs -d retrain_medical_classifier
docker compose -f docker/docker-compose.yml --profile airflow ps
~~~

| Serviço | Endereço | Acesso local |
|---|---|---|
| API | [OpenAPI](http://localhost:8000/docs) | Sem login |
| Prometheus | [localhost:9090](http://localhost:9090) | Sem login |
| Grafana | [Dashboard](http://localhost:3000/d/medical-classifier) | `admin` / `GRAFANA_ADMIN_PASSWORD` |
| Airflow | [localhost:8080](http://localhost:8080) | `admin` / senha gerada pelo standalone |

O exemplo usa Grafana `admin/admin`, **somente para demonstração local**.
Em volume já inicializado, alterar a variável não redefine a senha existente.
Para ler a senha do Airflow localmente, sem publicá-la:

~~~powershell
docker compose -f docker/docker-compose.yml exec -T airflow cat /opt/airflow/standalone_admin_password.txt
~~~

Portas vinculadas a `127.0.0.1`. Não exponha esta stack à Internet:
não há autenticação clínica, TLS ou hardening de produção.

Prometheus coleta a cada cinco segundos. O dashboard provisionado apresenta
total de predições, p50/p95, taxa de falhas internas e proporção de respostas
4xx/5xx. Scraping não aumenta os contadores. `make smoke` gera sucessos e erros
de validação, verifica readiness, target e **consultas dos painéis através do Grafana**.
Sem tráfego recente, taxas voltam a zero e percentis podem ficar sem dados;
gere novas requisições.

A DAG chama os seis comandos reutilizáveis: validar, treinar, exportar,
avaliar, medir e publicar. Uma execução real está em
[evidência Airflow](reports/airflow_verification.json);
a stack está em [consultas verificadas](reports/stack_verification.json).
Standalone/SQLite serve à demonstração, não à produção.

Para parar sem apagar dados:

~~~powershell
docker compose -f docker/docker-compose.yml --profile airflow stop
~~~

## Qualidade e CI

~~~powershell
make lint
make test
uv pip check
docker build -f docker/Dockerfile -t medical-classifier:ci .
uv run --frozen python scripts/smoke_image.py
~~~

GitHub Actions em pushes e PRs: Ruff e formatação, pytest, consistência de
dependências, build da API, smoke Docker com fixture sintética, build do Airflow
e importação real da DAG. A CI não recebe CSVs locais e não apresenta tempos
sintéticos como evidência de otimização. O gate real está no pipeline e na DAG.

Testes cobrem dados, grupos, conflitos, baseline, paridade, rejeição de benchmark,
publicação, corrupção, fallback, API, readiness, métricas e infraestrutura.
O smoke temporário remove apenas seu próprio container e fixture.

O notebook executou em kernel limpo; sete gráficos foram inspecionados.
As consultas reais do Grafana passaram; a captura visual automatizada da página
não ocorreu porque o navegador de automação não inicializou. Não alegamos
screenshots inexistentes.

## Arquitetura e decisão AWS

~~~mermaid
flowchart LR
    CSV[CSVs locais] --> VALID[Validação]
    VALID --> TRAIN[Treino por grupos]
    TRAIN --> ONNX[Exportação ONNX]
    ONNX --> EVAL[Avaliação e benchmark]
    EVAL --> REL[Release aprovada]
    REL --> API[FastAPI em Docker]
    API --> PROM[Prometheus]
    PROM --> GRAF[Grafana]
    AIR[Airflow opcional] -. mesmos comandos .-> VALID
~~~

**Somente documentado:** para serviço real-time contínuo, escolher AWS ECS
Fargate com ALB HTTPS, imagem no ECR e release versionada carregada de S3.
Fargate evita gerenciar servidores; ECS permite escala por métricas.
Referências: [Fargate](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html)
e [Service Auto Scaling](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-auto-scaling.html).

A escolha favorece processos aquecidos e controle da imagem, sem garantir menor
custo que funções sob demanda. Produção ainda exige IAM mínimo, segredos,
TLS, autenticação, rate limiting, varredura de vulnerabilidades, alertas,
rollback, testes de carga, privacidade e validação do domínio. Observabilidade
gerenciada pode substituir a stack local. Airflow standalone exige substituição
por uma implantação adequada. Nada disso foi provisionado.

## Roteiro STAR — até cinco minutos

| Tempo | Fala e demonstração |
|---|---|
| 0:00–0:40 — Situação | “Organizar resumos médicos em cinco categorias, sem diagnóstico ou decisão de urgência.” Mostrar escopo e classes. |
| 0:40–1:15 — Tarefa | “Entregar inferência conteinerizada, automação, monitoramento e otimização medida.” Mostrar requisitos, issues e Kanban. |
| 1:15–2:05 — Ação: dados | Mostrar EDA: 34,97% de overlap, conflitos, validação por grupos e matriz de confusão. Explicar as populações das métricas. |
| 2:05–2:50 — Ação: API | Fazer POST em /docs e mostrar /health. Explicar Factory/Strategy, passagem única e aprovação de artefatos. |
| 2:50–3:35 — Ação: automação | Mostrar CI verde e DAG com seis tarefas concluídas; explicar candidato e publicação condicionada. |
| 3:35–4:15 — Ação: monitoramento | Gerar tráfego com make smoke e mostrar volume, latência e erros no Grafana. |
| 4:15–5:00 — Resultado | Mostrar redução de p50 e paridade; reconhecer p95 HTTP observado, limitações dos dados e decisão AWS apenas documentada. |

**Vídeo pendente do autor:** gravar e incluir o link na issue #6. A entrega
acadêmica não está integralmente concluída enquanto isso faltar.

## Evidências por critério da rubrica

| Critério | Peso | Evidência |
|---|---:|---|
| Modelo e otimização | 20% | [EDA](docs/eda.md), [avaliação](reports/evaluation.json), [benchmark](reports/latency_benchmark.json) |
| CI/CD | 15% | [Workflow](.github/workflows/ci.yml) e [execuções](https://github.com/ygormartinelli/tech-challenge-fase-3/actions) |
| Airflow | 15% | [DAG](dags/retrain_medical_classifier.py) e [execução real](reports/airflow_verification.json) |
| Monitoramento | 20% | [Compose](docker/docker-compose.yml), [dashboard](docker/grafana/provisioning/dashboards/medical-classifier.json), [consultas](reports/stack_verification.json) |
| Documentação | 15% | Este README: reprodução, resultados, arquitetura, limitações e roteiro |
| Vídeo STAR | 15% | Pendente de gravação pelo autor e link na issue #6 |
