# Tech Challenge FIAP — Fase 3

Classificador educacional de categorias de condições médicas a partir de resumos
textuais. O projeto entrega uma API FastAPI, pipeline de treino automatizável,
observabilidade local e otimização de inferência com ONNX Runtime.

> Aviso: este projeto é exclusivamente educacional. Ele não realiza diagnóstico,
> triagem clínica nem recomendação de tratamento e não deve ser usado para decisões
> médicas.

## Escopo e arquitetura

O dataset Medical Abstracts TC Corpus possui cinco categorias: neoplasias,
doenças do sistema digestivo, doenças do sistema nervoso, doenças
cardiovasculares e condições patológicas gerais. O modelo usa TF-IDF com
regressão logística, uma escolha leve, interpretável e apropriada para CPU.

Para produção em tempo real, a arquitetura recomendada é AWS ECS Fargate:
Application Load Balancer recebe as requisições, tarefas ECS executam a API em
containers e Amazon Managed Service for Prometheus/Grafana centraliza a
observabilidade. O ECS Fargate evita gerenciar servidores e permite escalar a
API por CPU ou número de requisições. O deploy em nuvem é apenas documentado;
a entrega executável é totalmente local.

## Pré-requisitos

- Python 3.11 e [uv](https://docs.astral.sh/uv/)
- Docker Desktop com Docker Compose

Airflow não precisa ser instalado na máquina: ele é iniciado apenas pelo perfil
Docker opcional.

## Execução local

```bash
copy .env.example .env
make install
make train
make optimize
make benchmark
make run
```

A API estará disponível em `http://localhost:8000`. Exemplo:

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"medical_abstract\": \"Tumor cells were investigated in the patient.\"}"
```

Rotas públicas:

- `POST /predict`: recebe `medical_abstract` e retorna rótulo, categoria,
  confiança, variante do modelo e aviso de segurança.
- `GET /health`: confirma que o modelo foi carregado.
- `GET /metrics`: expõe métricas Prometheus.

## Treino e otimização

`make train` cria `models/classifier.joblib` e registra Accuracy e Macro-F1 em
`reports/evaluation.json`. `make optimize` exporta o mesmo pipeline para
`models/classifier.onnx`. `make benchmark` aquece ambos os modelos e compara
latência média, p50, p95 e throughput em `reports/latency_benchmark.json`.

O benchmark falha se os rótulos não forem equivalentes ou se ONNX Runtime não
reduzir a latência p50. Isso evita declarar uma otimização sem evidência.

## Monitoramento e orquestração

Após treinar o modelo, inicie a stack de observabilidade:

```bash
docker compose -f docker/docker-compose.yml up --build
```

- API: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (`admin` / `admin`)

O dashboard provisionado apresenta total de requisições, latência p95 e erros de
inferência. Gere tráfego com chamadas para `/predict` e aguarde um ciclo de
coleta do Prometheus.

Para subir o Airflow e a DAG `retrain_medical_classifier`:

```bash
docker compose -f docker/docker-compose.yml --profile airflow up --build
```

A DAG executa validação de dados, treino, exportação ONNX e benchmark usando os
mesmos comandos reutilizáveis da aplicação.

## Qualidade e CI/CD

```bash
make lint
make test
```

O GitHub Actions executa Ruff, Pytest e build da imagem Docker em cada push e
pull request.

## Roteiro STAR para vídeo

- **Situation:** hospitais precisam classificar rapidamente grandes volumes de
  textos; este protótipo organiza categorias, sem tomar decisões clínicas.
- **Task:** entregar API conteinerizada, baixa latência, CI/CD, Airflow e
  observabilidade.
- **Action:** demonstrar TF-IDF + regressão logística, conversão ONNX, Compose,
  dashboard, DAG e workflow do GitHub.
- **Result:** mostrar os relatórios de Macro-F1 e benchmark, métricas em Grafana
  e as limitações de uso seguro do protótipo.
