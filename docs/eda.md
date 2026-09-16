# EDA: decisões para a demonstração de urgência

A [análise executada](../notebooks/01_eda_synthetic_triage.ipynb) substitui as conclusões operacionais de categorias. A adaptação sintética foi autorizada porque o corpus sugerido não tem rótulos de urgência. Seus cinco temas foram usados como referência, sem reclassificar os abstracts originais.

## Evidências

| Verificação | Treino | Teste |
|---|---:|---:|
| Linhas sintéticas | 2.700 | 600 |
| Cenários-base distintos | 45 | 30 |
| Linhas por classe | 900 | 200 |
| Nulos / conflitos / textos fora do limite | 0 | 0 |
| Variações de redação por cenário | 60 | 20 |

Zero cenários e zero textos normalizados compartilhados entre partições. Há somente 75 cenários autorais, sem anotação clínica; o tamanho expandido não representa diversidade de pacientes. A contagem de 2.000 amostras do requisito precisa ser interpretada junto dessa limitação pela avaliação acadêmica.

Validação por cenário: 36 para treino e 9 para validação, primeiro fold estratificado com seed 42. Macro-F1 0,615 versus 0,167 do majoritário. Teste: 0,737 por redação e 0,738 em uma redação fixa por cenário. Houve 8 erros entre 30 cenários, incluindo três urgentes classificados como atenção. Não ajustar rótulos nem hiperparâmetros a partir desses erros de teste.

## Conclusões e ações

| Evidência / risco | Decisão implementada |
|---|---|
| Categoria de doença não é urgência | Alvo separado: 0 normal, 1 atenção, 2 urgente; origem sintética explícita |
| Variações do mesmo achado inflariam a avaliação | Split por cenário antes da expansão; suporte por cenário reportado |
| Classes equilibradas pelo autor | Não interpretar como prevalência; usar macro-F1 e métricas por classe |
| Teste com 14,81% de tokens OOV no vetorizador exploratório | Registrar cobertura e limitações; não assumir compreensão de termos novos |
| Laudos curtos e prefixos comuns | Benchmarks valem para este perfil; não extrapolar para prontuários extensos |
| Tokenização ASCII e unigramas | Paridade ONNX verificada; fragmentação de acentos e negação são limitações |
| Três urgentes subestimados | Não alegar segurança; manter erros e disclaimer na entrega |
| Release anterior tinha cinco doenças | Manifesto identifica tarefa e origem; API rejeita release incompatível |
| Latência deve ser demonstrada | Gate exige probabilidades equivalentes e p50 menor em três rodadas |

Proveniência: [manifesto de geração](../reports/data_generation.json), [EDA agregada](../reports/eda.json), [validação](../reports/validation.json) e [teste](../reports/evaluation.json). Sete figuras foram executadas e inspecionadas. A distribuição de comprimentos difere em parte porque o teste usa apenas as primeiras 20 combinações de estilo; isso não é drift hospitalar.

Reprodução: `make generate`, `make eda`, `make notebook`. As evidências históricas do corpus original de categorias permanecem identificadas no README; não comprovam desempenho de urgência. Aplicação real exigiria outra base, anotação especializada e validação externa, itens não realizados.
