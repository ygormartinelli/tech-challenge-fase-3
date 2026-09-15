# EDA: conclusões e decisões de desenvolvimento

Análise dos CSVs locais em 15/09/2026. O [notebook executado](../notebooks/01_eda_medical_abstracts.ipynb)
contém cálculos, sete gráficos, tabelas, verificações e conclusões. Reexecute com
`make eda` e `make notebook`. Os [resultados agregados](../reports/eda.json)
registram os hashes dos três arquivos e as versões utilizadas.

## Conclusão principal

Os dados permitem avançar com um classificador **educacional de categorias**,
mas não sustentam triagem clínica nem uma avaliação baseada apenas no teste
fornecido. A unidade observada é uma associação abstract–categoria; não há
identificador de paciente, hospital ou data. Preservamos os rótulos originais.

| Verificação | Treino | Teste |
|---|---:|---:|
| Linhas | 11.550 | 2.888 |
| Textos normalizados únicos | 9.445 | 2.770 |
| Linhas integralmente duplicadas | 0 | 0 |
| Repetições de texto além da primeira ocorrência | 2.105 | 118 |
| Grupos textuais com categorias distintas | 1.956 | 113 |
| Linhas pertencentes a esses grupos | 4.061 | 231 |
| Nulos nas duas colunas | 0 | 0 |
| Textos fora do limite de 10–20.000 caracteres | 0 | 0 |

Normalização para agrupamento: Unicode NFKC, casefold e espaços consecutivos.
Ela identifica grupos; não modifica os CSVs nem escolhe categorias.

## 1. Sobreposição e ambiguidade são o maior risco

**1.010 linhas de teste (34,97%) repetem textos do treino**, envolvendo 988 grupos
compartilhados. Restam 1.878 linhas de teste com textos inéditos em relação ao treino.
No corpus combinado, são 11.227 textos únicos e 2.929 grupos com mais de uma
categoria. A ausência de duplicatas de linhas completas escondia esse problema.

Isso não autoriza afirmar que os rótulos estão errados: um abstract pode tratar
de vários temas. A causa da anotação não é verificável com os campos disponíveis.
Deduplicar somente pelo texto ou escolher uma categoria por votação descartaria
informação sem justificativa. Não fizemos isso.

O teto empírico de acurácia de uma decisão única por grupo neste corpus combinado
é 77,76%: soma da maior contagem de categoria por grupo, dividida pelas 14.438
linhas. É uma propriedade desses rótulos e desse agrupamento, não uma previsão de
desempenho, limite universal ou garantia clínica.

Decisões implementadas: validação estratificada por grupos, vocabulário ajustado
somente na partição de treino e relatório separado para teste completo, textos
vistos e inéditos. A sobreposição não implica necessariamente inflação de
acurácia: neste corpus, os casos vistos têm desempenho menor, compatível com a
ambiguidade dos rótulos. Não atribuímos causalidade sem revisão de anotação.

## 2. Há desbalanceamento, mas não mudança relevante na composição fornecida

| Categoria | Treino | Teste |
|---|---:|---:|
| Neoplasias | 2.530 | 633 |
| Sistema digestivo | 1.195 | 299 |
| Sistema nervoso | 1.540 | 385 |
| Cardiovasculares | 2.441 | 610 |
| Condições patológicas gerais | 3.844 | 961 |

A maior classe tem 3,22 vezes o tamanho da menor. As proporções de treino e
teste são muito próximas, mas isso não demonstra independência textual.
Mantivemos `class_weight="balanced"`, Macro-F1, precisão/recall/F1 por classe
e comparação com um classificador majoritário. Não aplicamos oversampling.

## 3. Comprimento não exige truncamento nem remoção de extremos

Medianas de caracteres: 1.208 no treino e 1.221,5 no teste; percentis 95:
2.049 e 2.053,25. Máximos: 3.999 e 3.542. Todos cabem no limite atual da API.
As distribuições são próximas; o KS é apenas exploratório porque duplicatas
violam a independência das observações. Sem datas, não há análise temporal válida.

Decisões: não remover abstracts longos ou curtos por conveniência; rejeitar
payloads vazios, não textuais, muito curtos, acima do limite ou com campos extras.

## 4. O vocabulário sustenta um baseline leve em CPU

Na exploração lexical, 20.000 unigramas com `min_df=2` e stopwords em inglês
produzem uma matriz de treino com 99,65% de esparsidade. O teste não tem vetores
vazios; 3,29% das ocorrências de tokens ficam fora desse vocabulário limitado.
Os termos são consistentes com os temas das classes, mas os maiores TF-IDFs
médios não são explicações causais nem palavras exclusivas.

No subconjunto de textos inéditos, nenhum vizinho de treino alcançou cosseno
0,90 ou 0,95; máximo observado: 0,8136. Isso não exclui paráfrases ou dependência
de pacientes. O vocabulário e a busca de vizinhos são ajustados só no treino.

Essa exploração não é o transformador de produção: o classificador final usa
até **30.000 unigramas**, frequência linear e tokenização ASCII explícita de
dois ou mais caracteres, compatível com o corpus em inglês e com ONNX.

## 5. O modelo supera o majoritário, mas ainda erra

| População | Linhas avaliadas | Acurácia | Macro-F1 |
|---|---:|---:|---:|
| Majoritário na validação | 2.310 | 0,3329 | 0,0999 |
| Modelo na validação por grupos | 2.310 | 0,6623 | 0,6650 |
| Teste completo | 2.888 | 0,5945 | 0,5982 |
| Teste: textos inéditos | 1.878 | 0,7604 | 0,7601 |
| Teste: textos vistos | 1.010 | 0,2861 | 0,3112 |

Validação: primeiro fold de `StratifiedGroupKFold(5, shuffle=True, random_state=42)`;
9.240 linhas de ajuste e 2.310 de validação. Não é média de cinco folds. Depois,
o modelo final foi ajustado nas 11.550 linhas de treino. Nenhum hiperparâmetro
foi escolhido pelo score do teste. Como o teste já tinha sido usado antes desta
revisão, ele não deve ser chamado de avaliação cega.

No teste inédito, condições gerais apresentam recall de aproximadamente 57%,
contra 77–89% nas outras classes. A matriz de confusão mostra erros distribuídos
para categorias específicas. É um limite importante de uma classe abrangente;
melhorar o alvo depende de revisão das anotações, não apenas de outro algoritmo.
Os valores acima comparam populações distintas e não estimam um ganho causal.

Confiança é a maior probabilidade do modelo, sem calibração clínica. O notebook
mostra acurácia por faixa de confiança, sem treinar calibrador nem estabelecer
um limiar de triagem. Não há avaliação externa, por hospital ou demográfica.

## 6. A revisão também encontrou defeitos de engenharia

O teste completo detectou 22 divergências de rótulo na conversão antiga. Alinhar
tokenização resolveu parte delas; a frequência sublinear tinha fórmula diferente
no conversor, e quatro textos ainda apresentavam diferenças em bigramas.
Optamos por unigramas e TF linear, mantendo a implementação simples, e repetimos
os testes sem relaxar a tolerância de probabilidades. O
[histórico rejeitado](../reports/onnx_pre_fix.json) documenta o diagnóstico;
o [benchmark aprovado](../reports/latency_benchmark.json) contém os resultados finais.

Outras correções: UID do datasource Grafana, métricas de erro como taxa,
readiness 503, uma inferência por requisição, dependências travadas nos containers,
retreino isolado e publicação atômica após validação.

## Encaminhamento

As issues #1–#5 recebem testes e evidências das correções; #7 registra a EDA;
#6 concentra README, CI, roteiro STAR e entrega. Evoluir para múltiplos rótulos,
prontuários reais, português ou uso clínico exige uma nova decisão de produto,
novos dados e validação específica. Não faz parte desta entrega.

Fonte externa de referência: [Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus).
Os números deste documento foram calculados dos CSVs locais identificados por
SHA-256, e não copiados de uma descrição de terceiros.
