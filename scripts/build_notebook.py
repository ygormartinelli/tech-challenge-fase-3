"""Constrói e executa a EDA da simulação, com gráficos e trilha de auditoria."""
# ruff: noqa: E501

import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
CELLS = [
    (
        "md",
        """# EDA — urgência sintética, Fase 3

## tl;dr

**Demonstração acadêmica, não validação clínica.** O corpus sugerido não possui urgência. Com autorização do responsável, criamos 75 cenários fictícios nos seus cinco temas, com os rótulos pedagógicos `normal`, `atenção` e `urgente`. Não convertemos doença em urgência nem copiamos abstracts para o treino.

A expansão produz 2.700 linhas de treino e 600 de teste, mas apenas **45 + 30 cenários distintos**. Não são 3.300 amostras clínicas independentes. Não há cenários ou textos normalizados compartilhados entre as partições. O teste reservado apresentou 22/30 cenários corretos, macro-F1 0,738; três cenários urgentes foram classificados como atenção. O desempenho não autoriza triagem de pacientes.

## Contexto e métodos

Leitor: autor e avaliadores do Tech Challenge. Decisão: verificar a adaptação de alvo e os limites antes da entrega técnica. Esta análise substitui a EDA operacional anterior de cinco doenças, preservada no histórico Git.

### Premissas

- Fonte temática: [Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus). Os CSVs originais em `data/raw` não foram alterados.
- Fonte efetiva: `src/techchallenge_fase3/synthetic_cases.py` e gerador `pipelines/generate.py`; não houve anotação clínica especializada.
- Cada tema tem cinco achados por urgência. Seed 42 separa três para treino e dois para teste **antes** da expansão. Prefixos e sufixos neutros são comuns às classes, não regras de urgência.
- 60 variações por cenário de treino; 20 por cenário de teste. Isso atende uma contagem de linhas, não diversidade clínica. A aceitação dessa adaptação cabe à avaliação acadêmica.
- Validação: primeiro fold do StratifiedGroupKFold(5), por `scenario_id`; não são cinco treinos de validação cruzada. Pipeline fixo, sem ajuste pelo teste.
- TF-IDF unigramas + regressão logística balanceada. Somente `report_text` entra no modelo; IDs, tema e origem não são features. Tokenização ASCII é compartilhada com ONNX; termos acentuados são fragmentados. Negações e relações clínicas não são compreendidas semanticamente.

Reprodução: `make generate`, `make eda`, `make notebook`. Kernel limpo, seed fixa, hashes e versões registrados abaixo. As métricas são recalculadas, não apenas lidas de relatórios antigos.""",
    ),
    (
        "code",
        """from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from techchallenge_fase3.analysis import describe_corpus
from techchallenge_fase3.config import Settings
from techchallenge_fase3.data import LABEL_NAMES, load_dataset, validate_splits
from techchallenge_fase3.modeling import evaluate_model, scenario_evaluation, train_model, validate_baseline
from techchallenge_fase3.reporting import environment, file_hash

settings = Settings()
train = load_dataset(settings.train_path)
test = load_dataset(settings.test_path)
validate_splits(train, test)
eda = describe_corpus(train, test)
names = list(LABEL_NAMES.values())
colors = ['#227C9D', '#D99B23', '#B44747']
figure_dir = Path('reports/figures')
figure_dir.mkdir(parents=True, exist_ok=True)
pd.set_option('display.max_colwidth', 80)
plt.rcParams.update({'figure.figsize': (9, 4.5), 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

def save_figure(name):
    plt.tight_layout()
    plt.savefig(figure_dir / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
""",
    ),
    (
        "md",
        """## Dados

### Proveniência e contrato

Grão de cada linha: uma redação de um cenário fictício. `scenario_id` identifica sua família; `urgency_label` é o alvo pedagógico. Não há pacientes, hospitais, datas de eventos, dados demográficos ou desfechos reais. Datas de execução medem apenas a produção do artefato.""",
    ),
    (
        "code",
        """display(pd.DataFrame([{'arquivo': p.as_posix(), 'sha256': file_hash(p)} for p in (settings.train_path, settings.test_path, settings.labels_path)]))
display(pd.DataFrame([{'partição': name, 'linhas': eda[name]['rows'], 'cenários': eda['scenarios'][name], 'nulos': sum(eda[name]['nulls'].values()), 'textos repetidos': eda[name]['duplicate_text_rows_beyond_first'], 'conflitos de rótulo': eda[name]['conflicting_text_groups'], 'fora do limite API': eda[name]['outside_api_length']} for name in ('train', 'test')]))
assert eda['scenarios']['shared'] == 0
assert eda['overlap']['shared_text_groups'] == 0
assert len(train) == 2700 and len(test) == 600
display(Markdown('**Integridade:** nenhuma ausência, conflito ou violação do limite textual foi observada. Isso valida o gerador, não a representatividade clínica.'))
""",
    ),
    ("md", "### Classes: equilíbrio imposto pelo desenho, não prevalência hospitalar"),
    (
        "code",
        """classes = pd.DataFrame(eda['classes'])
fig, ax = plt.subplots()
x = np.arange(3)
ax.bar(x - .18, classes.train, .36, label='Treino: 2.700 linhas', color='#227C9D')
ax.bar(x + .18, classes.test, .36, label='Teste: 600 linhas', color='#D99B23')
ax.set(xticks=x, xticklabels=names, ylabel='Linhas sintéticas', ylim=(0, 1200), title='Classes balanceadas por construção')
ax.legend()
save_figure('class_distribution')
display(classes[['name', 'train', 'test', 'train_share', 'test_share']].round(3))
""",
    ),
    ("md", "### Variações não aumentam o número de cenários independentes"),
    (
        "code",
        """fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, values, title, unit in zip(axes, ([len(train), len(test)], [train.scenario_id.nunique(), test.scenario_id.nunique()]), ('Redações expandidas', 'Cenários-base distintos'), ('Linhas', 'Cenários'), strict=True):
    bars = ax.bar(['Treino', 'Teste'], values, color=['#227C9D', '#D99B23'])
    ax.bar_label(bars, padding=3)
    ax.set(title=title, ylabel=unit, ylim=(0, max(values) * 1.2))
save_figure('scenario_support')
display(Markdown('As 60/20 redações de cada cenário compartilham o mesmo achado. O suporte efetivo é pequeno; não calculamos intervalos de confiança tratando variações como observações independentes.'))
""",
    ),
    ("md", "### Comprimentos: contrato da API e custo de inferência"),
    (
        "code",
        """fig, ax = plt.subplots()
bins = np.linspace(min(train.report_text.str.len().min(), test.report_text.str.len().min()), max(train.report_text.str.len().max(), test.report_text.str.len().max()), 22)
for data, color, name in ((train, '#227C9D', 'Treino'), (test, '#D99B23', 'Teste')):
    values = data.report_text.str.len()
    ax.hist(values, bins=bins, weights=np.ones(len(values)) / len(values) * 100, histtype='step', linewidth=2, label=name, color=color)
ax.set(xlabel='Caracteres por redação', ylabel='% de linhas na partição', title='Laudos sintéticos curtos: não representam prontuários extensos')
ax.legend()
save_figure('text_lengths')
display(pd.DataFrame({name: eda[name]['characters'] for name in ('train', 'test')}).round(1))
""",
    ),
    ("md", "### Extensão por classe e possíveis atalhos do gerador"),
    (
        "code",
        """fig, ax = plt.subplots()
for index, label in enumerate(LABEL_NAMES):
    words = train.loc[train.urgency_label.eq(label), 'report_text'].str.split().str.len()
    ax.boxplot(words, positions=[index], widths=.5, tick_labels=[names[index]], patch_artist=True, boxprops={'facecolor': colors[index]})
ax.set(xticks=range(3), xticklabels=names, ylabel='Palavras por redação de treino', title='Comprimento também pode revelar o estilo autoral')
save_figure('words_by_class')
display(pd.DataFrame(eda['lexical']['top_mean_tfidf_terms_by_class']).rename(columns={str(k): v for k, v in LABEL_NAMES.items()}))
display(Markdown('Palavras como “sem”, “estável” ou termos de comprometimento podem servir de atalhos. O modelo aprende associações do autor; essa separabilidade não demonstra raciocínio clínico.'))
""",
    ),
    (
        "md",
        """### Sobreposição e cobertura lexical

Zero overlap exato não elimina semelhança de estilo. A similaridade usa vocabulário ajustado somente no treino. O teste compartilha prefixos e sufixos neutros; OOV e similaridade são diagnósticos lexicais, não validação externa.""",
    ),
    (
        "code",
        """lexical = eda['lexical']
display(pd.DataFrame([{'vocabulário EDA': lexical['vocabulary_size'], 'tokens OOV no teste (%)': lexical['test_oov_token_share'] * 100, 'vetores zerados': lexical['test_zero_vectors'], 'textos compartilhados': eda['overlap']['shared_text_groups'], 'cenários compartilhados': eda['scenarios']['shared']}]).round(3))
quantiles = lexical['near_duplicates']['unseen_test_similarity_quantiles']
fig, ax = plt.subplots()
bars = ax.bar(['Mínimo', 'p50', 'p90', 'p95', 'Máximo'], quantiles, color='#227C9D')
ax.bar_label(bars, fmt='%.2f', padding=3)
ax.set(ylim=(0, 1.1), ylabel='Similaridade cosseno (0–1)', title='Vizinho mais próximo do treino — 600 redações de teste')
save_figure('lexical_similarity')
display(eda['distribution'])
""",
    ),
    (
        "md",
        """## Resultados

### Baseline, validação por cenário e teste reservado

Modelo fixo, sem busca de hiperparâmetros. A classe majoritária é a referência mínima. Avaliamos também uma primeira redação fixa de cada cenário de teste, sem votação entre variantes. O teste não foi usado para modificar os achados ou otimizar o modelo.""",
    ),
    (
        "code",
        """validation = validate_baseline(train)
model = train_model(train)
test_metrics = evaluate_model(model, test)
scenario_metrics = scenario_evaluation(model, test)
scores = [validation['majority_baseline']['macro_f1'], validation['model']['macro_f1'], test_metrics['macro_f1'], scenario_metrics['metrics']['macro_f1']]
fig, ax = plt.subplots()
bars = ax.barh(['Majoritário (validação)', 'Modelo (validação: 9 cenários)', 'Teste (600 redações)', 'Teste (30 cenários)'], scores, color=['#778899', '#227C9D', '#D99B23', '#227C9D'])
ax.bar_label(bars, fmt='%.3f', padding=4)
ax.set(xlim=(0, 1), xlabel='Macro-F1 (0–1)', title='Resultados sintéticos, sem interpretação clínica')
ax.invert_yaxis()
save_figure('model_performance')
display(pd.DataFrame(scenario_metrics['metrics']['per_class']).T.rename(index={str(k): v for k, v in LABEL_NAMES.items()}).round(3))
assert set(model.classes_) == {0, 1, 2}
""",
    ),
    ("md", "### Matriz de confusão: os erros são parte da entrega"),
    (
        "code",
        """matrix = np.asarray(scenario_metrics['metrics']['confusion_matrix'])
assert matrix.sum() == 30
fig, ax = plt.subplots(figsize=(6, 5))
heatmap = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=10)
for row in range(3):
    for col in range(3):
        ax.text(col, row, str(matrix[row, col]), ha='center', va='center', color='white' if matrix[row, col] > 5 else '#17324D', fontsize=16)
ax.set(xticks=range(3), yticks=range(3), xticklabels=names, yticklabels=names, xlabel='Predição', ylabel='Rótulo pedagógico', title='Teste: uma redação por cenário (n = 30)')
fig.colorbar(heatmap, ax=ax, label='Cenários')
save_figure('confusion_scenarios')
errors = pd.DataFrame(scenario_metrics['errors'])
display(errors)
display(Markdown(f"**{len(errors)}/30 cenários incorretos**, {scenario_metrics['underassigned']} subestimados e {scenario_metrics['overassigned']} superestimados. Urgente → atenção: {matrix[2, 1]}; urgente → normal: {matrix[2, 0]}. Zero nessa última célula não comprova segurança."))
""",
    ),
    (
        "md",
        """## Conclusões

1. **Alvo corrigido:** três urgências, não cinco doenças. Proveniência sintética acompanha os dados, o modelo e cada resposta da API. Nenhum mapeamento doença → urgência é aplicado.
2. **Diversidade é o gargalo:** 75 cenários autorais e redações repetidas sustentam apenas uma demonstração de engenharia. Mais de 2.000 linhas não equivalem a 2.000 casos independentes. Confirmar a adaptação com a avaliação acadêmica.
3. **Avaliação sem vazamento de família:** 36/9 cenários no holdout e 45/30 no treino/teste final. Compartilhamento de estilo permanece; não há validade externa.
4. **Desempenho limitado:** macro-F1 de validação 0,615 versus 0,167 do majoritário; teste por cenário 0,738. Oito erros em 30 cenários, incluindo três urgentes subestimados. A API não pode ser usada para decisões clínicas.
5. **Engenharia verificável:** contrato valida origem, três classes e isolamento; exportação ONNX só é aprovada após equivalência de rótulos/probabilidades e ganho de latência medido. O monitoramento mede serviço, não qualidade clínica.
6. **Evolução responsável:** uma aplicação real exigiria dados de laudos com urgência anotada por especialistas, acesso autorizado, validação externa e temporal, estudo de negações e calibração. Esses itens não foram realizados.

### Verificação

Notebook executado de cima para baixo. Contagens, partições e matriz reconciliadas. Sete figuras calculadas a partir dos CSVs gerados; sem pacientes reais. Ambiente abaixo. Confiança retornada é a maior probabilidade do modelo, **não calibrada** e não é probabilidade de risco clínico.""",
    ),
    ("code", "display(environment())"),
]


def main() -> None:
    """Executa em kernel limpo e exporta a mesma análise para HTML."""
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.cells = [
        nbformat.v4.new_markdown_cell(text)
        if kind == "md"
        else nbformat.v4.new_code_cell(text)
        for kind, text in CELLS
    ]
    target = ROOT / "notebooks/01_eda_synthetic_triage.ipynb"
    target.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, target)
    subprocess.run(
        [sys.executable, "-m", "ruff", "check", str(target), "--fix"], check=True
    )
    subprocess.run([sys.executable, "-m", "ruff", "format", str(target)], check=True)
    notebook = nbformat.read(target, as_version=4)
    NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, target)
    html, _ = HTMLExporter().from_notebook_node(notebook)
    (ROOT / "reports/eda.html").write_text(html, encoding="utf-8")
    print(f"Executed notebook saved: {target}")


if __name__ == "__main__":
    main()
