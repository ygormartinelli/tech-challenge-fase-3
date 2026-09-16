"""Cenários fictícios autorais; rótulos pedagógicos, sem validação clínica."""

# Os cinco temas remetem ao corpus sugerido, sem converter seus rótulos.
# Cada frase é um cenário-base. Variações do mesmo cenário ficam juntas.
SCENARIOS = {
    "neoplasms": {
        0: (
            "Avaliação de controle sem lesão expansiva ou formação nodular suspeita.",
            "Não há massa identificável nem evidência de crescimento tumoral.",
            "Exame de acompanhamento sem alteração focal suspeita de neoplasia.",
            "Ausência de nódulos suspeitos, com estruturas preservadas no exame.",
            "Estudo sem sinais de massa ou comprometimento de estruturas adjacentes.",
        ),
        1: (
            "Nódulo indeterminado pequeno, sem invasão, com aspecto estável no controle.",
            "Lesão focal de crescimento lento, sem obstrução ou sangramento associado.",
            "Formação nodular localizada sem compressão de estruturas vizinhas.",
            "Espessamento focal persistente e estável, sem sinais de complicação aguda.",
            "Massa conhecida com dimensões inalteradas, sem compressão de via aérea.",
        ),
        2: (
            "Massa expansiva com compressão importante da via aérea e dificuldade respiratória intensa.",
            "Lesão tumoral associada a hemorragia ativa volumosa e instabilidade circulatória.",
            "Formação expansiva com compressão medular aguda e perda recente de força nas pernas.",
            "Massa com obstrução completa da via respiratória e queda acentuada de oxigenação.",
            "Lesão expansiva intracraniana com herniação e rebaixamento súbito de consciência.",
        ),
    },
    "digestive": {
        0: (
            "Exame abdominal sem sinais de inflamação, dilatação ou obstrução intestinal.",
            "Alças intestinais de calibre preservado, sem perfuração ou líquido livre.",
            "Fígado e vias biliares sem alterações relevantes no estudo.",
            "Não se identificam coleções, sangramento ou alterações abdominais agudas.",
            "Avaliação digestiva sem achados patológicos e com trânsito preservado.",
        ),
        1: (
            "Esteatose hepática discreta, sem sinais de complicação aguda.",
            "Cálculo vesicular sem dilatação biliar, febre ou inflamação da parede.",
            "Divertículos sem inflamação, perfuração ou abscesso associado.",
            "Espessamento gástrico discreto e estável, sem sangramento ativo.",
            "Pequeno cisto hepático estável, sem obstrução das vias biliares.",
        ),
        2: (
            "Ar livre na cavidade abdominal com perfuração intestinal e sinais de choque.",
            "Obstrução intestinal completa com isquemia de alça e dor abdominal intensa.",
            "Sangramento digestivo ativo volumoso com queda importante da pressão arterial.",
            "Ruptura de órgão abdominal com grande hemoperitônio e instabilidade circulatória.",
            "Perfuração de víscera com coleção extensa e deterioração clínica rápida.",
        ),
    },
    "neurologic": {
        0: (
            "Estudo do encéfalo sem hemorragia, efeito de massa ou alteração focal aguda.",
            "Não há sinais de isquemia recente, edema ou desvio de estruturas cerebrais.",
            "Estruturas encefálicas preservadas, sem lesão aguda identificável.",
            "Avaliação neurológica sem alteração estrutural ou compressão medular.",
            "Ausência de sangramento intracraniano e de dilatação ventricular.",
        ),
        1: (
            "Alterações crônicas de substância branca, sem lesão isquêmica recente.",
            "Atrofia cerebral discreta, sem efeito de massa ou hemorragia.",
            "Pequena lesão sequelar estável, sem sinais de edema ou sangramento recente.",
            "Protrusão discal crônica sem compressão medular ou déficit motor novo.",
            "Calcificação intracraniana estável, sem sinais de complicação aguda.",
        ),
        2: (
            "Hemorragia intracraniana extensa com desvio da linha média e perda de consciência.",
            "Oclusão arterial cerebral aguda com perda súbita da fala e fraqueza de um lado.",
            "Edema cerebral acentuado com sinais de herniação e deterioração rápida.",
            "Hematoma intracraniano em expansão com compressão importante do encéfalo.",
            "Compressão aguda da medula com paralisia de instalação recente.",
        ),
    },
    "cardiovascular": {
        0: (
            "Avaliação cardiovascular sem sinais de isquemia aguda ou alteração estrutural.",
            "Ritmo regular e função cardíaca preservada, sem derrame pericárdico.",
            "Vasos de calibre preservado, sem dissecção, trombo ou ruptura.",
            "Não há sinais de sobrecarga cardíaca ou comprometimento circulatório.",
            "Estudo cardíaco sem alterações relevantes e sem sinais de evento agudo.",
        ),
        1: (
            "Calcificação vascular discreta, sem obstrução significativa ou dissecção.",
            "Insuficiência valvar leve e estável, sem repercussão hemodinâmica.",
            "Pequeno derrame pericárdico estável, sem compressão das câmaras cardíacas.",
            "Espessamento valvar discreto, com função ventricular mantida.",
            "Alterações crônicas do ritmo, sem instabilidade circulatória ou dor atual.",
        ),
        2: (
            "Dissecção aguda de aorta com sinais de ruptura e instabilidade circulatória.",
            "Oclusão coronariana aguda com dor torácica intensa e sinais de choque.",
            "Derrame pericárdico volumoso com compressão cardíaca e queda importante da pressão.",
            "Tromboembolismo extenso com comprometimento hemodinâmico e hipoxemia intensa.",
            "Arritmia sustentada com perda de consciência e circulação comprometida.",
        ),
    },
    "general": {
        0: (
            "Exame de rotina dentro dos limites de referência, sem alteração relevante.",
            "Não há sinais de processo inflamatório, infeccioso ou lesão aguda.",
            "Estruturas avaliadas preservadas, sem coleção ou achado patológico.",
            "Avaliação sem sinais de sangramento, fratura ou comprometimento respiratório.",
            "Resultados de controle sem alterações significativas ou achados novos.",
        ),
        1: (
            "Alteração inflamatória discreta e localizada, sem coleção ou repercussão sistêmica.",
            "Pequena opacidade residual estável, sem insuficiência respiratória.",
            "Alteração laboratorial leve e persistente, sem sinais de deterioração clínica.",
            "Lesão superficial limitada, sem comprometimento vascular ou infeccioso profundo.",
            "Achado crônico estável, sem sangramento ou comprometimento de órgãos.",
        ),
        2: (
            "Pneumotórax extenso sob tensão com desvio de estruturas e dificuldade respiratória intensa.",
            "Infecção disseminada com hipotensão persistente e deterioração rápida de órgãos.",
            "Hemorragia ativa extensa após trauma com instabilidade circulatória.",
            "Comprometimento respiratório agudo grave com queda acentuada de oxigenação.",
            "Lesão traumática extensa com sangramento volumoso e perda de consciência.",
        ),
    },
}

PREFIXES = (
    "Laudo descritivo.",
    "Resultado do exame.",
    "Descrição dos achados observados.",
    "Registro da avaliação complementar.",
    "Síntese dos achados do estudo.",
    "Relato do exame realizado.",
)
SUFFIXES = (
    "Aquisição concluída com qualidade técnica satisfatória.",
    "Descrição referente às estruturas incluídas neste exame.",
    "Estudo realizado conforme a técnica habitual.",
    "Imagens disponíveis para revisão do examinador.",
    "Registro elaborado a partir da avaliação atual.",
    "Exame concluído sem intercorrência técnica.",
    "Achados descritos no campo de observações.",
    "Avaliação restrita à região examinada.",
    "Relatório disponibilizado para conferência.",
    "Descrição finalizada após revisão das imagens.",
)
