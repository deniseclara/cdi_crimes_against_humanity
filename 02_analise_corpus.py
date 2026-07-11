"""
Análise do corpus - Crimes contra a Humanidade / Sexta Comissão

Gera:
  Tabela 1 - Descrição do corpus por sessão
  Tabela 2 - Distribuição da EOPE por sessão
  Tabela 3 - TF-IDF distintivo por grupo (Favorável e Cético), SÓ UNIGRAMAS
  Tabela 4 - Casos qualitativos selecionados
  Figura 1 - Barras empilhadas EOPE por sessão
  Figura 2 - Dispersão TF-IDF (com adjustText)

Opcional, desligado por padrão (ver GERAR_APENDICE_BIGRAMAS abaixo):
  Tabela 5 - Bigramas mais frequentes e distintivos por grupo
  Figura 3 - Barras horizontais de bigramas por grupo

O artigo não inclui a camada de bigramas na versão final (nota de rodapé
sobre limite de extensão). O código continua aqui porque a análise foi
de fato executada; ative a flag abaixo se quiser reproduzir esses
resultados.

ATENÇÃO - pré-requisito não incluído neste repositório:
  Este script espera um CSV de entrada com as colunas:
    secao, num_reuniao, pais_porta_voz, texto, eh_coalizao, eh_observador,
    Posição
  As colunas eh_coalizao, eh_observador e Posição (a classificação EOPE
  vinda da CAHI) e a divisão de "secao" em "78ª Sessão (Out 2023)" vs.
  "78ª Sessão Reanimada (Abr 2024)" NÃO são produzidas por
  01_coleta_e_traducao.py. Essa etapa intermediária (provavelmente uma
  planilha de classificação manual cruzando os relatórios da CAHI com o
  corpus raspado) precisa ser documentada ou incluída neste repositório
  para o pipeline ser replicável de ponta a ponta.

Requisitos:
    pip install -r requirements.txt
"""

import re
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from adjustText import adjust_text

# ------------------------------------------------------------------
# CONFIGURAÇÃO
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_CSV = REPO_ROOT / "data" / "corpus_traduzido.csv"
PASTA_SAIDA = REPO_ROOT / "output"

N_TERMOS = 15       # termos TF-IDF por grupo (Tabela 3 / Figura 2)
N_BIGRAMAS = 20      # bigramas por grupo, só usado se GERAR_APENDICE_BIGRAMAS = True

# O artigo não usa a camada de bigramas na versão final. Mude para True
# só se quiser reproduzir essa análise exploratória.
GERAR_APENDICE_BIGRAMAS = False

CORES = {"Favorável": "#2166ac", "Cético": "#d6604d", "Neutro": "#f4a582"}
EOPE_ORDEM = ["Very Supportive", "Supportive", "Neutral", "Negative", "Opposed"]
EOPE_PT = {
    "Very Supportive": "Muito favorável",
    "Supportive": "Favorável",
    "Neutral": "Neutro",
    "Negative": "Negativo",
    "Opposed": "Contrário",
}
EOPE_CORES = {
    "Very Supportive": "#2166ac",
    "Supportive": "#74add1",
    "Neutral": "#ffffbf",
    "Negative": "#f46d43",
    "Opposed": "#d73027",
}
SECAO_ORDEM = [
    "78ª Sessão (Out 2023)",
    "78ª Sessão Reanimada (Abr 2024)",
    "79ª Sessão (Out 2024)",
]

PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# STOPWORDS
# ------------------------------------------------------------------
STOP_PAISES = [
    "china", "russia", "russian", "india", "iran", "islamic", "syria", "cuba",
    "nicaragua", "belarus", "venezuela", "egypt", "turkey", "ethiopia", "cameroon",
    "eritrea", "eritrean", "algeria", "qatar", "sudan", "saudi", "arabia", "vietnam",
    "pakistan", "dprk", "korea", "united states", "france", "germany", "uk",
    "britain", "australia", "canada", "japan", "singapore", "brazil", "mexico",
    "argentina", "colombia", "chile", "peru", "portugal", "romania", "netherlands",
    "switzerland", "austria", "ireland", "italy", "spain", "poland", "european",
    "union", "nordic", "canz", "baltic", "african", "arab", "grulac", "asean",
    "republic", "federation", "democratic", "peoples", "bolivarian", "plurinational",
    "holy see", "new zealand", "sri lanka", "kingdom", "emirates", "morocco",
    "moroccan", "senegal", "burkina", "faso", "liechtenstein", "djibouti",
    "indonesia", "malaysia", "philippines", "thai", "thailand", "sierra leone",
    "south africa",
]

STOP_PROC = [
    "madam", "chair", "mr", "ms", "chairman", "president", "delegation",
    "delegations", "statement", "behalf", "thank", "distinguished", "excellency",
    "colleagues", "colleague", "sixth", "committee", "agenda", "item", "session",
    "assembly", "general", "united", "nations", "government", "article", "draft",
    "articles", "crime", "crimes", "humanity", "against", "international", "law",
    "commission", "states", "state", "convention", "treaty", "paragraph",
    "paragraphs", "provisions", "provision", "text", "work", "noted", "expressed",
    "support", "view", "views", "also", "would", "may", "new york", "york", "april",
    "october", "november", "december", "january", "2023", "2024", "2025", "ilc",
    "icc", "resumed", "prevention", "punishment", "prevention punishment", "like",
    "said", "following", "manner", "ensure", "use", "made", "make", "take", "taken",
    "given", "need", "include", "included", "including", "regard", "regarding",
    "considered", "stated", "clear", "well", "important", "importance", "necessary",
    "however", "therefore", "furthermore", "moreover", "addition", "particular",
    "general", "specific", "especially", "relation", "connection", "accordance",
    "light", "basis", "terms", "note", "noting", "notes", "highlighted", "emphasized",
    "recalled", "called", "proposed", "suggested", "expressed", "indicated",
    "welcomed", "appreciated", "concerns", "concern", "question", "questions",
    "discussed", "raised", "added", "replied", "referred", "observed", "recognized",
    "acknowledged", "underlined", "stressed", "mentioned", "delivered", "adviser",
    "legal adviser", "representative", "permanent", "permanent representative",
    "mission", "recommendation", "substantive", "materials", "project", "section",
    "related", "year", "background", "reference", "annex", "document", "documents",
    "report", "reports", "working", "meeting", "meetings",
    # artefatos de OCR / tradução / letterhead
    "sirla", "desla", "yaludla", "ala", "don", "sleep", "oh", "la", "jl", "tion",
    "في", "لا", "er", "de", "du", "le", "les", "des", "un", "une", "et", "en",
    "para", "por", "com", "uma", "dos", "das", "pelo", "pela", "que", "ser",
    "god", "al", "relevant", "projects", "aligns", "organization", "organizations",
    "inshallah", "father", "son", "born", "birth", "welcome", "welcomes",
    "bilateral", "capacity", "forward", "continue",
    # artefatos de bigrama (mantidos porque o filtro de bigramas os usa também)
    "check delivery", "second second", "east street", "written comments",
    "judges observe", "statute criminal", "address issue",
]

STOP_SET = set(ENGLISH_STOP_WORDS) | set(STOP_PAISES) | set(STOP_PROC)
custom_stop = list(STOP_SET)

# ------------------------------------------------------------------
# 1) CARREGAR E PREPARAR
# ------------------------------------------------------------------
df = pd.read_csv(CORPUS_CSV, encoding="utf-8-sig")
estados = df[
    ~df["eh_coalizao"] &
    ~df["eh_observador"] &
    df["Posição"].notna()
].copy()


def atribuir_grupo(pos):
    if pos in ["Very Supportive", "Supportive"]:
        return "Favorável"
    elif pos == "Neutral":
        return "Neutro"
    else:
        return "Cético"


estados["grupo"] = estados["Posição"].apply(atribuir_grupo)


def limpar_texto(t):
    if not isinstance(t, str):
        return ""
    return " ".join(tok for tok in t.split() if tok.isascii() and len(tok) >= 3)


estados["texto_limpo"] = estados["texto"].apply(limpar_texto)

GRUPOS_TFIDF = ["Favorável", "Cético"]

# ------------------------------------------------------------------
# TABELA 1 - Descrição do corpus por sessão
# ------------------------------------------------------------------
print("Gerando Tabela 1...")
rows = []
for sec in SECAO_ORDEM:
    sub = df[df["secao"] == sec]
    sub_e = estados[estados["secao"] == sec]
    rows.append({
        "Sessão": sec,
        "Reuniões formais": sub["num_reuniao"].nunique(),
        "Total de discursos": len(sub),
        "Discursos de Estados (EOPE)": len(sub_e),
        "Países/entidades distintos": sub["pais_porta_voz"].nunique(),
    })
rows.append({
    "Sessão": "Total",
    "Reuniões formais": df["num_reuniao"].nunique(),
    "Total de discursos": len(df),
    "Discursos de Estados (EOPE)": len(estados),
    "Países/entidades distintos": df["pais_porta_voz"].nunique(),
})
tab1 = pd.DataFrame(rows)
tab1.to_excel(PASTA_SAIDA / "Tabela1_corpus_por_sessao.xlsx", index=False)
print("  -> Tabela1_corpus_por_sessao.xlsx")

# ------------------------------------------------------------------
# TABELA 2 - Distribuição EOPE por sessão
# ------------------------------------------------------------------
print("Gerando Tabela 2...")
tab2 = pd.crosstab(estados["Posição"], estados["secao"], margins=True, margins_name="Total")
tab2 = tab2.reindex(EOPE_ORDEM + ["Total"])
cols = [c for c in SECAO_ORDEM if c in tab2.columns] + ["Total"]
tab2 = tab2[cols]
tab2.index.name = "Posição EOPE"
tab2.to_excel(PASTA_SAIDA / "Tabela2_EOPE_por_sessao.xlsx")
print("  -> Tabela2_EOPE_por_sessao.xlsx")

# ------------------------------------------------------------------
# TABELA 3 - TF-IDF one-vs-rest (Favorável e Cético), SÓ UNIGRAMAS
#
# CORREÇÃO relativa à versão anterior: o vetorizador usava
# ngram_range=(1, 2), então bigramas ("genocide war", "internal affairs")
# entravam misturados no ranking de "termos mais distintivos" ao lado de
# unigramas de verdade. Isso não bate com o texto do artigo, que descreve
# a Tabela 3 como unigramas. Corrigido para ngram_range=(1, 1) abaixo.
#
# Nota: Neutro (n=16) excluído por insuficiência amostral.
# ------------------------------------------------------------------
print("Gerando Tabela 3 (TF-IDF, unigramas)...")

resultados_tfidf = {}
for g_alvo in GRUPOS_TFIDF:
    ta = estados[estados["grupo"] == g_alvo]["texto_limpo"].dropna().tolist()
    tr = estados[estados["grupo"] != g_alvo]["texto_limpo"].dropna().tolist()
    n_alvo = len(ta)

    vect = TfidfVectorizer(
        stop_words=custom_stop,
        ngram_range=(1, 1),  # unigramas apenas — ver nota acima
        max_features=10000,
        min_df=3,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vect.fit_transform(ta + tr)
    fn = vect.get_feature_names_out()

    d = X[:n_alvo].mean(0).A1 - X[n_alvo:].mean(0).A1
    top = d.argsort()[::-1][:N_TERMOS]
    resultados_tfidf[g_alvo] = [{"Termo": fn[i], "Score": round(d[i], 4)} for i in top]

tab3_data = {}
for g in GRUPOS_TFIDF:
    tab3_data[f"{g} — Termo"] = [t["Termo"] for t in resultados_tfidf[g]]
    tab3_data[f"{g} — Score"] = [t["Score"] for t in resultados_tfidf[g]]

tab3 = pd.DataFrame(tab3_data)
tab3.index = range(1, len(tab3) + 1)
tab3.index.name = "Rank"
tab3.to_excel(PASTA_SAIDA / "Tabela3_TFIDF_por_grupo.xlsx")
print("  -> Tabela3_TFIDF_por_grupo.xlsx")

# ------------------------------------------------------------------
# TABELA 4 - Casos qualitativos selecionados
# ------------------------------------------------------------------
print("Gerando Tabela 4...")
casos = [
    {"País": "Estados Unidos", "Sessão": "78ª Reanimada (Abr 2024)", "EOPE": "Very Supportive", "Coalizão hipotética": "Norte Global", "Critério": "Caso típico"},
    {"País": "União Europeia", "Sessão": "78ª Reanimada (Abr 2024)", "EOPE": "Very Supportive", "Coalizão hipotética": "Norte Global", "Critério": "Caso típico"},
    {"País": "Brasil", "Sessão": "78ª Reanimada (Abr 2024)", "EOPE": "Very Supportive", "Coalizão hipotética": "Sul Proativo", "Critério": "Caso típico"},
    {"País": "México", "Sessão": "78ª Reanimada (Abr 2024)", "EOPE": "Very Supportive", "Coalizão hipotética": "Sul Proativo", "Critério": "Caso típico"},
    {"País": "Rússia", "Sessão": "78ª Reanimada (Abr 2024)", "EOPE": "Opposed", "Coalizão hipotética": "Coalizão soberanista", "Critério": "Caso típico"},
    {"País": "China", "Sessão": "78ª Reanimada (Abr 2024)", "EOPE": "Opposed", "Coalizão hipotética": "Coalizão soberanista", "Critério": "Caso típico"},
    {"País": "Camarões", "Sessão": "78ª Reanimada (Abr 2024)", "EOPE": "Negative", "Coalizão hipotética": "N/A", "Critério": "Caso desviante: país africano no grupo cético"},
    {"País": "Singapura", "Sessão": "78ª Reanimada (Abr 2024)", "EOPE": "Supportive", "Coalizão hipotética": "N/A", "Critério": "Caso desviante: Ásia em posição intermediária"},
]
tab4 = pd.DataFrame(casos)
tab4.to_excel(PASTA_SAIDA / "Tabela4_casos_qualitativos.xlsx", index=False)
print("  -> Tabela4_casos_qualitativos.xlsx")

# ------------------------------------------------------------------
# FIGURA 1 - Barras empilhadas EOPE por sessão
# ------------------------------------------------------------------
print("\nGerando Figura 1...")
tab2_fig = pd.crosstab(estados["secao"], estados["Posição"])
tab2_fig = tab2_fig.reindex(SECAO_ORDEM)
tab2_fig = tab2_fig[[c for c in EOPE_ORDEM if c in tab2_fig.columns]]

labels_eixo = {
    "78ª Sessão (Out 2023)": "78ª\n(Out 2023)",
    "78ª Sessão Reanimada (Abr 2024)": "78ª Reanimada\n(Abr 2024)",
    "79ª Sessão (Out 2024)": "79ª\n(Out 2024)",
}
tab2_fig.index = [labels_eixo.get(i, i) for i in tab2_fig.index]

fig, ax = plt.subplots(figsize=(10, 5))
bottom = np.zeros(len(tab2_fig))

for pos in EOPE_ORDEM:
    if pos not in tab2_fig.columns:
        continue
    vals = tab2_fig[pos].values
    bars = ax.bar(tab2_fig.index, vals, bottom=bottom, color=EOPE_CORES[pos],
                  label=EOPE_PT[pos], edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        if v > 5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                     str(int(v)), ha="center", va="center", fontsize=8.5,
                     color="black" if pos == "Neutral" else "white", fontweight="bold")
    bottom += vals

ax.set_xlabel("Sessão", fontsize=11)
ax.set_ylabel("Número de discursos", fontsize=11)
ax.set_title("Figura 1 — Distribuição da EOPE por sessão\n"
             "(discursos de Estados; coalizões e observadores excluídos)", fontsize=11, pad=12)
ax.legend(title="Posição EOPE", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "Figura1_EOPE_por_sessao.png", dpi=300, bbox_inches="tight")
plt.savefig(PASTA_SAIDA / "Figura1_EOPE_por_sessao.pdf", bbox_inches="tight")
plt.close()
print("  -> Figura1_EOPE_por_sessao.png / .pdf")

# ------------------------------------------------------------------
# FIGURA 2 - Dispersão TF-IDF com adjustText (SÓ UNIGRAMAS, mesma correção da Tabela 3)
# ------------------------------------------------------------------
print("Gerando Figura 2...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

for ax, g_alvo in zip(axes, GRUPOS_TFIDF):
    ta = estados[estados["grupo"] == g_alvo]["texto_limpo"].dropna().tolist()
    tr = estados[estados["grupo"] != g_alvo]["texto_limpo"].dropna().tolist()
    n_alvo = len(ta)

    vect = TfidfVectorizer(
        stop_words=custom_stop,
        ngram_range=(1, 1),  # unigramas apenas — ver nota na Tabela 3
        max_features=10000,
        min_df=3,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vect.fit_transform(ta + tr)
    fn = vect.get_feature_names_out()

    d = X[:n_alvo].mean(0).A1 - X[n_alvo:].mean(0).A1
    freq = (X[:n_alvo] > 0).mean(0).A1
    top = d.argsort()[::-1][:N_TERMOS]

    xs, ys, labs = freq[top], d[top], fn[top]
    ax.scatter(xs, ys, color=CORES[g_alvo], s=70, zorder=3, alpha=0.9)
    texts = [ax.text(xi, yi, lab, fontsize=8, color="#1a1a1a") for xi, yi, lab in zip(xs, ys, labs)]
    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
        expand_points=(1.5, 1.8), expand_text=(1.4, 1.6),
        force_points=(0.4, 0.5), force_text=(0.5, 0.6),
        only_move={"points": "xy", "text": "xy"},
    )
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title(g_alvo, fontsize=13, fontweight="bold", color=CORES[g_alvo], pad=10)
    ax.set_xlabel("Proporção de documentos\ndo grupo que contêm o termo", fontsize=9)
    if ax is axes[0]:
        ax.set_ylabel("Score TF-IDF distintivo\n(grupo vs. demais)", fontsize=9)
    sns.despine(ax=ax)

fig.suptitle("Figura 2 — Vocabulário distintivo por coalizão discursiva (TF-IDF one-vs-rest)\n"
             "Eixo X: frequência do termo no grupo  |  Eixo Y: exclusividade vs. demais grupos",
             fontsize=10, y=1.03)
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "Figura2_TFIDF_dispersao.png", dpi=300, bbox_inches="tight")
plt.savefig(PASTA_SAIDA / "Figura2_TFIDF_dispersao.pdf", bbox_inches="tight")
plt.close()
print("  -> Figura2_TFIDF_dispersao.png / .pdf")

# ------------------------------------------------------------------
# APÊNDICE OPCIONAL - Tabela 5 e Figura 3 (bigramas), desligado por padrão
# ------------------------------------------------------------------
if GERAR_APENDICE_BIGRAMAS:
    print("\nGERAR_APENDICE_BIGRAMAS=True: gerando Tabela 5 e Figura 3 (bigramas)...")

    STOP_TOKENS_BIGRAMA = STOP_SET | {
        "check", "delivery", "east", "street", "second", "written", "comments",
        "judges", "observe", "please", "statute", "address", "leone", "africa",
        "west", "north", "sierra", "south",
    }
    BIGRAMAS_EXCLUIR = {
        "check delivery", "east street", "second second", "written comments",
        "judges observe", "please check", "statute criminal", "address issue",
        "sierra leone", "south africa", "west africa", "north korea",
        "check upon", "general assembly",
    }

    def extrair_bigramas(texto, stop_tok, excluir):
        if not isinstance(texto, str):
            return []
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", texto.lower())
        tokens = [t for t in tokens if t not in stop_tok]
        bigs = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
        return [b for b in bigs if b not in excluir]

    contagens, n_docs = {}, {}
    for g in GRUPOS_TFIDF:
        textos = estados[estados["grupo"] == g]["texto"].dropna()
        todos = []
        for t in textos:
            todos.extend(extrair_bigramas(t, STOP_TOKENS_BIGRAMA, BIGRAMAS_EXCLUIR))
        contagens[g] = Counter(todos)
        n_docs[g] = len(textos)

    tab5_rows = []
    for g_alvo in GRUPOS_TFIDF:
        g_outro = [g for g in GRUPOS_TFIDF if g != g_alvo][0]
        top = contagens[g_alvo].most_common(N_BIGRAMAS * 3)
        adicionados = 0
        for bigrama, freq in top:
            if adicionados >= N_BIGRAMAS:
                break
            freq_outro = contagens[g_outro].get(bigrama, 0)
            prev_alvo = freq / n_docs[g_alvo]
            prev_outro = freq_outro / n_docs[g_outro] if n_docs[g_outro] > 0 else 0
            razao = round(prev_alvo / prev_outro, 1) if prev_outro > 0 else float("inf")
            tab5_rows.append({
                "Grupo": g_alvo, "Bigrama": bigrama,
                "Freq. no grupo": freq, "Freq. no outro": freq_outro,
                "Razão de prevalência": razao,
            })
            adicionados += 1

    tab5 = pd.DataFrame(tab5_rows)
    tab5.to_excel(PASTA_SAIDA / "Tabela5_bigramas_por_grupo.xlsx", index=False)
    print("  -> Tabela5_bigramas_por_grupo.xlsx")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=False)
    for ax, g_alvo in zip(axes, GRUPOS_TFIDF):
        sub = tab5[tab5["Grupo"] == g_alvo].head(N_BIGRAMAS)
        bigramas = sub["Bigrama"].tolist()[::-1]
        freqs = sub["Freq. no grupo"].tolist()[::-1]
        razoes = sub["Razão de prevalência"].tolist()[::-1]
        bars = ax.barh(bigramas, freqs, color=CORES[g_alvo], alpha=0.85, edgecolor="white")
        for bar, razao in zip(bars, razoes):
            rotulo = f"×{razao:.1f}" if razao != float("inf") else "excl."
            ax.text(bar.get_width() + max(freqs) * 0.01, bar.get_y() + bar.get_height() / 2,
                    rotulo, va="center", fontsize=7.5, color="#555555")
        ax.set_title(g_alvo, fontsize=12, fontweight="bold", color=CORES[g_alvo], pad=8)
        ax.set_xlabel("Frequência absoluta no grupo", fontsize=9)
        ax.tick_params(axis="y", labelsize=8.5)
        sns.despine(ax=ax)

    fig.suptitle("Figura 3 — Bigramas mais frequentes por coalizão discursiva (apêndice)\n"
                 "× = razão de prevalência: quantas vezes mais frequente no grupo vs. no outro grupo",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(PASTA_SAIDA / "Figura3_bigramas_por_grupo.png", dpi=300, bbox_inches="tight")
    plt.savefig(PASTA_SAIDA / "Figura3_bigramas_por_grupo.pdf", bbox_inches="tight")
    plt.close()
    print("  -> Figura3_bigramas_por_grupo.png / .pdf")

# ------------------------------------------------------------------
# RESUMO FINAL
# ------------------------------------------------------------------
print("\n=== Concluído ===")
print(f"Arquivos em: {PASTA_SAIDA}")
print("  Tabela1_corpus_por_sessao.xlsx")
print("  Tabela2_EOPE_por_sessao.xlsx")
print("  Tabela3_TFIDF_por_grupo.xlsx")
print("  Tabela4_casos_qualitativos.xlsx")
print("  Figura1_EOPE_por_sessao.png / .pdf")
print("  Figura2_TFIDF_dispersao.png / .pdf")
if GERAR_APENDICE_BIGRAMAS:
    print("  Tabela5_bigramas_por_grupo.xlsx")
    print("  Figura3_bigramas_por_grupo.png / .pdf")
