# Coalizões discursivas na Sexta Comissão da ONU — corpus e análise

Código de coleta, tradução e análise textual dos discursos da Sexta Comissão
da Assembleia Geral da ONU sobre os Draft Articles on Crimes against Humanity,
cobrindo as sessões de outubro de 2023, abril de 2024 e outubro de 2024.

Referência: [título completo do artigo, autora, ano, periódico/instituição]

## Estrutura

```
src/
  01_coleta_e_traducao.py   raspagem dos PDFs da ONU e tradução via Google Cloud Translation API
  02_analise_corpus.py      análise TF-IDF, geração de tabelas e figuras
data/                       CSVs gerados localmente (não versionados; ver .gitignore)
output/                     tabelas (.xlsx) e figuras (.png / .pdf) geradas pela análise
requirements.txt            dependências Python
.env.example                modelo de configuração de variáveis de ambiente
```

## Requisitos

- Python 3.10 ou superior
- Conta no Google Cloud com a Translation API habilitada (necessária apenas para a etapa de tradução)

## Como rodar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Configurar a chave de API
cp .env.example .env
# abra o .env e preencha GOOGLE_API_KEY com sua chave

# 3. Coletar e traduzir o corpus
python src/01_coleta_e_traducao.py

# 4. Gerar tabelas e figuras
python src/02_analise_corpus.py
```

Para coletar os PDFs sem chamar a API de tradução (útil para inspeção do corpus bruto):

```bash
python src/01_coleta_e_traducao.py --pular-traducao
```

Para rodar a análise usando um corpus já existente em `data/corpus_traduzido.csv`:

```bash
python src/01_coleta_e_traducao.py --pular-raspagem
python src/02_analise_corpus.py
```

## Saídas de `02_analise_corpus.py`

| Arquivo | Conteúdo |
|---|---|
| `Tabela1_corpus_por_sessao.xlsx` | Corpus por sessão: reuniões, discursos e países |
| `Tabela2_EOPE_por_sessao.xlsx` | Distribuição da EOPE por sessão |
| `Tabela3_TFIDF_por_grupo.xlsx` | Quinze termos com maior score TF-IDF one-vs-rest por grupo (unigramas) |
| `Tabela4_casos_qualitativos.xlsx` | Casos selecionados para análise qualitativa |
| `Figura1_EOPE_por_sessao.png/.pdf` | Posicionamentos EOPE por sessão |
| `Figura2_TFIDF_dispersao.png/.pdf` | Vocabulário distintivo por coalizão (frequência × exclusividade) |

## Nota sobre a análise de bigramas

`02_analise_corpus.py` realiza a análise lexical usando unigramas (`ngram_range=(1, 1)`),
que é a camada reportada no artigo. O script também contém uma camada exploratória
de bigramas (Tabela 5 / Figura 3), não incluída na versão final do artigo por
razões de extensão. Essa camada está desligada por padrão e pode ser ativada
alterando a flag `GERAR_APENDICE_BIGRAMAS = True` no início do script.

## Etapa intermediária não incluída neste repositório

`02_analise_corpus.py` espera um CSV com as colunas `secao`, `eh_coalizao`,
`eh_observador` e `Posição`. Essas colunas não são geradas por
`01_coleta_e_traducao.py`: resultam de uma etapa de classificação que cruza
o corpus raspado com os relatórios de posicionamento da Crimes Against Humanity
Initiative (CAHI) e distingue as duas janelas temporais da 78ª Sessão (outubro
de 2023 e abril de 2024). Essa etapa não está documentada neste repositório.
Para acesso ao corpus já classificado ou dúvidas sobre replicação, entre em
contato com a autora.

## Segurança

Não inclua chaves de API no código-fonte. Armazene a chave no arquivo `.env`
(excluído do controle de versão pelo `.gitignore`) e acesse-a via variável de
ambiente `GOOGLE_API_KEY`, conforme o modelo em `.env.example`.
