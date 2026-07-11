# Coalizões discursivas na Sexta Comissão da ONU — corpus e análise

Código de coleta, tradução e análise textual dos discursos da Sexta Comissão
sobre os Draft Articles on Crimes against Humanity (sessões de 2023 e 2024),
referente ao artigo "Coalizões Discursivas na Sexta Comissão da ONU".

## Estrutura

```
src/
  01_coleta_e_traducao.py   raspagem dos PDFs + tradução via Google Cloud Translation API
  02_analise_corpus.py      TF-IDF, tabelas e figuras do artigo
data/                       CSVs brutos e traduzidos (gerados localmente, não versionados)
output/                     tabelas (.xlsx) e figuras (.png/.pdf) geradas pela análise
requirements.txt
.env.example
```

## Como rodar

```bash
pip install -r requirements.txt
cp .env.example .env   # preencha GOOGLE_API_KEY no .env
python src/01_coleta_e_traducao.py
python src/02_analise_corpus.py
```

Para rodar só a coleta, sem gastar cota da API de tradução:
```bash
python src/01_coleta_e_traducao.py --pular-traducao
```

## Lacuna conhecida no pipeline

`02_analise_corpus.py` espera um CSV com as colunas `secao`, `eh_coalizao`,
`eh_observador` e `Posição` (a classificação EOPE, derivada dos relatórios da
Crimes Against Humanity Initiative/CAHI). Essas colunas **não** são geradas
por `01_coleta_e_traducao.py`: existe uma etapa intermediária de
classificação (provavelmente manual, cruzando o corpus raspado com os
relatórios de posicionamento da CAHI e separando a sessão "78" em suas duas
janelas temporais, outubro de 2023 e abril de 2024) que ainda não está
documentada ou incluída neste repositório.

Antes de considerar o pipeline replicável de ponta a ponta por outra pessoa,
essa etapa precisa ganhar um script próprio (ex.: `01b_classificar_eope.py`)
ou, no mínimo, uma planilha de correspondência documentada no README.

## Nota sobre bigramas

O artigo não inclui a camada de análise de bigramas na versão final (ver
nota de rodapé sobre limite de extensão do trabalho). O código de
`02_analise_corpus.py` foi corrigido para gerar a Tabela 3 e a Figura 2
usando `ngram_range=(1, 1)` (só unigramas) — a versão anterior usava
`ngram_range=(1, 2)`, o que misturava bigramas como "genocide war" e
"internal affairs" no ranking de termos unigrama, sem que o texto do artigo
descrevesse isso. A análise de bigramas (Tabela 5 / Figura 3) continua no
script, mas desligada por padrão (`GERAR_APENDICE_BIGRAMAS = False`).

## Segurança

Nunca coloque chaves de API diretamente no código. Use o arquivo `.env`
(fora do controle de versão) e a variável `GOOGLE_API_KEY`.
