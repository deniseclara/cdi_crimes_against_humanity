"""
Coleta e tradução do corpus - Crimes Against Humanity, Sixth Committee (UN)

Este script junta as duas etapas que antes eram scripts separados:
  1) Raspagem dos PDFs de discursos nas páginas de cada sessão e extração de texto.
  2) Tradução para o inglês (via Google Cloud Translation API) dos discursos que
     não estão originalmente em inglês.

Saída: um único CSV com o corpus bruto raspado + colunas de tradução
       (texto_original, idioma_original, traduzido).

IMPORTANTE - o que este script NÃO faz:
  Ele não atribui a Escala Ordinal de Posicionamento Estatal (EOPE/CAHI) a cada
  discurso, não marca quais linhas são de coalizões/observadores (eh_coalizao,
  eh_observador), e não separa a sessão "78" em "78ª Sessão (Out 2023)" vs.
  "78ª Sessão Reanimada (Abr 2024)". Essas colunas são usadas pelo script de
  análise (02_analise_corpus.py) mas são produzidas por uma etapa de
  classificação manual/adicional que não fazia parte dos arquivos originais.
  Documente essa etapa antes de considerar o pipeline replicável de ponta a
  ponta (ver README.md).

Requisitos:
    pip install -r requirements.txt

Configuração:
    Copie .env.example para .env e preencha GOOGLE_API_KEY.
    Nunca coloque a chave diretamente no código.
"""

import os
import re
import time
import argparse
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
import pdfplumber
from dotenv import load_dotenv

load_dotenv()  # lê o arquivo .env, se existir

DetectorFactory.seed = 0

# ------------------------------------------------------------------
# CONFIGURAÇÃO (caminhos relativos ao repositório, não mais absolutos)
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

SESSIONS = ["78", "79"]
BASE_PAGE = "https://www.un.org/en/ga/sixth/{s}/cah.shtml"
BASE_PDF = "https://www.un.org"
HEADERS = {"User-Agent": "Mozilla/5.0 (Academic research scraper - contact: [seu e-mail de contato aqui])"}

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
URL_API_TRADUCAO = "https://translation.googleapis.com/language/translate/v2"
TAMANHO_MAXIMO_BLOCO = 4500
MAPA_CODIGO_IDIOMA = {"zh-cn": "zh-CN", "zh-tw": "zh-TW"}

CORPUS_BRUTO_CSV = DATA_DIR / "corpus_bruto.csv"
CORPUS_TRADUZIDO_CSV = DATA_DIR / "corpus_traduzido.csv"


# ------------------------------------------------------------------
# ETAPA 1 - RASPAGEM E EXTRAÇÃO DE TEXTO
# ------------------------------------------------------------------
def slug_to_key(slug: str) -> str:
    return re.sub(r"[^a-z_]", "", slug.lower())


def raspar_sessoes(sessions: list[str], pasta_dados: Path) -> pd.DataFrame:
    """Baixa os PDFs de cada sessão listada e extrai o texto de cada um."""
    records = []
    pasta_dados.mkdir(parents=True, exist_ok=True)

    for sessao in sessions:
        url_pagina = BASE_PAGE.format(s=sessao)
        print(f"[Sessão {sessao}] Buscando página: {url_pagina}")
        resp = requests.get(url_pagina, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.find_all("a", href=re.compile(r"pdfs/statements/cah/.*\.pdf", re.I))
        print(f"  -> {len(links)} links de PDF encontrados")

        pasta_sessao = pasta_dados / f"sessao_{sessao}"
        pasta_sessao.mkdir(parents=True, exist_ok=True)

        for link in links:
            href = link["href"]
            pdf_url = href if href.startswith("http") else BASE_PDF + href
            pais_link_text = link.get_text(strip=True)

            filename = os.path.basename(href)
            m = re.match(r"(\d+)mtg_([a-z_]+)", filename, re.I)
            num_reuniao = m.group(1) if m else "NA"
            pais_slug = m.group(2) if m else filename
            pais_key = slug_to_key(pais_slug)

            contexto = str(link) + "".join(str(s) for s in link.next_siblings)[:400]
            m_behalf = re.search(r"\(on\s+behalf\s+of\s+([^)]+)\)", contexto, re.I)
            representados = m_behalf.group(1).strip() if m_behalf else ""

            pdf_path = pasta_sessao / filename
            texto_extraido = ""
            try:
                if not pdf_path.exists():
                    r_pdf = requests.get(pdf_url, headers=HEADERS, timeout=20)
                    content_type = r_pdf.headers.get("Content-Type", "").lower()
                    if r_pdf.status_code == 200 and content_type.startswith("application/pdf"):
                        pdf_path.write_bytes(r_pdf.content)
                    else:
                        print(f"    [AVISO] Não é PDF válido: {pdf_url}")
                        continue
                    time.sleep(0.5)  # cortesia ao servidor

                with pdfplumber.open(pdf_path) as pdf:
                    texto_extraido = "\n".join((page.extract_text() or "") for page in pdf.pages)
            except Exception as e:
                print(f"    [ERRO] {pdf_url}: {e}")
                texto_extraido = ""

            records.append({
                "sessao": sessao,
                "num_reuniao": num_reuniao,
                "pais_porta_voz": pais_link_text,
                "pais_slug": pais_slug,
                "representa_grupo": bool(representados),
                "paises_representados": representados,
                "url_pdf": pdf_url,
                "caminho_arquivo": str(pdf_path),
                "texto": texto_extraido,
                "n_caracteres_texto": len(texto_extraido),
            })

    return pd.DataFrame(records)


# ------------------------------------------------------------------
# ETAPA 2 - DETECÇÃO DE IDIOMA E TRADUÇÃO
# ------------------------------------------------------------------
def detectar_idioma(texto: str) -> str:
    if not isinstance(texto, str) or len(texto.strip()) < 20:
        return "en"
    try:
        return detect(texto)
    except Exception:
        return "en"


def dividir_em_blocos(texto: str, tamanho_maximo: int) -> list[str]:
    paragrafos = texto.split("\n")
    blocos, atual = [], ""
    for p in paragrafos:
        candidato = (atual + "\n" + p) if atual else p
        if len(candidato) > tamanho_maximo and atual:
            blocos.append(atual)
            atual = p
        else:
            atual = candidato
    if atual:
        blocos.append(atual)
    return blocos


def chamar_api_traducao(textos: list[str], idioma_origem: str, api_key: str) -> list[str]:
    origem = MAPA_CODIGO_IDIOMA.get(idioma_origem, idioma_origem)
    params = {"key": api_key}
    payload = {"q": textos, "source": origem, "target": "en", "format": "text"}
    resp = requests.post(URL_API_TRADUCAO, params=params, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Erro {resp.status_code} da API: {resp.text[:500]}")
    dados = resp.json()
    return [t["translatedText"] for t in dados["data"]["translations"]]


def traduzir_texto(texto: str, idioma_origem: str, api_key: str, tentativas: int = 3) -> str:
    blocos = dividir_em_blocos(texto, TAMANHO_MAXIMO_BLOCO)
    for tentativa in range(1, tentativas + 1):
        try:
            traduzidos = chamar_api_traducao(blocos, idioma_origem, api_key)
            return "\n".join(traduzidos)
        except Exception as e:
            print(f"    [ERRO tentativa {tentativa}/{tentativas}] {e}")
            time.sleep(2 * tentativa)
    print("    [FALHA] Não consegui traduzir esse discurso. Mantendo o original.")
    return texto


def traduzir_corpus(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    print("Detectando idioma de cada discurso...")
    df["idioma_original"] = df["texto"].apply(detectar_idioma)
    print(df["idioma_original"].value_counts())

    df["texto_original"] = df["texto"]
    df["traduzido"] = False

    nao_ingles = df[df["idioma_original"] != "en"]
    total_caracteres = nao_ingles["n_caracteres_texto"].sum()
    print(f"\nTraduzindo {len(nao_ingles)} discurso(s), {total_caracteres} caracteres no total...")
    print("(primeiros 500.000 caracteres/mês grátis na API do Google, depois ~US$20/milhão)\n")

    inicio = time.time()
    for i, (idx, row) in enumerate(nao_ingles.iterrows(), start=1):
        texto_traduzido = traduzir_texto(row["texto_original"], row["idioma_original"], api_key)
        df.at[idx, "texto"] = texto_traduzido
        df.at[idx, "traduzido"] = True
        if i % 10 == 0 or i == len(nao_ingles):
            decorrido = time.time() - inicio
            print(f"  {i}/{len(nao_ingles)} traduzidos ({decorrido:.0f}s decorridos)")

    print(f"\nTradução concluída em {time.time() - inicio:.0f} segundos.")
    return df


# ------------------------------------------------------------------
# EXECUÇÃO
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Coleta e traduz o corpus de discursos.")
    parser.add_argument("--pular-raspagem", action="store_true",
                         help="Pula a raspagem e usa o CSV bruto já existente em data/corpus_bruto.csv")
    parser.add_argument("--pular-traducao", action="store_true",
                         help="Faz só a raspagem, sem chamar a API de tradução")
    args = parser.parse_args()

    if args.pular_raspagem:
        if not CORPUS_BRUTO_CSV.exists():
            raise SystemExit(f"{CORPUS_BRUTO_CSV} não existe. Rode sem --pular-raspagem primeiro.")
        df = pd.read_csv(CORPUS_BRUTO_CSV, encoding="utf-8-sig")
    else:
        df = raspar_sessoes(SESSIONS, DATA_DIR)
        df.to_csv(CORPUS_BRUTO_CSV, index=False, encoding="utf-8-sig")
        print(f"\nCorpus bruto salvo em: {CORPUS_BRUTO_CSV} ({len(df)} discursos)")

    if args.pular_traducao:
        return

    if not GOOGLE_API_KEY:
        raise SystemExit(
            "GOOGLE_API_KEY não encontrada. Copie .env.example para .env e preencha a chave, "
            "ou rode com --pular-traducao para só coletar os dados."
        )

    df = traduzir_corpus(df, GOOGLE_API_KEY)
    df.to_csv(CORPUS_TRADUZIDO_CSV, index=False, encoding="utf-8-sig")
    print(f"Corpus traduzido salvo em: {CORPUS_TRADUZIDO_CSV}")


if __name__ == "__main__":
    main()
