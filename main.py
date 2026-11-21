import io
import os
from textwrap import dedent

import streamlit as st
from dotenv import load_dotenv
from PIL import Image as PILImage

from agno.agent import Agent
from agno.media import Image as AgnoImage
from agno.models.anthropic import Claude
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from agno.tools.tavily import TavilyTools

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ---------------------------------------------------------------------------
# Provedores de IA disponíveis
# Para incluir um modelo novo, basta acrescentar o id na lista "modelos".
# ---------------------------------------------------------------------------
PROVEDORES = {
    "Claude (Anthropic)": {
        "classe": Claude,
        "modelos": ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"],
        "onde_obter": "https://console.anthropic.com/settings/keys",
    },
    "Gemini (Google)": {
        "classe": Gemini,
        "modelos": ["gemini-2.5-flash", "gemini-2.5-pro"],
        "onde_obter": "https://aistudio.google.com/apikey",
    },
    "OpenAI (ChatGPT)": {
        "classe": OpenAIChat,
        "modelos": ["gpt-4.1", "gpt-4.1-mini", "gpt-4o"],
        "onde_obter": "https://platform.openai.com/api-keys",
    },
}

# Etapas do pipeline, na ordem, com o percentual da barra ao iniciar cada uma
ETAPAS = [
    ("preparo", "🖼️ Redimensionando a imagem", 10),
    ("analise", "🩻 Med está analisando a imagem", 25),
    ("busca", "📚 Rio está buscando evidências científicas", 60),
    ("formato", "📝 Formatando o resultado", 90),
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
PROMPT_ANALISE = """
Você é um especialista em diagnóstico por imagem.
Analise a imagem médica e organize a resposta em português com as seguintes seções:

### 1. Tipo de imagem e região
- Identifique o tipo de exame (raio-X, ressonância, tomografia, etc.).
- Indique a região anatômica e a qualidade técnica.

### 2. Achados relevantes
- Liste achados visuais significativos.
- Aponte possíveis anomalias.

### 3. Avaliação diagnóstica
- Forneça diagnóstico principal com nível de confiança (alto, moderado, baixo).
- Liste diagnósticos diferenciais e justificativas visuais.

### 4. Resumo em linguagem simples
- Resuma os achados em linguagem simples, para o profissional usar ao explicar o exame ao paciente.
"""

PROMPT_BUSCA = """Com base na seguinte análise de imagem médica, realize uma pesquisa complementar.
 - Utilize a busca do Tavily para encontrar artigos e protocolos atuais (de preferência PubMed e diretrizes oficiais).
 - Forneça 2 a 3 links ou referências confiáveis.
 - Organize a resposta em markdown.

Resultado da análise médica: "{}"
"""

INSTRUCOES_PESQUISADOR = dedent("""
    Você é um pesquisador médico responsável por buscar informações complementares sobre os achados identificados.
    Forneça literatura recente e fontes confiáveis.
""")


# ---------------------------------------------------------------------------
# Funções do pipeline
# ---------------------------------------------------------------------------
def preprocess_img(arquivo, largura=600):
    """Redimensiona a imagem em memória (sem gravar arquivo temporário)."""
    imagem = PILImage.open(arquivo).convert("RGB")
    proporcao = imagem.height / imagem.width
    redimensionada = imagem.resize((largura, int(largura * proporcao)))
    buffer = io.BytesIO()
    redimensionada.save(buffer, format="PNG")
    return AgnoImage(content=buffer.getvalue(), format="png")


def format_res(res):
    """Remove marcações extras que alguns modelos devolvem."""
    res = (res or "").strip()
    if "</think>" in res:
        res = res.split("</think>")[-1].strip()
    return res.replace("```", "")


def criar_agentes(provedor, modelo, api_key):
    """Cria os dois agentes com o modelo e a chave escolhidos pelo usuário."""
    classe = PROVEDORES[provedor]["classe"]
    med = Agent(
        name="Med",
        role="Especialista em imagens médicas",
        model=classe(id=modelo, api_key=api_key),
        markdown=True,
    )
    rio = Agent(
        name="Rio",
        role="Pesquisador médico",
        instructions=INSTRUCOES_PESQUISADOR,
        model=classe(id=modelo, api_key=api_key),
        tools=[TavilyTools(api_key=TAVILY_API_KEY)],
        markdown=True,
    )
    return med, rio


def process_img_pipeline(arquivo, med, rio, avisar):
    """Executa as etapas e chama avisar(etapa) no início de cada uma."""
    avisar("preparo")
    imagem = preprocess_img(arquivo)

    avisar("analise")
    analise = med.run(PROMPT_ANALISE, images=[imagem]).content

    avisar("busca")
    pesquisa = rio.run(PROMPT_BUSCA.format(analise)).content

    avisar("formato")
    resultado = f"### 🩻 Resultado da Análise da Imagem\n{format_res(analise)}\n\n---\n\n"
    resultado += f"### 📚 Pesquisa Complementar\n{format_res(pesquisa)}"
    return resultado


def mensagem_de_erro(erro):
    """Traduz os erros mais comuns das APIs para o usuário."""
    texto = str(erro).lower()
    if any(p in texto for p in ("401", "unauthorized", "invalid api key", "api key not valid", "authentication", "permission")):
        return "A chave de API foi recusada. Confira se ela está correta e é do provedor escolhido."
    if any(p in texto for p in ("404", "not found", "does not exist")):
        return "Esse modelo não está disponível para a sua chave. Escolha outro modelo na barra lateral."
    if any(p in texto for p in ("429", "rate limit", "quota", "credit", "billing")):
        return "Limite de uso ou créditos esgotados nessa chave. Tente mais tarde ou use outra chave."
    return f"Não foi possível concluir a análise: {erro}"


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Med Agent", layout="centered", page_icon="🩻")

st.title("🩺 Análise de Imagens Médicas")
st.markdown("""
O **Med** é um especialista em diagnóstico por imagem e trabalha com o **Rio**, que é pesquisador médico.
Carregue uma **imagem médica** (fotografia, raio-X, ressonância, tomografia, ultrassom etc.)
e receba uma análise estruturada com referências científicas.

⚠️ **Atenção:** ferramenta apenas para fins educacionais. Não substitui a avaliação de um profissional de saúde.
""")

with st.sidebar:
    st.header("⚙️ Configurações")
    provedor = st.selectbox("Provedor de IA", list(PROVEDORES))
    modelo = st.selectbox("Modelo", PROVEDORES[provedor]["modelos"])
    modelo_manual = st.text_input("Outro modelo (opcional)", placeholder="ex.: id de um modelo mais novo")
    if modelo_manual.strip():
        modelo = modelo_manual.strip()

    api_key = st.text_input(
        f"Sua chave de API ({provedor.split(' ')[0]})",
        type="password",
        key=f"chave_{provedor}",
        help="A chave fica só nesta sessão do navegador e não é salva no servidor.",
    )
    st.caption(f"Não tem chave? [Crie aqui]({PROVEDORES[provedor]['onde_obter']}).")

    if not TAVILY_API_KEY:
        st.error("TAVILY_API_KEY não encontrada no .env do servidor.")

arquivo = st.file_uploader("Envie uma imagem médica (JPG, PNG)", type=["jpg", "jpeg", "png"])

if arquivo is None:
    st.info("⬆️ Envie uma imagem médica para começar.")
    st.stop()

st.image(arquivo, caption="Imagem enviada", use_container_width=True)

pronto = bool(api_key) and bool(TAVILY_API_KEY)
if not api_key:
    st.warning("🔑 Informe sua chave de API na barra lateral para liberar a análise.")

if st.button("🔍 Analisar imagem", disabled=not pronto, type="primary"):
    barra = st.progress(0, text="Iniciando...")
    status = st.status("Os agentes estão trabalhando...", expanded=True)
    etapas = {chave: (texto, pct) for chave, texto, pct in ETAPAS}

    def avisar(etapa):
        texto, pct = etapas[etapa]
        barra.progress(pct, text=texto)
        status.write(texto)

    try:
        med, rio = criar_agentes(provedor, modelo, api_key)
        resultado = process_img_pipeline(arquivo, med, rio, avisar)
    except Exception as erro:
        barra.empty()
        status.update(label="A análise foi interrompida", state="error")
        st.error(mensagem_de_erro(erro))
    else:
        barra.progress(100, text="✅ Concluído")
        status.update(label="Análise concluída", state="complete", expanded=False)
        st.markdown(resultado)