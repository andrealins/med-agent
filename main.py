##Instalando as bibliotecas 
# pip install jupyterlab
# pip install pillow
# pip install -q agno
# pip install google-genai
# pip install tavily-python -q
# pip install -q streamlit python-dotenv
# npm install -q localtunnel

import os
from dotenv import load_dotenv
from pathlib import Path
from agno.agent import Agent
from agno.tools.tavily import TavilyTools
from agno.models.google import Gemini
from agno.media import Image as AgnoImage
from PIL import Image as PILImage
from textwrap import dedent
import streamlit as st

##Inserindo as chaves

# 1. Definindo o caminho base para o arquivo
dotenv_path = Path('.') / '.env'

load_dotenv()
gemini_api = os.getenv("GEMINI_API_KEY")
tavily_api = os.getenv("TAVILY_API_KEY")


# Configurando o Streamlit
st.set_page_config(page_title="Med Agent", layout="centered", page_icon="🩻")

st.title("🩺 Análise de Imagens Médicas.")
st.markdown("""
O Med é um especialista em diagnóstico por imagem e trabalha com o Rio, que é pesquisador médico!
Carregue uma **imagem médica** (Fotografia, raio-X, ressonância, tomografia, ultrassom, etc.).
O agente de IA analisará e trará insights, achados e hipóteses diagnósticas.

⚠️ **Atenção:** Esta ferramenta é apenas para fins educacionais e não substitui avaliação médica profissional.
""")

# Menu dos modelos de LLM
st.sidebar.header("⚙️ Configurações")
id_model = st.sidebar.selectbox(
    "Modelo Gemini:",
    ("gemini-2.0-flash", "gemini-2.5-flash-preview-05-20"),
)

#Pré-processamento da imagem

def preprocess_img(img_path):
    image = PILImage.open(img_path)
    width, height = image.size
    aspect_ratio = width / height
    new_width = 600
    new_height = int(new_width / aspect_ratio)
    resized = image.resize((new_width, new_height))
    temp_path = "temp_img.png"
    resized.save(temp_path)
    return temp_path, resized

# Formatando a resposta
def format_res(res, return_thinking=False):
    res = res.strip()
    if return_thinking:
        res = res.replace("<think>", "[pensando...] ")
        res = res.replace("</think>", "\n---\n")
    else:
        if "</think>" in res:
            res = res.split("</think>")[-1].strip()
    res = res.replace("```","")
    return res

# Prompts e agentes
prompt_analysis = """
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

### 4. Explicação em linguagem leiga
- Traduza os achados em linguagem simples para o paciente.
"""

med_agent = Agent(
    name="Medical Image Agent",
    role="Especialista em imagens médicas",
    model=Gemini(id=id_model),
    markdown=True
)

prompt_search_template = """Com base na seguinte análise de imagem médica, realize uma pesquisa complementar.
 - Utilize Tavily ou PubMed para encontrar artigos e protocolos atuais.
 - Forneça 2 a 3 links ou referências confiáveis.
 - Organize a resposta em markdown.

Resultado da análise médica: "{}"
"""

research_agent = Agent(
    name="Researcher Agent",
    role="Pesquisador médico",
    instructions=dedent("""
        Você é um pesquisador médico responsável por buscar informações complementares sobre os achados identificados.
        Forneça literatura recente e fontes confiáveis.
    """),
    model=Gemini(id=id_model),
    tools=[TavilyTools(api_key=tavily_api)],
)

# Função principal da pipeline
def process_img_pipeline(agno_img):
    res = med_agent.run(prompt_analysis, images=[agno_img])
    analysis = res.content

    prompt_search = prompt_search_template.format(analysis)
    res_search = research_agent.run(prompt_search)

    result = f"### 🩻 Resultado da Análise da Imagem\n{format_res(analysis)}\n\n"
    result += "---\n\n"
    result += f"### 📚 Pesquisa Complementar\n{format_res(res_search.content)}"

    return result

# Enviando imagem
uploaded_img = st.file_uploader("Envie uma imagem médica (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    st.image(uploaded_img, caption="Imagem enviada", use_container_width=True)

    if st.button("🔍 Analisar imagem"):
        with st.spinner("Analisando imagem..."):
            img_path = f"temp_{uploaded_img.name}"
            with open(img_path, "wb") as f:
                f.write(uploaded_img.getbuffer())

            temp_path, _ = preprocess_img(img_path)
            agno_img = AgnoImage(filepath=temp_path)

            result = process_img_pipeline(agno_img)
            st.markdown(result, unsafe_allow_html=True)

            os.remove(img_path)
else:
    st.info("⬆️ Envie uma imagem médica para começar.")