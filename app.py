import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="Analisador de Células", layout="wide", page_icon="🔬")

st.title("🔬 Analisador de Células (Vermelhas e Verdes)")
st.write("Faça o upload de uma ou mais imagens para contar as células e medir a luminosidade.")

# Upload de múltiplos arquivos
arquivos_uploaddos = st.file_uploader(
    "Selecione as imagens das células", 
    type=["png", "jpg", "jpeg", "tif", "tiff"], 
    accept_multiple_files=True
)

if arquivos_uploaddos:
    dados_detalhados = []
    dados_resumo = []
    
    st.write(f"### Processando {len(arquivos_uploaddos)} imagem(ns)...")
    
    for arquivo in arquivos_uploaddos:
        # Converter o arquivo enviado para o formato do OpenCV
        file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error(f"Erro ao ler o arquivo: {arquivo.name}")
            continue
            
        # Converter para HSV (detecção de cor) e Tons de Cinza (luminosidade)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- DEFINIÇÃO DOS LIMITES DE CORES (VALORES PADRÃO) ---
        # Vermelho (Faixa 1 e 2 no HSV)
        lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([179, 255, 255])
        # Verde
        lower_green, upper_green = np.array([35, 100, 100]), np.array([85, 255, 255])

        # Criar Máscaras
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.addWeighted(mask_red1, 1.0, mask_red2, 1.0, 0.0)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Encontrar contornos
        contornos_vermelhos, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos_verdes, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        qtd_vermelhas = 0
        qtd_verdes = 0

        # Processar Células Vermelhas
        for c in contornos_vermelhos:
            if cv2.contourArea(c) < 10: continue  # Ignora ruídos menores que 10 pixels
            qtd_vermelhas += 1
            
            mask_celula = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask_celula, [c], -1, 255, -1)
            luminosidade = cv2.mean(gray, mask=mask_celula)[0]
            
            dados_detalhados.append({
                "Imagem": arquivo.name,
                "ID_Celula": f"Vermelha_{qtd_vermelhas}",
                "Cor": "Vermelho",
                "Luminosidade_Media": round(luminosidade, 2)
            })

        # Processar Células Verdes
        for c in contornos_verdes:
            if cv2.contourArea(c) < 10: continue
            qtd_verdes += 1
            
            mask_celula = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask_celula, [c], -1, 255, -1)
            luminosidade = cv2.mean(gray, mask=mask_celula)[0]
            
            dados_detalhados.append({
                "Imagem": arquivo.name,
                "ID_Celula": f"Verde_{qtd_verdes}",
                "Cor": "Verde",
                "Luminosidade_Media": round(luminosidade, 2)
            })
            
        # Adicionar o resumo de contagem desta imagem específica
        dados_resumo.append({
            "Imagem": arquivo.name,
            "Células Vermelhas": qtd_vermelhas,
            "Células Verdes": qtd_verdes,
            "Total de Células": qtd_vermelhas + qtd_verdes
        })

    # Mostrar resultados e gerar o botão de download do CSV
    if dados_detalhados:
        df_detalhado = pd.DataFrame(dados_detalhados)
        df_detalhado = df_detalhado.sort_values(by=["Imagem", "Luminosidade_Media"], ascending=[True, False])
        
        df_resumo = pd.DataFrame(dados_resumo)
        
        st.success("Processamento concluído com sucesso!")
        
        # Criar duas abas na tela para organizar as tabelas de forma limpa
        aba1, aba2 = st.tabs(["📊 Resumo de Contagem", "🔬 Dados Detalhados por Célula"])
        
        with aba1:
            st.write("### Quantidade de Células por Imagem")
            st.dataframe(df_resumo, use_container_width=True)
            
            # Botão de download do arquivo de Resumo
            csv_resumo = df_resumo.to_csv(index=False, sep=";").encode('utf-8')
            st.download_button(
                label="📥 Baixar Resumo de Contagem (CSV)",
                data=csv_resumo,
                file_name="resumo_contagem_celulas.csv",
                mime="text/csv"
            )
            
        with aba2:
            st.write("### Luminosidade e Cor de Cada Célula Individual (Ordenado por Brilho)")
            st.dataframe(df_detalhado, use_container_width=True)
            
            # Botão de download do arquivo Detalhado
            csv_detalhado = df_detalhado.to_csv(index=False, sep=";").encode('utf-8')
            st.download_button(
                label="📥 Baixar Dados Detalhados (CSV)",
                data=csv_detalhado,
                file_name="dados_detalhados_celulas.csv",
                mime="text/csv"
            )
    else:
        st.warning("Nenhuma célula vermelha ou verde foi detectada com os parâmetros atuais.")
