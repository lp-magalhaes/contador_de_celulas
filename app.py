import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="Analisador de Células", layout="wide", page_icon="🔬")

st.title("🔬 Analisador de Células (Vermelhas e Verdes)")
st.write("Faça o upload de uma ou mais imagens para contar os invólucros celulares e medir a luminosidade média da imagem.")

# Upload de múltiplos arquivos
arquivos_uploaddos = st.file_uploader(
    "Selecione as imagens das células", 
    type=["png", "jpg", "jpeg", "tif", "tiff"], 
    accept_multiple_files=True
)

if arquivos_uploaddos:
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

        # 1. CÁLCULO DA LUMINOSIDADE MÉDIA DA IMAGEM INTEIRA
        luminosidade_media_imagem = round(float(cv2.mean(gray)[0]), 2)

        # --- DEFINIÇÃO DOS LIMITES DE CORES ---
        # Baixamos a saturação/brilho mínimo para 40 para pegar bordas apagadas
        lower_red1, upper_red1 = np.array([0, 40, 40]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([160, 40, 40]), np.array([179, 255, 255])
        lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])

        # Criar Máscaras
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.addWeighted(mask_red1, 1.0, mask_red2, 1.0, 0.0)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # --- OPERAÇÕES MORFOLÓGICAS ---
        # Kernel circular para fechar as bordas pontilhadas/apagadas do invólucro
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

        # Encontrar contornos externos (RETR_EXTERNAL ignora o que está dentro)
        contornos_vermelhos, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos_verdes, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        qtd_vermelhas = 0
        qtd_verdes = 0

        # ÁREA MÍNIMA DO INVÓLUCRO (Ajuste se células reais sumirem)
        area_minima_involucro = 150 

        # Processar Células Vermelhas
        for c in contornos_vermelhos:
            if cv2.contourArea(c) < area_minima_involucro: 
                continue  
            qtd_vermelhas += 1

        # Processar Células Verdes
        for c in contornos_verdes:
            if cv2.contourArea(c) < area_minima_involucro: 
                continue  
            # CORREÇÃO: Removido o incremento acidental de células vermelhas aqui
            qtd_verdes += 1
            
        # Adicionar os dados consolidados da imagem
        dados_resumo.append({
            "Imagem": arquivo.name,
            "Invólucros Vermelhos": qtd_vermelhas,
            "Invólucros Verdes": qtd_verdes,
            "Total de Células": qtd_vermelhas + qtd_verdes,
            "Luminosidade Média da Imagem": luminosidade_media_imagem
        })

    # Mostrar resultados e gerar o botão de download do CSV
    if dados_resumo:
        df_resumo = pd.DataFrame(dados_resumo)
        
        st.success("Processamento concluído com sucesso!")
        
        st.write("### Resultados Analíticos por Imagem")
        st.dataframe(df_resumo, use_container_width=True)
        
        # Botão de download do arquivo de Resumo
        csv_resumo = df_resumo.to_csv(index=False, sep=";").encode('utf-8')
        st.download_button(
            label="📥 Baixar Relatório de Células e Luminosidade (CSV)",
            data=csv_resumo,
            file_name="analise_celulas_luminosidade.csv",
            mime="text/csv"
        )
    else:
        st.warning("Nenhuma célula foi detectada com os parâmetros atuais.")
