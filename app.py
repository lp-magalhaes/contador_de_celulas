import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="Analisador de Células", layout="wide", page_icon="🔬")

st.title("🔬 Analisador de Células Multicores")
st.write("Faça o upload de imagens para contar invólucros celulares (Vermelhos, Verdes, Azuis e Amarelos) e medir a luminosidade.")

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
        file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error(f"Erro ao ler o arquivo: {arquivo.name}")
            continue
            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- CÁLCULO DO FATOR DE CONVERSÃO PARA MICRÔMETROS ---
        # Dimensões reais conhecidas: 320 x 320 micrometros
        altura_px, largura_px = img.shape[:2]
        area_total_px = altura_px * largura_px
        area_total_um2 = 320 * 320  # 102400 um^2
        
        # Fator que converte 1 pixel quadrado para micrometro quadrado
        fator_conversao_area = area_total_um2 / area_total_px

        # 1. CÁLCULO DA LUMINOSIDADE MÉDIA DA IMAGEM INTEIRA
        luminosidade_media_imagem = round(float(cv2.mean(gray)[0]), 2)

        # --- DEFINIÇÃO DOS LIMITES DE CORES (HSV) ---
        # Vermelho (Faixa 1 e 2)
        lower_red1, upper_red1 = np.array([0, 40, 40]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([160, 40, 40]), np.array([179, 255, 255])
        
        # Verde
        lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])
        
        # Amarelo (Fica entre o Vermelho e o Verde)
        lower_yellow, upper_yellow = np.array([11, 40, 40]), np.array([34, 255, 255])
        
        # Azul (Fica após o Verde)
        lower_blue, upper_blue = np.array([90, 40, 40]), np.array([130, 255, 255])

        # Criar Máscaras de Cor
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.addWeighted(mask_red1, 1.0, mask_red2, 1.0, 0.0)
        
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # --- OPERAÇÕES MORFOLÓGICAS ---
        # Fecha as bordas pontilhadas e elimina pequenos pontos isolados
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

        # Encontrar contornos externos
        contornos_vermelhos, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos_verdes, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos_amarelos, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos_azuis, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ÁREA MÍNIMA DO INVÓLUCRO (Filtro de ruído)
        area_minima_involucro = 150 

        # Lista para guardar o tamanho de todas as células detectadas nesta imagem
        areas_celulas_um2 = []

        # Processar Células Vermelhas
        qtd_vermelhas = 0
        for c in contornos_vermelhos:
            area_px = cv2.contourArea(c)
            if area_px >= area_minima_involucro: 
                qtd_vermelhas += 1
                areas_celulas_um2.append(area_px * fator_conversao_area)

        # Processar Células Verdes
        qtd_verdes = 0
        for c in contornos_verdes:
            area_px = cv2.contourArea(c)
            if area_px >= area_minima_involucro: 
                qtd_verdes += 1
                areas_celulas_um2.append(area_px * fator_conversao_area)

        # Processar Células Amarelas
        qtd_amarelas = 0
        for c in contornos_amarelos:
            area_px = cv2.contourArea(c)
            if area_px >= area_minima_involucro: 
                qtd_amarelas += 1
                areas_celulas_um2.append(area_px * fator_conversao_area)

        # Processar Células Azuis
        qtd_azuis = 0
        for c in contornos_azuis:
            area_px = cv2.contourArea(c)
            if area_px >= area_minima_involucro: 
                qtd_azuis += 1
                areas_celulas_um2.append(area_px * fator_conversao_area)
            
        # Calcular o total geral desta imagem
        total_celulas = qtd_vermelhas + qtd_verdes + qtd_amarelas + qtd_azuis

        # --- CÁLCULO DAS MÉTRICAS DE ÁREA ---
        if areas_celulas_um2:
            area_media = round(float(np.mean(areas_celulas_um2)), 2)
            area_maxima = round(float(np.max(areas_celulas_um2)), 2)
            area_minima = round(float(np.min(areas_celulas_um2)), 2)
            
            # Cálculo da soma total das áreas para normalizar a luminosidade
            area_total_celulas_um2 = sum(areas_celulas_um2)
            # Luminosidade dividida pela área total ocupada pelas células
            luminosidade_por_area = round(luminosidade_media_imagem / area_total_celulas_um2, 6)
        else:
            area_media, area_maxima, area_minima = 0.0, 0.0, 0.0
            luminosidade_por_area = 0.0

        # Adicionar os dados consolidados da imagem
        dados_resumo.append({
            "Imagem": arquivo.name,
            "Invólucros Vermelhos": qtd_vermelhas,
            "Invólucros Verdes": qtd_verdes,
            "Invólucros Amarelos": qtd_amarelas,
            "Invólucros Azuis": qtd_azuis,
            "Total de Células": total_celulas,
            "Luminosidade Média da Imagem": luminosidade_media_imagem,
            "Área Média (µm²)": area_media,
            "Área Máxima (µm²)": area_maxima,
            "Área Mínima (µm²)": area_minima,
            "Luminosidade por Área (L/µm²)": luminosidade_por_area
        })

    # Mostrar resultados e gerar o botão de download do CSV
    if dados_resumo:
        df_resumo = pd.DataFrame(dados_resumo)
        
        st.success("Processamento concluído com sucesso!")
        st.write("### Resultados Analíticos por Imagem")
        st.dataframe(df_resumo, use_container_width=True)
        
        csv_resumo = df_resumo.to_csv(index=False, sep=";").encode('utf-8')
        st.download_button(
            label="📥 Baixar Relatório Multicores (CSV)",
            data=csv_resumo,
            file_name="analise_multicores_celulas.csv",
            mime="text/csv"
        )
    else:
        st.warning("Nenhuma célula foi detectada com os parâmetros atuais.")
