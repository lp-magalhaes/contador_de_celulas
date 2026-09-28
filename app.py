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
            
        # Criar uma cópia para desenhar as marcações sem alterar a matriz original de análise
        img_marcada = img.copy()
            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. CÁLCULO DA LUMINOSIDADE MÉDIA DA IMAGEM INTEIRA
        luminosidade_media_imagem = round(float(cv2.mean(gray)[0]), 2)

        # --- DEFINIÇÃO DOS LIMITES DE CORES (HSV) ---
        lower_red1, upper_red1 = np.array([0, 40, 40]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([160, 40, 40]), np.array([179, 255, 255])
        lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])
        lower_yellow, upper_yellow = np.array([11, 40, 40]), np.array([34, 255, 255])
        lower_blue, upper_blue = np.array([90, 40, 40]), np.array([130, 255, 255])

        # Criar Máscaras de Cor
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.addWeighted(mask_red1, 1.0, mask_red2, 1.0, 0.0)
        
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # --- OPERAÇÕES MORFOLÓGICAS ---
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

        # Lista para guardar o tamanho de todas as células detectadas (Agora puramente em Pixels)
        areas_celulas_px = []

        # Função auxiliar para processar contornos e desenhar círculos
        def processar_contornos(contornos, cor_bgr):
            qtd = 0
            for c in contornos:
                area_px = cv2.contourArea(c)
                if area_px >= area_minima_involucro: 
                    qtd += 1
                    areas_celulas_px.append(area_px)
                    
                    # Calcular o círculo mínimo delimitador para a célula
                    (x, y), raio = cv2.minEnclosingCircle(c)
                    centro = (int(x), int(y))
                    raio = int(raio)
                    
                    # Desenhar o círculo na imagem (espessura 2)
                    cv2.circle(img_marcada, centro, raio, cor_bgr, 2)
            return qtd

        # Processar cada grupo de cor (Cores em BGR para o OpenCV)
        qtd_vermelhas = processar_contornos(contornos_vermelhos, (0, 0, 255))
        qtd_verdes = processar_contornos(contornos_verdes, (0, 255, 0))
        qtd_amarelos = processar_contornos(contornos_amarelos, (0, 255, 255))
        qtd_azuis = processar_contornos(contornos_azuis, (255, 0, 0))
            
        # Calcular o total geral desta imagem
        total_celulas = qtd_vermelhas + qtd_verdes + qtd_amarelos + qtd_azuis

        # --- CÁLCULO DAS MÉTRICAS DE ÁREA (EM PIXELS) ---
        if areas_celulas_px:
            area_media = round(float(np.mean(areas_celulas_px)), 2)
            area_maxima = round(float(np.max(areas_celulas_px)), 2)
            area_minima = round(float(np.min(areas_celulas_px)), 2)
            
            # Cálculo da soma total das áreas em pixel
            area_total_celulas_px = sum(areas_celulas_px)
            # Luminosidade dividida pela área total ocupada pelas células
            luminosidade_por_area = round(luminosidade_media_imagem / area_total_celulas_px, 6)
        else:
            area_media, area_maxima, area_minima = 0.0, 0.0, 0.0
            luminosidade_por_area = 0.0

        # Adicionar os dados consolidados da imagem
        dados_resumo.append({
            "Imagem": arquivo.name,
            "Invólucros Vermelhos": qtd_vermelhas,
            "Invólucros Verdes": qtd_verdes,
            "Invólucros Amarelos": qtd_amarelos,
            "Invólucros Azuis": qtd_azuis,
            "Total de Células": total_celulas,
            "Luminosidade Média da Imagem": luminosidade_media_imagem,
            "Área Média (px)": area_media,
            "Área Máxima (px)": area_maxima,
            "Área Mínima (px)": area_minima,
            "Luminosidade por Área (L/px)": luminosidade_por_area
        })

        # --- EXIBIÇÃO DA IMAGEM PROCESSADA ---
        st.write(f"#### Visualização: {arquivo.name}")
        # Converter BGR para RGB para o Streamlit exibir corretamente
        img_rgb = cv2.cvtColor(img_marcada, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption=f"Células identificadas na imagem {arquivo.name}", use_container_width=True)

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
            file_name="analise_multicores_celulas_pixels.csv",
            mime="text/csv"
        )
    else:
        st.warning("Nenhuma célula foi detectada com os parâmetros atuais.")
