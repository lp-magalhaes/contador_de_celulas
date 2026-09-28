import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="Analisador de Células Automático", layout="wide", page_icon="🔬")

st.title("🔬 Analisador de Células 100% Inteligente e Autônomo")
st.write("Processamento totalmente automatizado: segmentação, separação de células coladas e filtros de ruído adaptativos.")

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
            
        img_marcada = img.copy()
        
        # --- ESPAÇO DE CORES HSL (Para Luminosidade) e HSV (Para Cores) ---
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Extração do canal L (Luminosidade do HSL)
        canal_l = hls[:, :, 1]
        luminosidade_media_hsl = round(float(cv2.mean(canal_l)), 2)

        # --- CÁLCULO DO FATOR DE CONVERSÃO PARA MICRÔMETROS ---
        altura_px, largura_px = img.shape[:2]
        area_total_px = altura_px * largura_px
        area_total_um2 = 320 * 320  # 102400 um²
        fator_conversao_area = area_total_um2 / area_total_px

        # --- SEPARAÇÃO AUTOMÁTICA DE CÉLULAS COLADAS (WATERSHED) ---
        suavizada = cv2.GaussianBlur(canal_l, (5, 5), 0)
        
        # Limiarização Automática por Otsu
        _, thresh = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        abertura = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        fundo_da_imagem = cv2.dilate(abertura, kernel, iterations=3)
        
        # Transformada de Distância para encontrar o centro de cada célula colada
        dist_transform = cv2.distanceTransform(abertura, cv2.DIST_L2, 5)
        _, primeiro_plano_certo = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
        
        primeiro_plano_certo = np.uint8(primeiro_plano_certo)
        regiao_desconhecida = cv2.subtract(fundo_da_imagem, primeiro_plano_certo)
        
        _, marcadores = cv2.connectedComponents(primeiro_plano_certo)
        marcadores = marcadores + 1
        marcadores[regiao_desconhecida == 255] = 0
        
        marcadores = cv2.watershed(img, marcadores)
        
        mascara_objetos_dinamica = np.zeros_like(thresh)
        mascara_objetos_dinamica[marcadores > 1] = 255

        # --- DEFINIÇÃO DOS LIMITES DE CORES ORIGINAIS (HSV) ---
        lower_red1, upper_red1 = np.array([0, 40, 40]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([160, 40, 40]), np.array([179, 255, 255])
        lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])
        lower_yellow, upper_yellow = np.array([11, 40, 40]), np.array([34, 255, 255])
        lower_blue, upper_blue = np.array([90, 40, 40]), np.array([130, 255, 255])

        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.addWeighted(mask_red1, 1.0, mask_red2, 1.0, 0.0)
        
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        mask_red = cv2.bitwise_and(mask_red, mascara_objetos_dinamica)
        mask_green = cv2.bitwise_and(mask_green, mascara_objetos_dinamica)
        mask_yellow = cv2.bitwise_and(mask_yellow, mascara_objetos_dinamica)
        mask_blue = cv2.bitwise_and(mask_blue, mascara_objetos_dinamica)

        contornos_vermelhos, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos_verdes, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos_amarelos, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos_azuis, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- CÁLCULO DO FILTRO DE TAMANHO ADAPTATIVO (AUTOMÁTICO) ---
        todos_contornos = list(contornos_vermelhos) + list(contornos_verdes) + list(contornos_amarelos) + list(contornos_azuis)
        areas_brutas = [cv2.contourArea(c) for c in todos_contornos if cv2.contourArea(c) > 0]
        
        if areas_brutas:
            area_minima_adaptativa = np.median(areas_brutas) * 0.15
            area_minima_involucro = max(20.0, area_minima_adaptativa)
        else:
            area_minima_involucro = 150.0

        areas_celulas_um2 = []

        def processar_contornos_dinamicos(contornos, cor_bgr):
            qtd = 0
            for c in contornos:
                area_px = cv2.contourArea(c)
                if area_px >= area_minima_involucro:
                    qtd += 1
                    area_um2 = area_px * fator_conversao_area
                    areas_celulas_um2.append(area_um2)
                    
                    (x, y), raio = cv2.minEnclosingCircle(c)
                    cv2.circle(img_marcada, (int(x), int(y)), int(raio), cor_bgr, 2)
            return qtd

        qtd_vermelhas = processar_contornos_dinamicos(contornos_vermelhos, (0, 0, 255))
        qtd_verdes = processar_contornos_dinamicos(contornos_verdes, (0, 255, 0))
        qtd_amarelas = processar_contornos_dinamicos(contornos_amarelos, (0, 255, 255))
        qtd_azuis = processar_contornos_dinamicos(contornos_azuis, (255, 0, 0))
            
        total_celulas = qtd_vermelhas + qtd_verdes + qtd_amarelas + qtd_azuis

        # --- MÉTRICAS DE ÁREA EM MICRÔMETROS ---
        if areas_celulas_um2:
            area_media = round(float(np.mean(areas_celulas_um2)), 2)
            area_maxima = round(float(np.max(areas_celulas_um2)), 2)
            area_minima = round(float(np.min(areas_celulas_um2)), 2)
            
            area_total_celulas_um2 = sum(areas_celulas_um2)
            luminosidade_por_area = round(luminosidade_media_hsl / area_total_celulas_um2, 6)
        else:
            area_media, area_maxima, area_minima = 0.0, 0.0, 0.0
            luminosidade_por_area = 0.0

        dados_resumo.append({
            "Imagem": arquivo.name,
            "Invólucros Vermelhos": qtd_vermelhas,
            "Invólucros Verdes": qtd_verdes,
            "Invólucros Amarelos": qtd_amarelas,
            "Invólucros Azuis": qtd_azuis,
            "Total de Células": total_celulas,
            "Luminosidade Média HSL (L)": luminosidade_media_hsl,
            "Área Média (µm²)": area_media,
            "Área Máxima (µm²)": area_maxima,
            "Área Mínima (µm²)": area_minima,
            "Luminosidade por Área (L/µm²)": luminosidade_por_area
        })

        st.write(f"#### Células Identificadas: {arquivo.name}")
        img_rgb = cv2.cvtColor(img_marcada, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption=f"Análise automática de {arquivo.name}", use_container_width=True)

    if dados_resumo:
        df_resumo = pd.DataFrame(dados_resumo)
        
        st.success("Processamento concluído com sucesso!")
        st.write("### Resultados Analíticos por Imagem")
        st.dataframe(df_resumo, use_container_width=True)
        
        csv_resumo = df_resumo.to_csv(index=False, sep=";").encode('utf-8')
        st.download_button(
            label="📥 Baixar Relatório Avançado (CSV)",
            data=csv_resumo,
            file_name="analise_automatica_celulas.csv",
            mime="text/csv"
        )
    else:
        st.warning("Nenhuma célula foi detectada com os parâmetros atuais.")
