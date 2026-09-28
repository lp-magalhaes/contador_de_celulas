import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="Analisador de Células Avançado", layout="wide", page_icon="🔬")

st.title("🔬 Analisador de Células Dinâmico (HSL & Precisão Avançada)")
st.write("Upload de imagens para contagem e medição de células com segmentação adaptativa por cor e métricas em HSL.")

# Painel Lateral para ajuste fino da sensibilidade dinâmica
st.sidebar.header("🎛️ Ajuste de Sensibilidade")
fator_ajuste_dinamico = st.sidebar.slider(
    "Sensibilidade de Captura (Filtro Adaptativo)", 
    min_value=0.5, max_value=1.5, value=1.0, step=0.1,
    help="Valores menores tornam o filtro mais rígido (evita ruído). Valores maiores capturam células mais fracas."
)

area_minima_involucro = st.sidebar.number_input(
    "Área Mínima da Célula (px)", 
    min_value=10, max_value=1000, value=150, step=10,
    help="Elimina pequenos ruídos ou poeira que não sejam células."
)

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
        
        # --- MUDANÇA PARA ESPAÇO DE CORES HSL (HLS no OpenCV) ---
        # OpenCV usa a ordem H (Matiz), L (Luminosidade), S (Saturação)
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Mantido para máscaras auxiliares de matiz
        
        # Extração do canal L (Luminosidade do HSL)
        canal_l = hls[:, :, 1]
        luminosidade_media_hsl = round(float(cv2.mean(canal_l)[0]), 2)

        # --- CÁLCULO DO FATOR DE CONVERSÃO PARA MICRÔMETROS ---
        altura_px, largura_px = img.shape[:2]
        area_total_px = altura_px * largura_px
        area_total_um2 = 320 * 320  # 102400 um² [1]
        fator_conversao_area = area_total_um2 / area_total_px [1]

        # --- ABORDAGEM DINÂMICA DE IDENTIFICAÇÃO ---
        # 1. Filtro Bilateral para reduzir ruído preservando as bordas das células
        suavizada = cv2.bilateralFilter(canal_l, 9, 75, 75)
        
        # 2. Limiarização Adaptativa Dinâmica (Otsu) para mapear onde EXISTEM objetos contra o fundo
        _, mascara_objetos_dinamica = cv2.threshold(
            suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Aplicar peso de sensibilidade do usuário
        if fator_ajuste_dinamico != 1.0:
            kernel_ajuste = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            if fator_ajuste_dinamico > 1.0:
                mascara_objetos_dinamica = cv2.dilate(mascara_objetos_dinamica, kernel_ajuste, iterations=1)
            else:
                mascara_objetos_dinamica = cv2.erode(mascara_objetos_dinamica, kernel_ajuste, iterations=1)

        # Máscaras de cores base originais [1]
        lower_red1, upper_red1 = np.array([0, 40, 40]), np.array([10, 255, 255]) [1]
        lower_red2, upper_red2 = np.array([160, 40, 40]), np.array([179, 255, 255]) [1]
        lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255]) [1]
        lower_yellow, upper_yellow = np.array([11, 40, 40]), np.array([34, 255, 255]) [1]
        lower_blue, upper_blue = np.array([90, 40, 40]), np.array([130, 255, 255]) [1]

        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1) [1]
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2) [1]
        mask_red = cv2.addWeighted(mask_red1, 1.0, mask_red2, 1.0, 0.0) [1]
        
        mask_green = cv2.inRange(hsv, lower_green, upper_green) [1]
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow) [1]
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue) [1]

        # 3. INTERSECÇÃO DINÂMICA: Une o limite de cor à máscara de segmentação inteligente por Otsu
        mask_red = cv2.bitwise_and(mask_red, mascara_objetos_dinamica)
        mask_green = cv2.bitwise_and(mask_green, mascara_objetos_dinamica)
        mask_yellow = cv2.bitwise_and(mask_yellow, mascara_objetos_dinamica)
        mask_blue = cv2.bitwise_and(mask_blue, mascara_objetos_dinamica)

        # Operações morfológicas para limpar bordas pontilhadas [1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel) [1]
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel) [1]
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel) [1]
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel) [1]

        # Encontrar contornos [1]
        contornos_vermelhos, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) [1]
        contornos_verdes, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) [1]
        contornos_amarelos, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) [1]
        contornos_azuis, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) [1]

        areas_celulas_um2 = [] [1]

        # Função de processamento com desenho de círculos e armazenamento convertida para um2
        def processar_contornos_dinamicos(contornos, cor_bgr):
            qtd = 0
            for c in contornos:
                area_px = cv2.contourArea(c) [1]
                if area_px >= area_minima_involucro: [1]
                    qtd += 1
                    # Conversão em tempo real da área de pixel para micrômetro quadrado [1]
                    area_um2 = area_px * fator_conversao_area [1]
                    areas_celulas_um2.append(area_um2) [1]
                    
                    # Desenhar contorno circular exato ao redor do objeto
                    (x, y), raio = cv2.minEnclosingCircle(c)
                    cv2.circle(img_marcada, (int(x), int(y)), int(raio), cor_bgr, 2)
            return qtd

        qtd_vermelhas = processar_contornos_dinamicos(contornos_vermelhos, (0, 0, 255))
        qtd_verdes = processar_contornos_dinamicos(contornos_verdes, (0, 255, 0))
        qtd_amarelas = processar_contornos_dinamicos(contornos_amarelos, (0, 255, 255))
        qtd_azuis = processar_contornos_dinamicos(contornos_azuis, (255, 0, 0))
            
        total_celulas = qtd_vermelhas + qtd_verdes + qtd_amarelas + qtd_azuis [1]

        # --- MÉTRICAS DE ÁREA EM MICRÔMETROS (µm²) --- [1]
        if areas_celulas_um2: [1]
            area_media = round(float(np.mean(areas_celulas_um2)), 2) [1]
            area_maxima = round(float(np.max(areas_celulas_um2)), 2) [1]
            area_minima = round(float(np.min(areas_celulas_um2)), 2) [1]
            
            area_total_celulas_um2 = sum(areas_celulas_um2) [1]
            luminosidade_por_area = round(luminosidade_media_hsl / area_total_celulas_um2, 6)
        else:
            area_media, area_maxima, area_minima = 0.0, 0.0, 0.0 [1]
            luminosidade_por_area = 0.0 [1]

        dados_resumo.append({
            "Imagem": arquivo.name, [1]
            "Invólucros Vermelhos": qtd_vermelhas, [1]
            "Invólucros Verdes": qtd_verdes, [1]
            "Invólucros Amarelos": qtd_amarelas, [1]
            "Invólucros Azuis": qtd_azuis, [1]
            "Total de Células": total_celulas, [1]
            "Luminosidade Média HSL (L)": luminosidade_media_hsl,
            "Área Média (µm²)": area_media, [1]
            "Área Máxima (µm²)": area_maxima, [1]
            "Área Mínima (µm²)": area_minima, [1]
            "Luminosidade por Área (L/µm²)": luminosidade_por_area
        })

        # Renderização das imagens marcadas na tela
        st.write(f"#### Células Identificadas: {arquivo.name}")
        img_rgb = cv2.cvtColor(img_marcada, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption=f"Análise de {arquivo.name}", use_container_width=True)

    if dados_resumo: [1]
        df_resumo = pd.DataFrame(dados_resumo) [1]
        
        st.success("Processamento concluído com sucesso!") [1]
        st.write("### Resultados Analíticos por Imagem") [1]
        st.dataframe(df_resumo, use_container_width=True) [1]
        
        csv_resumo = df_resumo.to_csv(index=False, sep=";").encode('utf-8') [1]
        st.download_button( [1]
            label="📥 Baixar Relatório Avançado (CSV)", [1]
            data=csv_resumo, [1]
            file_name="analise_avancada_celulas.csv", [1]
            mime="text/csv" [1]
        )
    else:
        st.warning("Nenhuma célula foi detectada com os parâmetros atuais.") [1]
