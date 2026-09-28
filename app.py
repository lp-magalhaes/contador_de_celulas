import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="Analisador de Células Automático", layout="wide", page_icon="🔬")

st.title("🔬 Analisador de Células 100% Inteligente e Autônomo")
st.write("Processamento totalmente automatizado: segmentação adaptativa universal, separação por Watershed e classificação de cor posterior.")

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
        
        # --- ETAPA 1: EXTRAÇÃO DO CONTEXTO DE COR (HSL e HSV) ---
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        canal_l = hls[:, :, 1]  # Luminosidade (HSL)
        canal_s = hls[:, :, 2]  # Saturação (HSL) - Excelente para destacar objetos coloridos do fundo neutro
        
        # Correção definitiva do cálculo da tupla de média
        luminosidade_media_hsl = round(float(cv2.mean(canal_l)[0]), 2)

        # --- ETAPA 2: CÁLCULO DE CONVERSÃO PARA MICRÔMETROS ---
        altura_px, largura_px = img.shape[:2]
        area_total_px = altura_px * largura_px
        area_total_um2 = 320 * 320  # 102400 um²
        fator_conversao_area = area_total_um2 / area_total_px

        # --- ETAPA 3: SEGMENTAÇÃO ADAPTATIVA UNIVERSAL (Fundo vs Objeto) ---
        # Suavização para remover texturas e ruídos internos
        suavizada = cv2.GaussianBlur(canal_l, (5, 5), 0)
        
        # Limiarização por Otsu combinando o canal de Saturação e Luminosidade para máxima detecção
        _, thresh_l = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh_s = cv2.threshold(canal_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Mescla os dois limiares: Garante detecção se o objeto tiver contraste de brilho OU saturação de cor
        thresh = cv2.bitwise_or(thresh_l, thresh_s)
        
        # Operação morfológica para limpar pequenas poeiras isoladas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        abertura = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        fundo_da_imagem = cv2.dilate(abertura, kernel, iterations=3)
        
        # --- ETAPA 4: SEPARAÇÃO DE CÉLULAS COLADAS (WATERSHED) ---
        dist_transform = cv2.distanceTransform(abertura, cv2.DIST_L2, 5)
        # O corte adaptativo dinâmico de 0.35 identifica o cume de separação das esferas coladas
        _, primeiro_plano_certo = cv2.threshold(dist_transform, 0.35 * dist_transform.max(), 255, 0)
        
        primeiro_plano_certo = np.uint8(primeiro_plano_certo)
        regiao_desconhecida = cv2.subtract(fundo_da_imagem, primeiro_plano_certo)
        
        _, marcadores = cv2.connectedComponents(primeiro_plano_certo)
        marcadores = marcadores + 1
        marcadores[regiao_desconhecida == 255] = 0
        
        # Aplicação do algoritmo divisor de águas
        marcadores = cv2.watershed(img, marcadores)

        # --- ETAPA 5: CLASSIFICAÇÃO AUTOMÁTICA DE COR E MÉTRICAS ---
        qtd_vermelhas = 0
        qtd_verdes = 0
        qtd_amarelas = 0
        qtd_azuis = 0
        areas_celulas_um2 = []
        
        # Obtém os IDs únicos de cada célula segmentada pelo Watershed
        labels_unicos = np.unique(marcadores)
        
        # Coleta a mediana das áreas para o filtro dinâmico de ruído secundário
        areas_candidatas = []
        for label in labels_unicos:
            if label <= 1: continue  # Pula fundo da imagem e bordas divisórias (-1)
            areas_candidatas.append(np.sum(marcadores == label))
            
        area_minima_involucro = np.median(areas_candidatas) * 0.15 if areas_candidatas else 20.0
        area_minima_involucro = max(20.0, area_minima_involucro)

        # Varre cada célula individualmente para determinar sua cor predominante
        for label in labels_unicos:
            if label <= 1: 
                continue
                
            # Cria uma máscara isolada apenas para a célula atual
            mascara_celula = np.zeros_like(canal_l)
            mascara_celula[marcadores == label] = 255
            
            # Calcula a área em pixels desta célula específica
            area_px = np.sum(marcadores == label)
            if area_px < area_minima_involucro:
                continue
                
            # Adiciona ao cálculo de áreas em micrômetros
            area_um2 = area_px * fator_conversao_area
            areas_celulas_um2.append(area_um2)
            
            # Obtém o valor médio do Matiz (Hue) da região interna desta célula no espaço HSV
            matiz_medio = cv2.mean(hsv[:, :, 0], mask=mascara_celula)[0]
            
            # Determina a cor com base no valor nativo do Matiz (H) do OpenCV (Vai de 0 a 180)
            if (matiz_medio >= 0 and matiz_medio <= 10) or (matiz_medio >= 160 and matiz_medio <= 180):
                cor_bgr = (0, 0, 255)  # Vermelho
                qtd_vermelhas += 1
            elif matiz_medio > 10 and matiz_medio <= 34:
                cor_bgr = (0, 255, 255)  # Amarelo
                qtd_amarelas += 1
            elif matiz_medio > 34 and matiz_medio <= 85:
                cor_bgr = (0, 255, 0)  # Verde
                qtd_verdes += 1
            elif matiz_medio > 85 and matiz_medio <= 130:
                cor_bgr = (255, 0, 0)  # Azul
                qtd_azuis += 1
            else:
                cor_bgr = (255, 255, 255)  # Outra cor/Indefinida (Desenha círculo branco)
            
            # Encontra o contorno da célula isolada para desenhar o círculo delimitador perfeito
            contornos, _ = cv2.findContours(mascara_celula, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contornos:
                (x, y), raio = cv2.minEnclosingCircle(contornos[0])
                cv2.circle(img_marcada, (int(x), int(y)), int(raio), cor_bgr, 2)
            
        total_celulas = qtd_vermelhas + qtd_verdes + qtd_amarelas + qtd_azuis

        # --- ETAPA 6: COMPILAÇÃO DAS MÉTRICAS ANALÍTICAS ---
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
        st.image(img_rgb, caption=f"Análise automática universal de {arquivo.name}", use_container_width=True)

    if dados_resumo:
        df_resumo = pd.DataFrame(dados_resumo)
        
        st.success("Processamento concluído com sucesso!")
        st.write("### Resultados Analíticos por Imagem")
        st.dataframe(df_resumo, use_container_width=True)
        
        csv_resumo = df_resumo.to_csv(index=False, sep=";").encode('utf-8')
        st.download_button(
            label="📥 Baixar Relatório Avançado (CSV)",
            data=csv_resumo,
            file_name="analise_automatica_universal.csv",
            mime="text/csv"
        )
    else:
        st.warning("Nenhuma célula foi detectada com os parâmetros atuais.")
