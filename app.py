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
    dados_finais = []
    
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
        lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])

        # Criar Máscaras
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.addWeighted(mask_red1, 1.0, mask_red2, 1.0, 0.0)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Encontrar contornos
        contornos_vermelhos, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos_verdes, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Processar Células Vermelhas
        for i, c in enumerate(contornos_vermelhos):
            if cv2.contourArea(c) < 10: continue  # Ignora ruídos menores que 10 pixels
            mask_celula = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask_celula, [c], -1, 255, -1)
            luminosidade = cv2.mean(gray, mask=mask_celula)[0]
            
            dados_finais.append({
                "Imagem": arquivo.name,
                "ID_Celula": f"Vermelha_{i+1}",
                "Cor": "Vermelho",
                "Luminosidade_Media": round(luminosidade, 2)
            })

        # Processar Células Verdes
        for i, c in enumerate(contornos_verdes):
            if cv2.contourArea(c) < 10: continue
            mask_celula = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask_celula, [c], -1, 255, -1)
            luminosidade = cv2.mean(gray, mask=mask_celula)[0]
            
            dados_finais.append({
                "Imagem": arquivo.name,
                "ID_Celula": f"Verde_{i+1}",
                "Cor": "Verde",
                "Luminosidade_Media": round(luminosidade, 2)
            })

    # Mostrar resultados e gerar o botão de download do CSV
    if dados_finais:
        df = pd.DataFrame(dados_finais)
        df = df.sort_values(by=["Imagem", "Luminosidade_Media"], ascending=[True, False])
        
        st.success("Processamento concluído com sucesso!")
        
        # Exibir tabelas formatadas
        st.write("### Resultados Calculados (Ordenados por Brilho)")
        st.dataframe(df, use_container_width=True)
        
        # Botão de download do CSV
        csv = df.to_csv(index=False, sep=";").encode('utf-8')
        st.download_button(
            label="📥 Baixar Resultados em CSV",
            data=csv,
            file_name="resultados_celulas.csv",
            mime="text/csv"
        )
    else:
        st.warning("Nenhuma célula vermelha ou verde foi detectada com os parâmetros atuais.")
