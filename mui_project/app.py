import streamlit as st
import numpy as np
import cv2
from PIL import Image

# 1. Importar Módulos (Actualizado: Clasificación eliminada, solo Segmentación y DNN)
from models import (
    segformer_processor, 
    segformer_model, 
    segformer_color_map, 
    face_detector_net 
)
from processing_functions import (
    segment_image, 
    detect_faces_dnn, 
    detect_orb_keypoints, 
    apply_clahe,  
    apply_adaptive_threshold, 
    apply_morphological_op
)

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Monitor Urbano Inteligente (MUI)", layout="wide")

def main():
    st.title("🏙️ Monitor Urbano Inteligente (MUI)")
    
    uploaded_file = st.file_uploader("Sube una imagen de escena urbana", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is None:
        st.info("Sube una imagen para comenzar el análisis.")
        return

    # Convertir la imagen cargada a PIL (RGB) y OpenCV (BGR)
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_cv_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR) 
    
    # --- BARRA LATERAL (Fase 3: Preprocesamiento y Refinamiento) ---
    st.sidebar.header("⚙️ Módulo de Preprocesamiento")
    processed_image_cv = image_cv_bgr.copy()

    # NUEVA HERRAMIENTA: CLAHE (Para mejorar DNN)
    st.sidebar.subheader("Mejora de Contraste (CLAHE)")
    apply_clahe_option = st.sidebar.checkbox("Aplicar Mejora de Contraste CLAHE")
    if apply_clahe_option:
        # Parámetros controlados por el usuario
        clip_limit = st.sidebar.slider("Límite de Recorte (Clip Limit)", 1.0, 10.0, 2.0, step=0.5)
        tile_grid_size = st.sidebar.slider("Tamaño de Cuadrícula (Tile Size)", 2, 16, 8, step=2)
        
        processed_image_cv = apply_clahe(processed_image_cv, clip_limit, tile_grid_size)
        st.sidebar.success(f"Contraste CLAHE aplicado.")


    # 1. Umbralización Adaptativa (Mantener para ORB)
    st.sidebar.subheader("Umbralización Adaptativa")
    threshold_option = st.sidebar.selectbox("Método de Umbralización", ["Ninguno", "Media (Mean)", "Gaussiana (Gaussian)"])
    if threshold_option != "Ninguno":
        block_size = st.sidebar.slider("Tamaño del Bloque (impar)", 3, 51, 11, step=2)
        C_value = st.sidebar.slider("Constante C", -10, 10, 2)
        processed_image_cv = apply_adaptive_threshold(processed_image_cv, threshold_option, block_size, C_value)
        st.sidebar.success(f"Umbralización {threshold_option} aplicada.")

    # 2. Operaciones Morfológicas (Mantener para ORB)
    st.sidebar.subheader("Operaciones Morfológicas")
    morph_option = st.sidebar.selectbox("Operación Morfológica", ["Ninguno", "Apertura (Limpiar ruido)", "Cierre (Rellenar huecos)"])
    if morph_option != "Ninguno":
        kernel_size = st.sidebar.slider("Tamaño del Elemento Estructurante (EE)", 1, 15, 3)
        processed_image_cv = apply_morphological_op(processed_image_cv, morph_option, kernel_size)
        st.sidebar.success(f"{morph_option} aplicada.")

    # Convertir la imagen final preprocesada de vuelta a PIL (RGB)
    processed_image_rgb = cv2.cvtColor(processed_image_cv, cv2.COLOR_BGR2RGB)
    final_image_pil = Image.fromarray(processed_image_rgb)
    
    # Mostrar la imagen de partida
    st.subheader("🖼️ Imagen de Partida (Preprocesada)")
    st.image(final_image_pil, caption='Imagen de escena urbana lista para el análisis.', width='stretch')
    
    st.markdown("---")
    
    # --- PESTAÑAS DE ANÁLISIS (Fase 1 y 2) ---
    # FASE 1: Solo Segmentación
    tab1, tab2 = st.tabs(["Fase 1: Segmentación Categórica", "Fase 2: Detección e Inspección"])

    # Fase 1: Segmentación ahora es el ÚNICO foco relevante
    with tab1:
        st.header("Análisis de Contexto: Mapeo de Píxeles con SegFormer")
        
        if st.button("🗺️ Ejecutar Mapeo de Píxeles (Segmentación)", key="btn_segmentacion_unica"):
            with st.spinner('Segmentando con SegFormer (ADE20K)...'):
                segmented_img_pil, labels_detected = segment_image(
                    final_image_pil, 
                    segformer_processor, 
                    segformer_model, 
                    segformer_color_map
                )
                st.subheader("Mapa de Segmentación Categórica")
                st.image(segmented_img_pil, caption='Máscara de Segmentación', width='stretch')
                st.subheader("Leyenda de Clases")
                
                # Mostrar leyenda de clases detectadas
                for label, color in labels_detected.items():
                    color_hex = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'
                    st.markdown(
                        f"<span style='display:inline-block; width:12px; height:12px; background-color:{color_hex}; border: 1px solid #333; margin-right:5px;'></span> **{label.capitalize().replace('_', ' ')}**",
                        unsafe_allow_html=True
                    )
                st.success("Segmentación completada. Resultados precisos para contexto urbano.")

    # Fase 2: Detección DNN y ORB
    with tab2:
        st.header("Detección de Objetos y Análisis Estructural")
        detection_option = st.radio("Selecciona el tipo de análisis:", ["Detección de Transeúntes (DNN)", "Análisis Estructural (ORB)"])
        
        if st.button(f"Ejecutar {detection_option}", key="btn_phase2"):
            if detection_option == "Detección de Transeúntes (DNN)":
                with st.spinner('Detectando rostros con Modelo DNN...'):
                    # NOTA: Usamos processed_image_cv aquí, si el usuario activó CLAHE, 
                    # el detector se beneficia de ese preprocesamiento.
                    img_with_faces, count = detect_faces_dnn(processed_image_cv.copy(), face_detector_net) 
                    st.subheader(f"Resultado: Rostros Detectados ({count})")
                    result_img_pil = Image.fromarray(cv2.cvtColor(img_with_faces, cv2.COLOR_BGR2RGB))
                    st.image(result_img_pil, width='stretch')
            
            elif detection_option == "Análisis Estructural (ORB)":
                with st.spinner('Identificando keypoints estructurales con ORB...'):
                    img_with_keypoints, count = detect_orb_keypoints(processed_image_cv.copy())
                    st.subheader(f"Resultado: Keypoints Estructurales ({count})")
                    result_img_pil = Image.fromarray(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
                    st.image(result_img_pil, width='stretch')

# --- EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    main()