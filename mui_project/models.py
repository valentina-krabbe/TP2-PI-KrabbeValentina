"""
Este archivo maneja la carga de modelos grandes, utilizando @st.cache_resource para
garantizar que solo se carguen una vez, mejorando la velocidad de la aplicación Streamlit.
"""

import streamlit as st
import cv2
import os
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor



@st.cache_resource
def load_segmentation_model():
    """Carga SegFormer para Segmentación Semántica."""
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    
    # Mapeo de color base para visualización
    color_map = {
        'road': (128, 64, 128),     
        'sky': (70, 130, 180),      
        'building': (70, 70, 70),   
        'person': (220, 20, 60),    
        'car': (0, 0, 142)          
    }
    return processor, model, color_map

@st.cache_resource
def load_dnn_face_detector():
    """Carga el modelo de detección de rostros DNN de OpenCV."""
    
    # Rutas relativas a los archivos descargados, usando la ubicación de models.py
    script_dir = os.path.dirname(__file__)
    
    # Definir rutas completas a los archivos
    model_file = os.path.join(script_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
    config_file = os.path.join(script_dir, 'deploy.prototxt')
    
    # Verificar existencia antes de cargar
    if not os.path.exists(model_file) or not os.path.exists(config_file):
        raise FileNotFoundError(
            "ERROR: Faltan archivos del modelo DNN. Asegúrate de que 'deploy.prototxt' y "
            f"'res10_300x300_ssd_iter_140000.caffemodel' estén en la carpeta {script_dir}"
        )

    # Cargar la red neuronal
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    return net

# Exportar una instancia global de los modelos para que app.py los importe
segformer_processor, segformer_model, segformer_color_map = load_segmentation_model()
face_detector_net = load_dnn_face_detector() # Reemplaza face_cascade