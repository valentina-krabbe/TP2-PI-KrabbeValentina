import numpy as np
import cv2
from PIL import Image
import torch


# --- Fase 1: Segmentación ---
def segment_image(image_pil, processor, model, color_map):
    """Segmentación Semántica con SegFormer."""
    # Redimensionar si es muy grande para evitar errores de memoria
    if image_pil.width > 1024 or image_pil.height > 1024:
        image_pil = image_pil.resize((512, 512)) 
        
    inputs = processor(images=image_pil, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits 

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image_pil.size[::-1], 
        mode='bilinear',
        align_corners=False
    )
    
    pred_mask = torch.argmax(upsampled_logits.squeeze(), dim=0).cpu().numpy()
    
    height, width = pred_mask.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    idx2label = model.config.id2label
    random_colors = {}

    for idx in np.unique(pred_mask):
        label = idx2label.get(idx, f"Clase_{idx}")
        
        if label in color_map:
            color = color_map[label]
        elif label in random_colors:
            np.random.seed(idx * 13)
            color = tuple(np.random.randint(0, 256, 3).tolist())
            random_colors[label] = color
            
        color_image[pred_mask == idx] = color

    detected_labels = {}
    for idx in np.unique(pred_mask):
        label = idx2label.get(idx, f"Clase_{idx}")
        if (pred_mask == idx).any():
            color_rgb = tuple(color_image[pred_mask == idx][0])
            detected_labels[label] = (color_rgb[0]/255, color_rgb[1]/255, color_rgb[2]/255)
    
    original_img_np = np.array(image_pil)
    if original_img_np.shape[:2] != color_image.shape[:2]:
          color_image = cv2.resize(color_image, (original_img_np.shape[1], original_img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
          
    blended_image = cv2.addWeighted(original_img_np, 0.6, color_image, 0.4, 0)
    
    return Image.fromarray(blended_image), detected_labels


# --- Fase 2: Detección DNN ---
def detect_faces_dnn(image_cv_bgr, net):
    """Detección de Rostros con Red Neuronal Profunda (DNN)."""
    
    (h, w) = image_cv_bgr.shape[:2]
    
    #Crear un blob (preprocesamiento)
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image_cv_bgr, (300, 300)), 
        1.0, 
        (300, 300), 
        (104.0, 177.0, 123.0)
    )

    #Ejecutar la inferencia
    net.setInput(blob)
    detections = net.forward()
    
    img_with_faces = image_cv_bgr.copy()
    count = 0
    confidence_threshold = 0.5  #Umbral de confianza
    
    # 3. Iterar sobre las detecciones
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            count += 1
            
            # Obtener y reescalar las coordenadas de la caja delimitadora (bbox)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Dibujar el rectángulo
            cv2.rectangle(img_with_faces, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
            # Mostrar la confianza
            text = f"{confidence * 100:.2f}%"
            cv2.putText(img_with_faces, text, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    return img_with_faces, count

def detect_orb_keypoints(image_cv_bgr):
    """Análisis Estructural con ORB."""
    image_gris = cv2.cvtColor(image_cv_bgr, cv2.COLOR_BGR2GRAY)
    # nfeatures=1000 buen número para análisis urbano
    orb = cv2.ORB_create(nfeatures=1000, fastThreshold=20) 
    keypoints, _ = orb.detectAndCompute(image_gris, None)
    
    image_with_keypoints = cv2.drawKeypoints(
        image_cv_bgr, keypoints, None, color=(255, 0, 0), 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    return image_with_keypoints, len(keypoints)

# --- Fase 3: Preprocesamiento (Operaciones de CV) ---

def apply_clahe(image_cv_bgr, clip_limit, tile_grid_size):
    """Aplica Ecualización de Histograma Adaptativa (CLAHE) para mejorar el contraste local."""
    
    #Convertir a espacio de color LAB (L=Luminosidad)
    lab = cv2.cvtColor(image_cv_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    #Crear el objeto CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    
    #Aplicar CLAHE al canal de Luminosidad (L)
    cl = clahe.apply(l)
    
    #Combinar los canales de nuevo
    limg = cv2.merge((cl, a, b))
    
    #Convertir de nuevo a BGR
    final_image_cv = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final_image_cv

def apply_adaptive_threshold(image_cv_bgr, method_option, block_size, C_value):
    """Aplica Umbralización Adaptativa."""
    gray_image = cv2.cvtColor(image_cv_bgr, cv2.COLOR_BGR2GRAY)
    
    if method_option == "Media (Mean)":
        method = cv2.ADAPTIVE_THRESH_MEAN_C
    else: # Gaussiana
        method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        
    
    if block_size % 2 == 0:
        block_size += 1 

    processed_image_gray = cv2.adaptiveThreshold(
        gray_image, 255, method, cv2.THRESH_BINARY, block_size, C_value
    )
    return cv2.cvtColor(processed_image_gray, cv2.COLOR_GRAY2BGR)

def apply_morphological_op(image_cv_bgr, op_option, kernel_size):
    """Aplica Operaciones Morfológicas (Apertura/Cierre)."""
    morph_image_gray = cv2.cvtColor(image_cv_bgr, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if op_option == "Apertura (Limpiar ruido)":
        processed_image_gray = cv2.morphologyEx(morph_image_gray, cv2.MORPH_OPEN, kernel)
    else:
        processed_image_gray = cv2.morphologyEx(morph_image_gray, cv2.MORPH_CLOSE, kernel)
        
    return cv2.cvtColor(processed_image_gray, cv2.COLOR_GRAY2BGR)