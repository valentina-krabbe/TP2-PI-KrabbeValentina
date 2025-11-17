# 🏙️ Monitor Urbano Inteligente (MUI)

## 🎓 Datos del Proyecto

| Categoría | Información |
| :--- | :--- |
| **Alumna** | Valentina Micaela Zoe Krabbe |
| **DNI** | 45207992 |
| **Materia** | Procesamiento de Imágenes |
| **Profesor** | Lucas Ariel De rito |
| **Institución** | Instituto Superior Santo Domingo |
El **Monitor Urbano Inteligente (MUI)** es una aplicación modular de Visión por Computadora (CV) y Deep Learning (DL), construida con Streamlit y OpenCV. Está diseñada para el análisis automatizado y la inspección de elementos clave en escenas urbanas: personas, infraestructura y contexto ambiental.

## 🚀 Contexto de Uso: Aplicaciones del MUI

El MUI se enfoca en tres fases de análisis, utilizando el preprocesamiento adecuado para cada tarea:


### Resumen de Roles y Ubicación de Archivos

| Archivo/Carpeta | Contenido | Rol en el Proyecto |
| :--- | :--- | :--- |
| **`app.py`** | Código Streamlit | Define la interfaz de usuario. |
| **`models.py`** | Código Python | Inicializa los modelos pesados. |
| **`deploy.prototxt` / `.caffemodel`** | Modelos DNN | Pesos y arquitectura de la red de detección de rostros. |
| **`images/`** | Imágenes (JPG, PNG) | Recursos para probar las funcionalidades. |
### 1. 🗺️ Fase 1: Segmentación Categórica (Análisis de Contexto)

**Objetivo:** Proporcionar un **mapa temático** de la escena, identificando la distribución y el tipo de entorno urbano.

| Elemento Analizado | Propósito | Tecnología Clave |
| :--- | :--- | :--- |
| **Calles y Entorno** | **Mapeo de Píxeles:** Identifica y marca la ubicación de la carretera (`road`), el cielo (`sky`), edificios, vehículos y otros componentes. | **SegFormer (ADE20K):** Modelo de Segmentación Semántica altamente preciso para entornos complejos. |

---

### 2. 👥 Fase 2: Detección e Inspección (Análisis Estructural y Humano)

**Objetivo:** Localizar objetos específicos (personas) y analizar la integridad estructural de la infraestructura.

| Elemento Analizado | Propósito | Tecnología Clave |
| :--- | :--- | :--- |
| **Personas** | **Detección de Rostros:** Ubica y cuenta personas. | **OpenCV DNN (SSD):** Red Neuronal Profunda optimizada para detección rápida. |
| **Edificios / Infraestructura**| **Análisis Estructural (ORB):** Identifica puntos clave (esquinas, intersecciones) para crear una "huella dactilar" estructural. | **ORB:** Algoritmo rápido usado para **monitorear deterioro** y el **registro** de imágenes de infraestructura. |

---

### 3. ⚙️ Fase 3: Preprocesamiento (Optimización Condicional)

Este módulo permite ajustar la imagen de entrada para **optimizar** el rendimiento del modelo objetivo.

| Operación | Aplicación Principal | Propósito |
| :--- | :--- | :--- |
| **Mejora de Contraste (CLAHE)**| **Detección DNN (Rostros)** | Aumenta el contraste y la visibilidad de los rasgos faciales en áreas sombreadas sin distorsionar el color original. |
| **Umbralización / Morfología** | **Análisis ORB (Estructuras)** | Acentúa bordes y simplifica la imagen a figuras binarias, facilitando el reconocimiento de esquinas y patrones geométricos por ORB. |

---

## ⚠️ Recomendaciones Operacionales

La efectividad del MUI se maximiza al aplicar la técnica de preprocesamiento adecuada a la tarea que se va a ejecutar.

### Detección de Rostros (DNN)

| Acción | Razón |
| :--- | :--- |
| **Aplicar CLAHE** | **RECOMENDADO:** CLAHE mejora el contraste local del rostro, lo que es invaluable para el DNN en **condiciones de baja iluminación o contraluz**, sin alterar los gradientes de color. |
| **Evitar Umbralización** | **NO RECOMENDADO:** El DNN fue entrenado en imágenes fotográficas. La umbralización (Gaussiana o Media) **destruye los gradientes de luz y sombra** necesarios para reconocer los rasgos faciales, llevando a fallos en la detección. |

### Análisis Estructural (ORB)

| Acción | Razón |
| :--- | :--- |
| **Aplicar Umbralización** | **RECOMENDADO:** Los algoritmos como ORB requieren bordes duros. La umbralización (especialmente la Gaussiana) simplifica la imagen a formas puras, **maximizando la detección de esquinas**. |

---

## 🛠️ Instalación y Ejecución

Para desplegar y usar el MUI, sigue estos pasos:

1.  **Entorno:** Activa el entorno Conda donde instalaste todas las dependencias (ej., `p_imagenes`).
    ```bash
    conda activate p_imagenes
    ```
2.  **Archivos del Modelo:** Asegúrate de que los archivos de la red DNN (`deploy.prototxt` y `res10_300x300_ssd_iter_140000.caffemodel`) estén en la carpeta raíz del proyecto (`mui_project`).
3.  **Ejecución:** Lanza la aplicación desde la carpeta raíz.
    ```bash
    streamlit run app.py

    ```


