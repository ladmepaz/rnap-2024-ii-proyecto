# **Proyecto: Modelado Predictivo con Redes Neuronales utilizando PyTorch**

Este repositorio contiene las instrucciones para que los estudiantes desarrollen un proyecto donde implementen un modelo predictivo basado en redes neuronales utilizando PyTorch y datos de su elección.

---

## **Instrucciones del Proyecto**

### **1. Preparación de datos**

1. **Selecciona un conjunto base:**
   - Descarga un conjunto de datos relevante para tu proyecto (por ejemplo, clasificación de imágenes de perros y gatos, análisis de texto como clasificación de sentimientos, o reconocimiento de patrones en audio, etc.).
   - Usa bibliotecas como `torchvision.datasets`, `pandas` o `datasets` según el tipo de datos elegido.

2. **Crea un conjunto de prueba propio:**
   - Si trabajas con clasificación de imágenes, adquiere tus propias imágenes alineadas con las categorías del conjunto base.
   - Si trabajas con texto, adquiere nuevos ejemplos alineados con las categorías del conjunto base para utilizarlos exclusivamente como conjunto de prueba.
   - Si trabajas con audio, graba o adquiere muestras propias que correspondan a las clases del conjunto base.
     
3. **Preprocesamiento de datos:**
   - Aplica transformaciones necesarias según el tipo de datos:
     - Normalización o estandarización.
     - Manejo de valores nulos o categóricos.
     - Aumentación de datos en el caso de imágenes (opcional).

4. **División de datos:**
   - Divide el conjunto base en entrenamiento y validación.
   - Asegúrate de mantener el conjunto propio como prueba independiente.

### **2. Diseño del modelo predictivo basado en redes neuronales**

1. **Selecciona un modelo adecuado:**
   - Debe ser una arquitectura basada en redes neuronales. Ejemplos:
     - Redes neuronales completamente conectadas.
     - Modelos con capas recurrentes o transformers si el tipo de dato lo justifica.

2. **Implementa la arquitectura:**
   - Utiliza PyTorch para definir las capas de la red neuronal y el forward pass.
   - Personaliza la arquitectura según el tipo de problema y datos.

3. **Visualiza la arquitectura:**
   - Usa herramientas como `torchinfo` o un resumen manual para detallar los componentes del modelo.

### **3. Entrenamiento del modelo**

1. Configura los hiperparámetros:
   - Tasa de aprendizaje (`learning rate`), tamaño del lote (`batch size`), número de épocas, entre otros.

2. Configura:
   - La función de pérdida (e.g., `torch.nn.CrossEntropyLoss` o `mean_squared_error` para regresión).
   - Un optimizador (e.g., `torch.optim.Adam` o `SGD`).

3. Implementa el ciclo de entrenamiento:
   - Realiza el forward pass.
   - Calcula la pérdida y ajusta los parámetros del modelo.
   - Monitorea la pérdida y otras métricas relevantes.

### **4. Evaluación del modelo**

1. Evalúa el rendimiento del modelo en:
   - El conjunto de validación.
   - El conjunto de prueba propio.

2. Calcula las siguientes métricas según el tipo de problema:
   - Clasificación: Precisión, Recall, F1-score.
   - Regresión: Error cuadrático medio (MSE), coeficiente de determinación (R²).

3. Visualiza los resultados:
   - Gráficas de desempeño (e.g., pérdida por época, curvas ROC, etc.).
   - Ejemplos de predicciones correctas e incorrectas (si aplica).

### **5. Sintonización de hiperparámetros**

1. Selecciona al menos tres hiperparámetros para analizar, por ejemplo:
   - Tasa de aprendizaje.
   - Tamaño del lote.
   - Profundidad del modelo o número de neuronas en una capa.

2. Realiza experimentos variando estos hiperparámetros.

3. Reporta los resultados:
   - Incluye una tabla o gráfica que muestre cómo los cambios impactaron las métricas.
   - Analiza los resultados y describe las configuraciones óptimas encontradas.

### **6. Reporte de resultados**

El cuaderno debe incluir:
1. Descripción del conjunto de datos base y el conjunto de prueba propio.
2. Explicación del modelo utilizado.
3. Gráficas de desempeño (e.g., pérdida y precisión).
4. Métricas finales en el conjunto de prueba propio.
5. Análisis de sintonización de hiperparámetros.
6. Discusión de limitaciones y posibles mejoras del modelo.

---

## **Entrega**

1. **Formato:**
   - Jupyter Notebook (.ipynb).

2. **Criterios:**
   - Comentarios claros y explicaciones en el código.
   - Visualizaciones de resultados.
   - Cuaderno bien documentado.

3. **Archivos a entregar:**
   - Código fuente.
   - Cuaderno con descripciones y resultados.
   - Datos del conjunto de prueba propio.
   - Modelo entrenado

---

## **Criterios de evaluación**

1. Correcta preparación de los datos (incluyendo el conjunto de prueba propio).
2. Implementación de una arquitectura basada en redes neuronales funcional.
3. Entrenamiento y evaluación adecuadas del modelo.
4. Cuaderno completo y bien documentado.
5. Análisis riguroso de los hiperparámetros seleccionados.
6. Calidad del código y las visualizaciones.
