# Sistema de Entrenamiento Masivo de Modelos de Predicción de Demanda

Este sistema permite entrenar masivamente modelos de predicción de demanda personalizados por comerciante y tipo de producto, utilizando técnicas avanzadas como validación cruzada, regularización adaptativa y búsqueda exhaustiva de hiperparámetros.

## Características Principales

- **Entrenamiento masivo**: Entrena modelos con miles o cientos de miles de iteraciones para minimizar el error promedio.
- **Personalización**: Crea modelos específicos para cada comerciante y tipo de producto.
- **Visualización**: Muestra gráficamente la convergencia del error y las curvas de demanda/ganancia.
- **Comparación**: Compara diferentes modelos entrenados para analizar su rendimiento.
- **Guardado incremental**: Guarda los modelos periódicamente durante el entrenamiento.
- **Historial de errores**: Registra y analiza la evolución del error durante el entrenamiento.

## Archivos del Sistema

- `entrenar_masivamente.py`: Script principal con menú interactivo para todas las funcionalidades.
- `entrenar_modelo_personalizado.py`: Script para entrenamiento avanzado con opciones detalladas.
- `visualizador_convergencia.py`: Herramienta para visualizar y comparar modelos entrenados.
- `gestor_historial_entrenamiento.py`: Gestiona el historial de errores durante el entrenamiento.
- `entrenamiento_masivo.py`: Clase principal que implementa el algoritmo de entrenamiento masivo.

## Cómo Usar el Sistema

### Entrenamiento Básico

```bash
python entrenar_masivamente.py
```

Selecciona la opción 1 en el menú para entrenar un modelo con configuración básica.

### Entrenamiento Avanzado

```bash
python entrenar_modelo_personalizado.py --iteraciones 50000 --comerciante MiTienda --producto Carne --usar_mercado
```

### Visualizar Modelos Entrenados

```bash
python visualizador_convergencia.py --listar
python visualizador_convergencia.py --comerciante MiTienda --producto Carne
```

### Comparar Modelos

```bash
python visualizador_convergencia.py --comparar --modelos "MiTienda:Carne,MiTienda:Pescado"
```

### Analizar Historial de Entrenamiento

```bash
python gestor_historial_entrenamiento.py --listar
python gestor_historial_entrenamiento.py --analizar --comerciante MiTienda --producto Carne
```

## Recomendaciones para Entrenamiento Masivo

1. **Número de iteraciones**: Para modelos con error muy pequeño, se recomienda usar al menos 50,000 iteraciones.
2. **Datos simulados**: Aumentar el número de datos simulados por iteración (200-500) mejora la generalización.
3. **Datos de mercado**: Activar la opción `--usar_mercado` incorpora tendencias reales de mercado.
4. **Visualización**: Monitorear la convergencia del error ayuda a determinar cuándo detener el entrenamiento.
5. **Búsqueda de hiperparámetros**: El sistema realiza automáticamente búsquedas periódicas para optimizar el aprendizaje.

## Ejemplo de Uso Completo

```python
# Importar el entrenador masivo
from entrenamiento_masivo import EntrenadorMasivo

# Crear entrenador personalizado
entrenador = EntrenadorMasivo(
    id_comerciante="MiTienda",
    tipo_producto="Carne"
)

# Entrenar masivamente
resultado = entrenador.entrenar_masivamente(
    num_iteraciones=100000,
    datos_simulados_por_iteracion=300,
    usar_datos_mercado=True,
    verbose=True,
    guardar_cada=1000,
    visualizar_progreso=True
)

# Obtener recomendación de precio óptimo
recomendacion = entrenador.recomendar_precio_optimo()
print(f"Precio óptimo: ${recomendacion['precio_optimo']:.2f}")
print(f"Demanda esperada: {recomendacion['demanda_esperada']:.2f} kg")
print(f"Ganancia esperada: ${recomendacion['ganancia_esperada']:.2f}")

# Visualizar curva de demanda
entrenador.visualizar_curva_demanda()
```
