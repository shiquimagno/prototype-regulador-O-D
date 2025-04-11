import numpy as np
import csv
import time
import random
import os
import json
from datetime import datetime
import pickle

# Importar módulos necesarios
from simulador_mercado import generar_dataset_simulado, simular_escenario_precio
from optimizador_ganancias import OptimizadorGanancias
from importador_datos_mercado import importar_datos_mercado, obtener_factores_estacionales_reales

# Intentar importar matplotlib para visualización
try:
    import matplotlib.pyplot as plt
    matplotlib_disponible = True
except ImportError:
    matplotlib_disponible = False
    print("Nota: matplotlib no está instalado. La visualización no estará disponible.")
    print("Para instalar matplotlib, ejecuta: pip install matplotlib")

class EntrenadorMasivo:
    """Clase para entrenar masivamente modelos de predicción de demanda personalizados
    por comerciante y producto, utilizando técnicas avanzadas como validación cruzada,
    regularización adaptativa y búsqueda exhaustiva de hiperparámetros."""
    
    def __init__(self, archivo_datos='mondongos.csv', directorio_modelos='modelos_entrenados',
                 id_comerciante=None, tipo_producto=None):
        """Inicializa el entrenador masivo con los parámetros base.
        
        Args:
            archivo_datos: Ruta al archivo CSV con los datos reales de ventas
            directorio_modelos: Directorio donde se guardarán los modelos entrenados
            id_comerciante: Identificador único del comerciante (para personalización)
            tipo_producto: Tipo de producto que se está vendiendo (para personalización)
        """
        self.archivo_datos = archivo_datos
        self.directorio_modelos = directorio_modelos
        self.id_comerciante = id_comerciante if id_comerciante else 'general'
        self.tipo_producto = tipo_producto if tipo_producto else 'mondongo'
        self.datos_reales = self._leer_datos_csv()
        self.mejor_modelo = None
        self.historial_entrenamiento = []
        self.metricas_validacion = {}
        self.configuracion = {
            'iteraciones_totales': 0,
            'mejor_error': float('inf'),
            'fecha_ultimo_entrenamiento': None,
            'tiempo_total_entrenamiento': 0
        }
        
        # Crear directorio de modelos si no existe
        if not os.path.exists(self.directorio_modelos):
            os.makedirs(self.directorio_modelos)
        
        # Cargar configuración previa si existe
        self._cargar_configuracion()
    
    def _leer_datos_csv(self):
        """Lee los datos del archivo CSV y los devuelve como una lista de filas."""
        datos = []
        try:
            with open(self.archivo_datos, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Saltamos la cabecera
                for row in reader:
                    datos.append(row)
            return datos
        except FileNotFoundError:
            print(f"Advertencia: No se encontró el archivo {self.archivo_datos}")
            print("Se utilizarán solo datos simulados para el entrenamiento.")
            return []
    
    def _guardar_datos_csv(self, datos, archivo=None):
        """Guarda los datos en el archivo CSV."""
        if archivo is None:
            archivo = self.archivo_datos
            
        with open(archivo, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['precio', 'ofertaItems', 'itemsVendidos'])
            writer.writerows(datos)
    
    def _cargar_configuracion(self):
        """Carga la configuración previa del entrenamiento si existe."""
        archivo_config = os.path.join(self.directorio_modelos, f'config_{self.id_comerciante}_{self.tipo_producto}.json')
        if os.path.exists(archivo_config):
            try:
                with open(archivo_config, 'r') as f:
                    self.configuracion = json.load(f)
                print(f"Configuración cargada: {self.configuracion['iteraciones_totales']} iteraciones previas")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error al cargar configuración: {e}")
    
    def _guardar_configuracion(self):
        """Guarda la configuración actual del entrenamiento."""
        archivo_config = os.path.join(self.directorio_modelos, f'config_{self.id_comerciante}_{self.tipo_producto}.json')
        with open(archivo_config, 'w') as f:
            json.dump(self.configuracion, f)
    
    def _normalizar_datos(self, X, y):
        """Normaliza los datos para mejorar la convergencia del modelo."""
        X_mean = np.mean(X)
        X_std = np.std(X) if np.std(X) > 0 else 1
        X_norm = (X - X_mean) / X_std
        
        y_mean = np.mean(y)
        y_std = np.std(y) if np.std(y) > 0 else 1
        y_norm = (y - y_mean) / y_std if y_std > 0 else y
        
        return X_norm, y_norm, X_mean, X_std, y_mean, y_std
    
    def _dividir_datos(self, X, y, porcentaje_entrenamiento=0.8):
        """Divide los datos en conjuntos de entrenamiento y validación."""
        # Crear índices y mezclarlos
        indices = list(range(len(X)))
        random.shuffle(indices)
        
        # Calcular punto de división
        punto_division = int(len(indices) * porcentaje_entrenamiento)
        
        # Dividir índices
        indices_entrenamiento = indices[:punto_division]
        indices_validacion = indices[punto_division:]
        
        # Crear conjuntos
        X_entrenamiento = X[indices_entrenamiento]
        y_entrenamiento = y[indices_entrenamiento]
        X_validacion = X[indices_validacion]
        y_validacion = y[indices_validacion]
        
        return X_entrenamiento, y_entrenamiento, X_validacion, y_validacion
    
    def _calcular_caracteristicas(self, X):
        """Calcula características adicionales para el modelo de múltiples pesos.
        
        Args:
            X: Array de precios
            
        Returns:
            Matriz de características [X, X^2, log(X)]
        """
        # Crear matriz de características
        X_squared = X**2
        X_log = np.log1p(X)  # log(1+x) para evitar log(0)
        
        # Combinar en una matriz
        X_features = np.column_stack((X, X_squared, X_log))
        return X_features
    
    def _entrenar_modelo_avanzado(self, X, y, learning_rate=0.01, epochs=500, 
                                lambda_reg=0.01, early_stopping=True, 
                                paciencia=30, verbose=False):
        """Entrena un modelo avanzado con múltiples características y regularización adaptativa."""
        # Calcular características
        X_features = self._calcular_caracteristicas(X)
        
        # Inicializar pesos
        W = np.zeros(X_features.shape[1])
        error_history = []
        best_W = W.copy()
        best_error = float('inf')
        counter_paciencia = 0
        
        # Entrenamiento con descenso de gradiente
        for epoch in range(epochs):
            # Predicción con el modelo actual
            y_pred = np.dot(X_features, W)
            
            # Calcular error (error cuadrático medio) con término de regularización
            error = y_pred - y
            mse = np.mean(error**2) + lambda_reg * np.sum(W**2)  # Regularización L2
            error_history.append(mse)
            
            # Guardar el mejor modelo hasta ahora
            if mse < best_error:
                best_error = mse
                best_W = W.copy()
                counter_paciencia = 0
            else:
                counter_paciencia += 1
            
            # Early stopping si el error no mejora durante 'paciencia' épocas
            if early_stopping and counter_paciencia >= paciencia:
                if verbose:
                    print(f"Early stopping en época {epoch+1}")
                break
            
            # Calcular gradiente con regularización
            gradient = np.zeros_like(W)
            for j in range(len(W)):
                gradient[j] = np.mean(error * X_features[:, j]) + 2 * lambda_reg * W[j]
            
            # Actualizar pesos
            W = W - learning_rate * gradient
            
            # Mostrar progreso si verbose es True
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                print(f"Época {epoch+1}/{epochs}, Error: {mse:.6f}, Pesos: {W}")
        
        # Usar los mejores pesos encontrados
        if best_error < error_history[0]:
            W = best_W
            if verbose:
                print(f"Usando los mejores pesos encontrados con error {best_error:.6f}")
        
        return W, error_history
    
    def _validacion_cruzada(self, X, y, k_folds=5, learning_rate=0.01, epochs=500, lambda_reg=0.01, verbose=False):
        """Realiza validación cruzada de k-folds para evaluar el modelo."""
        # Crear índices para los folds
        indices = list(range(len(X)))
        random.shuffle(indices)
        tamaño_fold = len(indices) // k_folds
        
        errores_validacion = []
        pesos = []
        
        for i in range(k_folds):
            # Crear índices de validación y entrenamiento
            inicio_val = i * tamaño_fold
            fin_val = (i + 1) * tamaño_fold if i < k_folds - 1 else len(indices)
            indices_validacion = indices[inicio_val:fin_val]
            indices_entrenamiento = [idx for idx in indices if idx not in indices_validacion]
            
            # Crear conjuntos de entrenamiento y validación
            X_entrenamiento = X[indices_entrenamiento]
            y_entrenamiento = y[indices_entrenamiento]
            X_validacion = X[indices_validacion]
            y_validacion = y[indices_validacion]
            
            # Entrenar modelo
            W, _ = self._entrenar_modelo_avanzado(
                X_entrenamiento, y_entrenamiento, 
                learning_rate=learning_rate, 
                epochs=epochs, 
                lambda_reg=lambda_reg,
                verbose=False
            )
            
            # Evaluar en conjunto de validación
            X_val_features = self._calcular_caracteristicas(X_validacion)
            y_pred_val = np.dot(X_val_features, W)
            mse_val = np.mean((y_pred_val - y_validacion)**2)
            errores_validacion.append(mse_val)
            pesos.append(W)
            
            if verbose:
                print(f"Fold {i+1}/{k_folds}, Error de validación: {mse_val:.6f}")
        
        # Calcular error promedio y desviación estándar
        error_promedio = np.mean(errores_validacion)
        error_std = np.std(errores_validacion)
        
        # Seleccionar el mejor peso (el del fold con menor error)
        mejor_fold = np.argmin(errores_validacion)
        mejor_peso = pesos[mejor_fold]
        
        if verbose:
            print(f"Error promedio: {error_promedio:.6f} ± {error_std:.6f}")
            print(f"Mejor fold: {mejor_fold+1} con error {errores_validacion[mejor_fold]:.6f}")
        
        return mejor_peso, error_promedio, error_std
    
    def _busqueda_hiperparametros(self, X, y, verbose=False):
        """Realiza una búsqueda exhaustiva de hiperparámetros para encontrar los mejores valores."""
        # Definir grids de hiperparámetros
        learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
        lambdas_reg = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5]
        epochs_list = [300, 500, 1000]
        
        mejor_error = float('inf')
        mejores_params = {}
        resultados = []
        
        # Búsqueda en grid
        total_combinaciones = len(learning_rates) * len(lambdas_reg) * len(epochs_list)
        combinacion_actual = 0
        
        for lr in learning_rates:
            for lambda_reg in lambdas_reg:
                for epochs in epochs_list:
                    combinacion_actual += 1
                    if verbose:
                        print(f"Probando combinación {combinacion_actual}/{total_combinaciones}: ")
                        print(f"  learning_rate={lr}, lambda_reg={lambda_reg}, epochs={epochs}")
                    
                    # Usar validación cruzada para evaluar esta combinación
                    W, error_cv, error_std = self._validacion_cruzada(
                        X, y, k_folds=5, 
                        learning_rate=lr, 
                        epochs=epochs, 
                        lambda_reg=lambda_reg,
                        verbose=False
                    )
                    
                    # Guardar resultado
                    resultado = {
                        'learning_rate': lr,
                        'lambda_reg': lambda_reg,
                        'epochs': epochs,
                        'error_cv': error_cv,
                        'error_std': error_std,
                        'W': W.tolist()
                    }
                    resultados.append(resultado)
                    
                    # Actualizar mejor modelo
                    if error_cv < mejor_error:
                        mejor_error = error_cv
                        mejores_params = {
                            'learning_rate': lr,
                            'lambda_reg': lambda_reg,
                            'epochs': epochs,
                            'W': W
                        }
                        
                        if verbose:
                            print(f"  ¡Nuevo mejor modelo! Error CV: {error_cv:.6f} ± {error_std:.6f}")
        
        if verbose:
            print("\nMejores hiperparámetros encontrados:")
            print(f"  learning_rate: {mejores_params['learning_rate']}")
            print(f"  lambda_reg: {mejores_params['lambda_reg']}")
            print(f"  epochs: {mejores_params['epochs']}")
            print(f"  Error CV: {mejor_error:.6f}")
        
        return mejores_params, resultados
    
    def entrenar_masivamente(self, num_iteraciones=10000, datos_simulados_por_iteracion=100, 
                           usar_datos_mercado=True, verbose=True, guardar_cada=1000,
                           visualizar_progreso=True):
        """Entrena masivamente el modelo con miles de iteraciones para minimizar el error.
        
        Args:
            num_iteraciones: Número total de iteraciones de entrenamiento
            datos_simulados_por_iteracion: Número de datos simulados a generar en cada iteración
            usar_datos_mercado: Si es True, importa datos de mercados reales
            verbose: Si es True, muestra información detallada
            guardar_cada: Cada cuántas iteraciones guardar el modelo
            visualizar_progreso: Si es True, muestra gráficos de progreso
            
        Returns:
            Diccionario con el mejor modelo y métricas de entrenamiento
        """
        tiempo_inicio = time.time()
        errores_por_iteracion = []
        iteraciones_completadas = 0
        mejor_error_global = self.configuracion['mejor_error']
        
        # Cargar modelo previo si existe
        modelo_previo = self._cargar_modelo()
        if modelo_previo:
            self.mejor_modelo = modelo_previo
            if verbose:
                print(f"Modelo previo cargado con error: {mejor_error_global:.6f}")
                print(f"Continuando entrenamiento desde {self.configuracion['iteraciones_totales']} iteraciones previas")
        
        # Importar datos de mercado si se solicita
        datos_mercado = []
        if usar_datos_mercado:
            try:
                datos_mercado = importar_datos_mercado()
                if verbose:
                    print(f"Se importaron {len(datos_mercado)} registros de datos de mercados reales")
            except Exception as e:
                print(f"Error al importar datos de mercado: {e}")
        
        # Bucle principal de entrenamiento masivo
        for iteracion in range(num_iteraciones):
            # Generar datos simulados para esta iteración
            datos_simulados = generar_dataset_simulado(num_precios=datos_simulados_por_iteracion)
            
            # Combinar con datos reales y de mercado
            datos_combinados = datos_simulados.copy()
            
            # Dar más peso a los datos reales duplicándolos
            if self.datos_reales:
                for _ in range(5):  # Duplicar 5 veces para dar más peso
                    datos_combinados.extend(self.datos_reales)
            
            # Añadir datos de mercados reales si están disponibles
            if datos_mercado:
                for _ in range(3):  # Duplicar 3 veces para dar peso significativo
                    datos_combinados.extend(datos_mercado)
            
            # Preparar datos para entrenamiento
            X = np.array([float(row[0]) for row in datos_combinados])  # Precios
            y = np.array([float(row[2]) for row in datos_combinados])  # Ventas (demanda)
            
            # Normalizar datos
            X_norm, y_norm, X_mean, X_std, y_mean, y_std = self._normalizar_datos(X, y)
            
            # Cada 1000 iteraciones, realizar búsqueda de hiperparámetros
            if iteracion % 1000 == 0 and iteracion > 0:
                if verbose:
                    print(f"\nIteración {iteracion}: Realizando búsqueda de hiperparámetros...")
                mejores_params, _ = self._busqueda_hiperparametros(X_norm, y_norm, verbose=verbose)
                learning_rate = mejores_params['learning_rate']
                lambda_reg = mejores_params['lambda_reg']
                epochs = mejores_params['epochs']
            else:
                # Usar hiperparámetros predeterminados o los mejores encontrados previamente
                learning_rate = 0.01
                lambda_reg = 0.01
                epochs = 500
                if self.mejor_modelo and 'hiperparametros' in self.mejor_modelo:
                    learning_rate = self.mejor_modelo['hiperparametros'].get('learning_rate', learning_rate)
                    lambda_reg = self.mejor_modelo['hiperparametros'].get('lambda_reg', lambda_reg)
                    epochs = self.mejor_modelo['hiperparametros'].get('epochs', epochs)
            
            # Entrenar modelo con validación cruzada
            W, error_cv, _ = self._validacion_cruzada(
                X_norm, y_norm, k_folds=5,
                learning_rate=learning_rate,
                epochs=epochs,
                lambda_reg=lambda_reg,
                verbose=False
            )
            
            # Guardar error para seguimiento
            errores_por_iteracion.append(error_cv)
            
            # Actualizar mejor modelo global si este es mejor
            if error_cv < mejor_error_global:
                mejor_error_global = error_cv
                self.mejor_modelo = {
                    'W': W,
                    'X_mean': X_mean,
                    'X_std': X_std,
                    'y_mean': y_mean,
                    'y_std': y_std,
                    'error_final': error_cv,
                    'hiperparametros': {
                        'learning_rate': learning_rate,
                        'lambda_reg': lambda_reg,
                        'epochs': epochs
                    },
                    'iteracion': iteracion + self.configuracion['iteraciones_totales'],
                    'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Guardar inmediatamente si hay una mejora significativa
                if error_cv < mejor_error_global * 0.9:  # Mejora del 10% o más
                    self._guardar_modelo()
                    if verbose:
                        print(f"\n¡Mejora significativa! Modelo guardado con error: {error_cv:.6f}")
            
            # Mostrar progreso
            iteraciones_completadas = iteracion + 1
            if verbose and (iteracion % 100 == 0 or iteracion == num_iteraciones - 1):
                tiempo_actual = time.time()
                tiempo_transcurrido = tiempo_actual - tiempo_inicio
                iteraciones_por_segundo = (iteracion + 1) / tiempo_transcurrido if tiempo_transcurrido > 0 else 0
                tiempo_estimado = (num_iteraciones - iteracion - 1) / iteraciones_por_segundo if iteraciones_por_segundo > 0 else 0
                
                print(f"\rIteración {iteracion+1}/{num_iteraciones} | "  
                      f"Error: {error_cv:.6f} | Mejor: {mejor_error_global:.6f} | "  
                      f"Velocidad: {iteraciones_por_segundo:.1f} it/s | "  
                      f"Tiempo restante: {tiempo_estimado/60:.1f} min", end="")
            
            # Guardar modelo periódicamente
            if (iteracion + 1) % guardar_cada == 0:
                self._guardar_modelo()
                if verbose:
                    print(f"\nModelo guardado en iteración {iteracion+1}")
                
                # Visualizar progreso si se solicita
                if visualizar_progreso and matplotlib_disponible:
                    self._visualizar_progreso(errores_por_iteracion)
        
        # Actualizar configuración
        tiempo_total = time.time() - tiempo_inicio
        self.configuracion['iteraciones_totales'] += iteraciones_completadas
        self.configuracion['mejor_error'] = mejor_error_global
        self.configuracion['fecha_ultimo_entrenamiento'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.configuracion['tiempo_total_entrenamiento'] += tiempo_total
        self._guardar_configuracion()
        
        # Guardar modelo final
        self._guardar_modelo()
        
        if verbose:
            print(f"\n\nEntrenamiento masivo completado:")
            print(f"  Iteraciones totales: {self.configuracion['iteraciones_totales']}")
            print(f"  Mejor error: {mejor_error_global:.6f}")
            print(f"  Tiempo total: {tiempo_total/60:.2f} minutos")
            print(f"  Modelo guardado en: {self._obtener_ruta_modelo()}")
        
        # Visualizar resultados finales
        if visualizar_progreso and matplotlib_disponible:
            self._visualizar_progreso(errores_por_iteracion, titulo="Progreso Final de Entrenamiento")
        
        return {
            'mejor_modelo': self.mejor_modelo,
            'errores_por_iteracion': errores_por_iteracion,
            'tiempo_total': tiempo_total,
            'iteraciones_completadas': iteraciones_completadas
        }
    
    def _obtener_ruta_modelo(self):
        """Obtiene la ruta del archivo donde se guarda el modelo."""
        return os.path.join(self.directorio_modelos, f'modelo_{self.id_comerciante}_{self.tipo_producto}.pkl')
    
    def _guardar_modelo(self):
        """Guarda el mejor modelo en un archivo."""
        if self.mejor_modelo is None:
            return False
        
        ruta_modelo = self._obtener_ruta_modelo()
        with open(ruta_modelo, 'wb') as f:
            pickle.dump(self.mejor_modelo, f)
        
        # También guardar en CSV para compatibilidad
        ruta_csv = os.path.join(self.directorio_modelos, f'modelo_{self.id_comerciante}_{self.tipo_producto}.csv')
        with open(ruta_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['parametro', 'valor'])
            for key, value in self.mejor_modelo.items():
                if key == 'W':
                    for i, w in enumerate(value):
                        writer.writerow([f'W_{i}', w])
                elif key == 'hiperparametros':
                    for param, val in value.items():
                        writer.writerow([f'hiperparametros_{param}', val])
                elif key not in ['fecha', 'iteracion']:
                    writer.writerow([key, value])
        
        return True
    
    def _cargar_modelo(self):
        """Carga el mejor modelo desde un archivo."""
        ruta_modelo = self._obtener_ruta_modelo()
        if os.path.exists(ruta_modelo):
            try:
                with open(ruta_modelo, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error al cargar modelo: {e}")
                return None
        return None
    
    def _visualizar_progreso(self, errores, titulo="Progreso de Entrenamiento"):
        """Visualiza el progreso del entrenamiento mostrando la evolución del error."""
        if not matplotlib_disponible:
            return
        
        plt.figure(figsize=(12, 6))
        
        # Gráfico de error por iteración
        plt.subplot(1, 2, 1)
        plt.plot(errores)
        plt.title(f"{titulo} - Error por Iteración")
        plt.xlabel('Iteración')
        plt.ylabel('Error (MSE)')
        plt.grid(True)
        
        # Gráfico de error suavizado (media móvil)
        plt.subplot(1, 2, 2)
        window_size = min(50, len(errores))
        if window_size > 0:
            errores_suavizados = np.convolve(errores, np.ones(window_size)/window_size, mode='valid')
            plt.plot(errores_suavizados)
            plt.title(f"{titulo} - Error Suavizado")
            plt.xlabel('Iteración')
            plt.ylabel('Error (MSE) - Media Móvil')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predecir_demanda(self, precio):
        """Predice la demanda para un precio dado usando el modelo entrenado."""
        if self.mejor_modelo is None:
            print("Error: El modelo no ha sido entrenado. Ejecute entrenar_masivamente primero.")
            return None
        
        # Extraer parámetros del modelo
        W = self.mejor_modelo['W']
        X_mean = self.mejor_modelo['X_mean']
        X_std = self.mejor_modelo['X_std']
        y_mean = self.mejor_modelo['y_mean']
        y_std = self.mejor_modelo['y_std']
        
        # Normalizar el precio
        precio_norm = (precio - X_mean) / X_std
        
        # Calcular características
        precio_squared_norm = precio_norm**2
        precio_log_norm = np.log1p(precio_norm)
        X_features = np.array([precio_norm, precio_squared_norm, precio_log_norm])
        
        # Calcular predicción normalizada
        y_pred_norm = np.dot(W, X_features)
        
        # Desnormalizar para obtener la predicción en la escala original
        demanda_pred = (y_pred_norm * y_std) + y_mean if y_std > 0 else y_pred_norm
        
        # Asegurar un valor mínimo razonable (mínimo 0.1 kg = 100 gramos)
        demanda_pred = max(0.1, demanda_pred)
        
        return demanda_pred
    
    def predecir_demanda_y_ganancia(self, precio, costo=None):
        """Predice la demanda y ganancia para un precio dado usando el modelo entrenado."""
        demanda_pred = self.predecir_demanda(precio)
        if demanda_pred is None:
            return None, None
        
        # Si no se proporciona costo, estimarlo como un porcentaje del precio
        if costo is None:
            costo = precio * 0.6  # Estimación por defecto: 60% del precio
        
        # Calcular ganancia esperada
        ganancia_pred = (precio - costo) * demanda_pred
        
        return demanda_pred, ganancia_pred
    
    def recomendar_precio_optimo(self, rango_precios=None, costo=None):
        """Recomienda el precio que maximiza las ganancias."""
        if self.mejor_modelo is None:
            print("Error: El modelo no ha sido entrenado. Ejecute entrenar_masivamente primero.")
            return None
        
        # Si no se proporciona un rango, usar un rango predeterminado
        if rango_precios is None:
            if self.datos_reales:
                precios_unicos = set([float(row[0]) for row in self.datos_reales])
                rango_precios = sorted(list(precios_unicos))
                # Añadir más precios intermedios para una búsqueda más fina
                precios_adicionales = []
                for i in range(len(rango_precios) - 1):
                    precio_medio = (rango_precios[i] + rango_precios[i+1]) / 2
                    precios_adicionales.append(precio_medio)
                rango_precios.extend(precios_adicionales)
                rango_precios.sort()
            else:
                # Rango predeterminado si no hay datos reales
                rango_precios = np.linspace(5, 25, 41)  # Precios de 5 a 25 con incrementos de 0.5
        
        # Evaluar ganancias para cada precio
        resultados = []
        for precio in rango_precios:
            demanda_pred, ganancia_pred = self.predecir_demanda_y_ganancia(precio, costo)
            resultados.append((precio, demanda_pred, ganancia_pred))
        
        # Ordenar por ganancia (descendente)
        resultados.sort(key=lambda x: x[2], reverse=True)
        
        # Devolver el precio óptimo y sus métricas
        precio_optimo, demanda_optima, ganancia_optima = resultados[0]
        
        return {
            'precio_optimo': precio_optimo,
            'demanda_esperada': demanda_optima,
            'ganancia_esperada': ganancia_optima,
            'top_precios': resultados[:5]  # Top 5 precios con mejores ganancias
        }
    
    def visualizar_curva_demanda(self, rango_precios=None):
        """Visualiza la curva de demanda predicha por el modelo."""
        if not matplotlib_disponible:
            print("No se puede visualizar la curva de demanda porque matplotlib no está instalado.")
            return
        
        if self.mejor_modelo is None:
            print("Error: El modelo no ha sido entrenado. Ejecute entrenar_masivamente primero.")
            return
        
        # Si no se proporciona un rango, usar un rango predeterminado
        if rango_precios is None:
            rango_precios = np.linspace(5, 25, 100)  # 100 puntos entre 5 y 25
        
        # Predecir demanda para cada precio
        demandas = [self.predecir_demanda(precio) for precio in rango_precios]
        
        # Crear gráfico
        plt.figure(figsize=(12, 6))
        
        # Curva de demanda
        plt.subplot(1, 2, 1)
        plt.plot(rango_precios, demandas)
        plt.title("Curva de Demanda")
        plt.xlabel("Precio")
        plt.ylabel("Demanda (kg)")
        plt.grid(True)
        
        # Curva de ganancia
        plt.subplot(1, 2, 2)
        ganancias = [(precio - precio * 0.6) * demanda for precio, demanda in zip(rango_precios, demandas)]
        plt.plot(rango_precios, ganancias)
        plt.title("Curva de Ganancia")
        plt.xlabel("Precio")
        plt.ylabel("Ganancia")
        plt.grid(True)
        
        # Marcar precio óptimo
        indice_optimo = np.argmax(ganancias)
        precio_optimo = rango_precios[indice_optimo]
        ganancia_optima = ganancias[indice_optimo]
        plt.scatter([precio_optimo], [ganancia_optima], color='red', s=100, zorder=5)
        plt.annotate(f"Óptimo: ${precio_optimo:.2f}", 
                    (precio_optimo, ganancia_optima),
                    xytext=(10, -30), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        plt.tight_layout()
        plt.show()

# Función auxiliar para usar directamente
def entrenar_masivamente(num_iteraciones=10000, archivo_datos='mondongos.csv', 
                        id_comerciante=None, tipo_producto=None, verbose=True):
    """Función auxiliar para entrenar masivamente un modelo directamente.
    
    Args:
        num_iteraciones: Número total de iteraciones de entrenamiento
        archivo_datos: Ruta al archivo CSV con los datos reales de ventas
        id_comerciante: Identificador único del comerciante (para personalización)
        tipo_producto: Tipo de producto que se está vendiendo (para personalización)
        verbose: Si es True, muestra información detallada
    
    Returns:
        Entrenador con el modelo entrenado y recomendación de precio óptimo
    """
    entrenador = EntrenadorMasivo(
        archivo_datos=archivo_datos,
        id_comerciante=id_comerciante,
        tipo_producto=tipo_producto
    )
    
    # Entrenar masivamente
    resultado = entrenador.entrenar_masivamente(
        num_iteraciones=num_iteraciones,
        verbose=verbose,
        guardar_cada=1000,
        visualizar_progreso=matplotlib_disponible
    )
    
    # Obtener recomendación de precio óptimo
    recomendacion = entrenador.recomendar_precio_optimo()
    
    if verbose and recomendacion:
        print("\nRecomendación de precio óptimo:")
        print(f"Precio: {recomendacion['precio_optimo']:.2f}")
        print(f"Demanda esperada: {recomendacion['demanda_esperada']:.2f} kg")
        print(f"Ganancia esperada: {recomendacion['ganancia_esperada']:.2f}")
        print("\nTop 5 precios con mejores ganancias:")
        for i, (precio, demanda, ganancia) in enumerate(recomendacion['top_precios']):
            print(f"{i+1}. Precio: {precio:.2f}, Demanda: {demanda:.2f} kg, Ganancia: {ganancia:.2f}")
    
    # Visualizar curva de demanda
    if matplotlib_disponible:
        entrenador.visualizar_curva_demanda()
    
    return entrenador, recomendacion

# Si se ejecuta como script principal
if __name__ == "__main__":
    print("Entrenamiento Masivo de Modelos de Predicción de Demanda")
    print("=======================================================\n")
    
    # Solicitar parámetros al usuario
    try:
        num_iteraciones = int(input("Número de iteraciones (recomendado 10000+): ") or "10000")
        id_comerciante = input("ID del comerciante (opcional, Enter para 'general'): ") or None
        tipo_producto = input("Tipo de producto (opcional, Enter para 'mondongo'): ") or None
        
        print(f"\nIniciando entrenamiento masivo con {num_iteraciones} iteraciones...\n")
        entrenador, recomendacion = entrenar_masivamente(
            num_iteraciones=num_iteraciones,
            id_comerciante=id_comerciante,
            tipo_producto=tipo_producto,
            verbose=True
        )
        
        print("\nEntrenamiento completado con éxito!")
    except KeyboardInterrupt:
        print("\n\nEntrenamiento interrumpido por el usuario.")
    except Exception as e:
        print(f"\n\nError durante el entrenamiento: {e}")