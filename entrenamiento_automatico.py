import numpy as np
import csv
import time
import random
from datetime import datetime

# Importar el simulador de mercado
from simulador_mercado import generar_dataset_simulado, simular_escenario_precio

# Intentar importar matplotlib para visualización, pero hacer que sea opcional
try:
    import matplotlib.pyplot as plt
    matplotlib_disponible = True
except ImportError:
    matplotlib_disponible = False
    print("Nota: matplotlib no está instalado. La visualización no estará disponible.")
    print("Para instalar matplotlib, ejecuta: pip install matplotlib")

class EntrenadorAutomatico:
    """Clase para entrenar automáticamente el modelo de predicción de demanda
    utilizando técnicas avanzadas como validación cruzada y regularización."""
    
    def __init__(self, archivo_datos='mondongos.csv'):
        """Inicializa el entrenador con los parámetros base.
        
        Args:
            archivo_datos: Ruta al archivo CSV con los datos reales
        """
        self.archivo_datos = archivo_datos
        self.datos_reales = self._leer_datos_csv()
        self.mejor_modelo = None
        self.historial_entrenamiento = []
        self.metricas_validacion = {}
    
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
    
    def _entrenar_modelo_con_regularizacion(self, X, y, learning_rate=0.01, epochs=100, 
                                           lambda_reg=0.01, early_stopping=True, 
                                           paciencia=20, verbose=False):
        """Entrena un modelo de regresión lineal con regularización L2 y early stopping."""
        # Inicializar peso w
        w = 0
        error_history = []
        best_w = w
        best_error = float('inf')
        counter_paciencia = 0
        
        # Entrenamiento con descenso de gradiente
        for epoch in range(epochs):
            # Predicción con el modelo actual
            y_pred = w * X
            
            # Calcular error (error cuadrático medio) con término de regularización
            error = y_pred - y
            mse = np.mean(error**2) + lambda_reg * (w**2)  # Regularización L2
            error_history.append(mse)
            
            # Guardar el mejor modelo hasta ahora
            if mse < best_error:
                best_error = mse
                best_w = w
                counter_paciencia = 0  # Reiniciar contador de paciencia
            else:
                counter_paciencia += 1
            
            # Early stopping si el error no mejora durante 'paciencia' épocas
            if early_stopping and counter_paciencia >= paciencia:
                if verbose:
                    print(f"Early stopping en época {epoch+1}")
                break
            
            # Actualizar peso w con regularización
            gradient = np.mean(error * X) + lambda_reg * w  # Gradiente con regularización
            w = w - learning_rate * gradient
            
            # Mostrar progreso si verbose es True
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Época {epoch+1}/{epochs}, Error: {mse:.6f}, Peso w: {w:.6f}")
        
        # Usar el mejor peso encontrado
        if best_error < np.mean(error_history):
            w = best_w
            if verbose:
                print(f"Usando el mejor peso encontrado: w = {w:.6f} con error {best_error:.6f}")
        
        return w, error_history
    
    def _validacion_cruzada(self, X, y, k_folds=5, learning_rate=0.01, epochs=100, lambda_reg=0.01):
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
            w, _ = self._entrenar_modelo_con_regularizacion(
                X_entrenamiento, y_entrenamiento, 
                learning_rate=learning_rate, 
                epochs=epochs, 
                lambda_reg=lambda_reg,
                verbose=False
            )
            
            # Evaluar en conjunto de validación
            y_pred_val = w * X_validacion
            mse_val = np.mean((y_pred_val - y_validacion)**2)
            errores_validacion.append(mse_val)
            pesos.append(w)
        
        # Calcular error promedio y desviación estándar
        error_promedio = np.mean(errores_validacion)
        error_std = np.std(errores_validacion)
        
        # Seleccionar el mejor peso (el del fold con menor error)
        mejor_fold = np.argmin(errores_validacion)
        mejor_peso = pesos[mejor_fold]
        
        return mejor_peso, error_promedio, error_std
    
    def _busqueda_hiperparametros(self, X, y, verbose=False):
        """Realiza una búsqueda de hiperparámetros para encontrar los mejores valores."""
        # Definir grids de hiperparámetros
        learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
        lambdas_reg = [0.0, 0.001, 0.01, 0.1, 0.5]
        
        mejor_error = float('inf')
        mejores_params = {}
        resultados = []
        
        # Dividir datos para validación
        X_train, y_train, X_val, y_val = self._dividir_datos(X, y, porcentaje_entrenamiento=0.8)
        
        if verbose:
            print("Iniciando búsqueda de hiperparámetros...")
        
        # Probar todas las combinaciones
        for lr in learning_rates:
            for lambda_reg in lambdas_reg:
                # Entrenar modelo con estos hiperparámetros
                w, _ = self._entrenar_modelo_con_regularizacion(
                    X_train, y_train,
                    learning_rate=lr,
                    lambda_reg=lambda_reg,
                    epochs=200,  # Más épocas para búsqueda de hiperparámetros
                    verbose=False
                )
                
                # Evaluar en conjunto de validación
                y_pred_val = w * X_val
                mse_val = np.mean((y_pred_val - y_val)**2)
                
                # Guardar resultados
                resultados.append({
                    'learning_rate': lr,
                    'lambda_reg': lambda_reg,
                    'w': w,
                    'mse_val': mse_val
                })
                
                if mse_val < mejor_error:
                    mejor_error = mse_val
                    mejores_params = {
                        'learning_rate': lr,
                        'lambda_reg': lambda_reg,
                        'w': w,
                        'mse_val': mse_val
                    }
                
                if verbose:
                    print(f"LR={lr}, Lambda={lambda_reg}: MSE={mse_val:.6f}, w={w:.6f}")
        
        if verbose:
            print(f"\nMejores parámetros encontrados:")
            print(f"Learning Rate: {mejores_params['learning_rate']}")
            print(f"Lambda Regularización: {mejores_params['lambda_reg']}")
            print(f"Peso w: {mejores_params['w']:.6f}")
            print(f"MSE Validación: {mejores_params['mse_val']:.6f}")
        
        return mejores_params, resultados
    
    def entrenar_con_datos_simulados(self, num_precios=20, num_dias=365, 
                                    incluir_datos_reales=True, verbose=False):
        """Entrena el modelo utilizando datos simulados y opcionalmente datos reales."""
        # Generar datos simulados
        if verbose:
            print(f"Generando {num_precios} precios simulados con {num_dias} días cada uno...")
        
        datos_simulados = generar_dataset_simulado(num_precios=num_precios, num_dias=num_dias)
        
        # Combinar con datos reales si se solicita y existen
        datos_combinados = datos_simulados.copy()
        if incluir_datos_reales and self.datos_reales:
            if verbose:
                print(f"Combinando con {len(self.datos_reales)} registros de datos reales")
            
            # Dar más peso a los datos reales duplicándolos
            for _ in range(3):  # Duplicar 3 veces para dar más peso
                datos_combinados.extend(self.datos_reales)
        
        # Preparar datos para entrenamiento
        X = np.array([float(row[0]) for row in datos_combinados])  # Precios
        y = np.array([float(row[2]) for row in datos_combinados])  # Ventas (demanda)
        
        # Normalizar datos
        X_norm, y_norm, X_mean, X_std, y_mean, y_std = self._normalizar_datos(X, y)
        
        # Buscar mejores hiperparámetros
        if verbose:
            print("\nBuscando mejores hiperparámetros...")
        
        mejores_params, resultados_busqueda = self._busqueda_hiperparametros(X_norm, y_norm, verbose=verbose)
        
        # Realizar validación cruzada con los mejores hiperparámetros
        if verbose:
            print("\nRealizando validación cruzada con los mejores hiperparámetros...")
        
        mejor_w, cv_error, cv_std = self._validacion_cruzada(
            X_norm, y_norm,
            k_folds=5,
            learning_rate=mejores_params['learning_rate'],
            lambda_reg=mejores_params['lambda_reg']
        )
        
        if verbose:
            print(f"Validación cruzada completada:")
            print(f"Error promedio: {cv_error:.6f} ± {cv_std:.6f}")
            print(f"Mejor peso w: {mejor_w:.6f}")
        
        # Entrenar modelo final con todos los datos
        if verbose:
            print("\nEntrenando modelo final con todos los datos...")
        
        w_final, error_history = self._entrenar_modelo_con_regularizacion(
            X_norm, y_norm,
            learning_rate=mejores_params['learning_rate'],
            lambda_reg=mejores_params['lambda_reg'],
            epochs=300,  # Más épocas para el modelo final
            verbose=verbose
        )
        
        # Guardar el mejor modelo y métricas
        self.mejor_modelo = {
            'w': w_final,
            'X_mean': X_mean,
            'X_std': X_std,
            'y_mean': y_mean,
            'y_std': y_std,
            'error_final': error_history[-1] if error_history else 0,
            'hiperparametros': {
                'learning_rate': mejores_params['learning_rate'],
                'lambda_reg': mejores_params['lambda_reg']
            }
        }
        
        self.metricas_validacion = {
            'cv_error': cv_error,
            'cv_std': cv_std,
            'error_history': error_history
        }
        
        self.historial_entrenamiento.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_datos': len(datos_combinados),
            'num_datos_reales': len(self.datos_reales),
            'num_datos_simulados': len(datos_simulados),
            'error_final': self.mejor_modelo['error_final'],
            'w': w_final
        })
        
        if verbose:
            print(f"\nEntrenamiento completado con éxito!")
            print(f"Error final: {self.mejor_modelo['error_final']:.6f}")
            print(f"Peso w final: {w_final:.6f}")
        
        return self.mejor_modelo, self.metricas_validacion
    
    def predecir_demanda(self, precio):
        """Predice la demanda para un precio dado usando el modelo entrenado."""
        if self.mejor_modelo is None:
            print("Error: El modelo no ha sido entrenado. Ejecute entrenar_con_datos_simulados primero.")
            return None
        
        # Extraer parámetros del modelo
        w = self.mejor_modelo['w']
        X_mean = self.mejor_modelo['X_mean']
        X_std = self.mejor_modelo['X_std']
        y_mean = self.mejor_modelo['y_mean']
        y_std = self.mejor_modelo['y_std']
        
        # Normalizar el precio
        precio_norm = (precio - X_mean) / X_std
        
        # Calcular predicción normalizada
        y_pred_norm = w * precio_norm
        
        # Desnormalizar para obtener la predicción en la escala original
        y_pred = (y_pred_norm * y_std) + y_mean if y_std > 0 else y_pred_norm
        
        # Asegurar un valor mínimo razonable (mínimo 0.1 kg = 100 gramos)
        return max(0.1, y_pred)
    
    def evaluar_modelo_con_simulacion(self, num_precios=10, num_simulaciones=100, verbose=False):
        """Evalúa el modelo entrenado con datos simulados adicionales."""
        if self.mejor_modelo is None:
            print("Error: El modelo no ha sido entrenado. Ejecute entrenar_con_datos_simulados primero.")
            return None
        
        # Generar rango de precios para evaluación
        min_precio = 5.0
        max_precio = 25.0
        precios_eval = np.linspace(min_precio, max_precio, num_precios)
        
        resultados = []
        errores = []
        
        for precio in precios_eval:
            # Simular demanda real para este precio
            stats_simuladas = simular_escenario_precio(precio, num_simulaciones=num_simulaciones)
            demanda_simulada = stats_simuladas['media']
            
            # Predecir con nuestro modelo
            demanda_predicha = self.predecir_demanda(precio)
            
            # Calcular error
            error_abs = abs(demanda_predicha - demanda_simulada)
            error_rel = error_abs / demanda_simulada if demanda_simulada > 0 else 0
            
            resultados.append({
                'precio': precio,
                'demanda_simulada': demanda_simulada,
                'demanda_predicha': demanda_predicha,
                'error_abs': error_abs,
                'error_rel': error_rel * 100,  # En porcentaje
                'stats_simuladas': stats_simuladas
            })
            
            errores.append(error_rel * 100)  # En porcentaje
            
            if verbose:
                print(f"Precio: {precio:.2f}, ")
                print(f"  Demanda simulada: {demanda_simulada:.2f}")
                print(f"  Demanda predicha: {demanda_predicha:.2f}")
                print(f"  Error: {error_rel*100:.2f}%")
        
        # Calcular error promedio
        error_promedio = np.mean(errores)
        
        if verbose:
            print(f"\nError promedio: {error_promedio:.2f}%")
        
        return resultados, error_promedio
    
    def visualizar_resultados(self):
        """Visualiza los resultados del entrenamiento y evaluación."""
        if not matplotlib_disponible:
            print("No se puede visualizar porque matplotlib no está instalado.")
            print("Para instalar matplotlib, ejecuta: pip install matplotlib")
            return
        
        if self.mejor_modelo is None:
            print("Error: El modelo no ha sido entrenado. Ejecute entrenar_con_datos_simulados primero.")
            return
        
        # Crear figura con subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Curva de aprendizaje
        if 'error_history' in self.metricas_validacion and self.metricas_validacion['error_history']:
            axs[0, 0].plot(self.metricas_validacion['error_history'])
            axs[0, 0].set_title('Curva de Aprendizaje')
            axs[0, 0].set_xlabel('Época')
            axs[0, 0].set_ylabel('Error Cuadrático Medio')
            axs[0, 0].grid(True)
        
        # 2. Relación Precio-Demanda
        # Generar datos para la curva
        precios = np.linspace(5, 25, 100)
        demandas = [self.predecir_demanda(p) for p in precios]
        
        # Datos reales
        precios_reales = [float(row[0]) for row in self.datos_reales] if self.datos_reales else []
        demandas_reales = [float(row[2]) for row in self.datos_reales] if self.datos_reales else []
        
        axs[0, 1].plot(precios, demandas, 'b-', label='Modelo')
        if precios_reales:
            axs[0, 1].scatter(precios_reales, demandas_reales, color='red', label='Datos reales')
        axs[0, 1].set_title('Relación Precio-Demanda')
        axs[0, 1].set_xlabel('Precio')
        axs[0, 1].set_ylabel('Demanda (kg)')
        axs[0, 1].grid(True)
        axs[0, 1].legend()
        
        # 3. Evaluación con simulación
        resultados_eval, _ = self.evaluar_modelo_con_simulacion(num_precios=10, verbose=False)
        
        precios_eval = [r['precio'] for r in resultados_eval]
        demandas_sim = [r['demanda_simulada'] for r in resultados_eval]
        demandas_pred = [r['demanda_predicha'] for r in resultados_eval]
        
        axs[1, 0].plot(precios_eval, demandas_sim, 'g-', label='Simulada')
        axs[1, 0].plot(precios_eval, demandas_pred, 'b--', label='Predicha')
        axs[1, 0].set_title('Evaluación del Modelo')
        axs[1, 0].set_xlabel('Precio')
        axs[1, 0].set_ylabel('Demanda (kg)')
        axs[1, 0].grid(True)
        axs[1, 0].legend()
        
        # 4. Error relativo por precio
        errores_rel = [r['error_rel'] for r in resultados_eval]
        
        axs[1, 1].bar(precios_eval, errores_rel)
        axs[1, 1].set_title('Error Relativo por Precio')
        axs[1, 1].set_xlabel('Precio')
        axs[1, 1].set_ylabel('Error Relativo (%)')
        axs[1, 1].grid(True)
        
        # Ajustar layout y mostrar
        plt.tight_layout()
        plt.show()
    
    def guardar_modelo_entrenado(self, archivo='modelo_entrenado.csv'):
        """Guarda los parámetros del modelo entrenado en un archivo CSV."""
        if self.mejor_modelo is None:
            print("Error: El modelo no ha sido entrenado. Ejecute entrenar_con_datos_simulados primero.")
            return False
        
        try:
            with open(archivo, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['parametro', 'valor'])
                for key, value in self.mejor_modelo.items():
                    if key != 'hiperparametros':
                        writer.writerow([key, value])
                
                # Guardar hiperparámetros
                for key, value in self.mejor_modelo['hiperparametros'].items():
                    writer.writerow([f"hiperparametros_{key}", value])
            
            print(f"Modelo guardado exitosamente en {archivo}")
            return True
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")
            return False

# Función principal para entrenar automáticamente
def entrenar_automaticamente(num_precios=50, num_dias=365, incluir_datos_reales=True, verbose=True):
    """Función principal para entrenar automáticamente el modelo y visualizar resultados."""
    inicio = time.time()
    
    # Crear entrenador
    entrenador = EntrenadorAutomatico()
    
    # Entrenar modelo
    if verbose:
        print(f"Iniciando entrenamiento automático con {num_precios} precios y {num_dias} días simulados...")
    
    modelo, metricas = entrenador.entrenar_con_datos_simulados(
        num_precios=num_precios,
        num_dias=num_dias,
        incluir_datos_reales=incluir_datos_reales,
        verbose=verbose
    )
    
    # Evaluar modelo
    if verbose:
        print("\nEvaluando modelo con simulaciones adicionales...")
    
    _, error_promedio = entrenador.evaluar_modelo_con_simulacion(verbose=verbose)
    
    # Visualizar resultados
    if matplotlib_disponible and verbose:
        print("\nVisualizando resultados...")
        entrenador.visualizar_resultados()
    
    # Guardar modelo
    entrenador.guardar_modelo_entrenado()
    
    # Mostrar tiempo total
    tiempo_total = time.time() - inicio
    if verbose:
        print(f"\nEntrenamiento completado en {tiempo_total:.2f} segundos")
        print(f"Error promedio del modelo: {error_promedio:.2f}%")
        print(f"Peso final del modelo w = {modelo['w']:.6f}")
    
    return modelo, entrenador

# Ejemplo de uso:
# modelo, entrenador = entrenar_automaticamente(num_precios=50, num_dias=365, verbose=True)