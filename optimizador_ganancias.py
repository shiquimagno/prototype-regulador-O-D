import numpy as np
import csv
from datetime import datetime
from simulador_mercado import generar_dataset_simulado, simular_escenario_precio
from importador_datos_mercado import importar_datos_mercado, obtener_factores_estacionales_reales

class OptimizadorGanancias:
    """Clase para optimizar las ganancias del comerciante considerando costos de adquisición
    y precios de venta, equilibrando oferta y demanda."""
    
    def __init__(self, archivo_datos='mondongos.csv', archivo_costos=None, usar_datos_mercado=False):
        """Inicializa el optimizador con los parámetros base.
        
        Args:
            archivo_datos: Ruta al archivo CSV con los datos reales de ventas
            archivo_costos: Ruta al archivo CSV con los costos de adquisición (opcional)
            usar_datos_mercado: Si es True, importa datos de mercados reales
        """
        self.archivo_datos = archivo_datos
        self.archivo_costos = archivo_costos
        self.usar_datos_mercado = usar_datos_mercado
        self.datos_reales = self._leer_datos_csv()
        self.datos_mercado = self._importar_datos_mercado() if usar_datos_mercado else []
        self.costos_adquisicion = self._leer_costos() if archivo_costos else self._estimar_costos()
        self.mejor_modelo = None
        self.historial_optimizacion = []
        self.metricas_validacion = {}
        
        # Parámetros para el modelo de múltiples pesos
        self.num_pesos = 3  # Aumentamos a 3 pesos para capturar relaciones más complejas
        
        # Factores estacionales para ajustar el modelo
        self.factores_estacionales = self._obtener_factores_estacionales() if usar_datos_mercado else {}
    
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
            print("Se utilizarán solo datos simulados para la optimización.")
            return []
    
    def _leer_costos(self):
        """Lee los costos de adquisición del archivo CSV."""
        costos = {}
        try:
            with open(self.archivo_costos, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Saltamos la cabecera
                for row in reader:
                    precio = float(row[0])
                    costo = float(row[1])
                    costos[precio] = costo
            return costos
        except FileNotFoundError:
            print(f"Advertencia: No se encontró el archivo de costos {self.archivo_costos}")
            print("Se estimarán los costos automáticamente.")
            return self._estimar_costos()
    
    def _estimar_costos(self, factor_costo=0.6):
        """Estima los costos de adquisición como un porcentaje del precio de venta.
        
        Args:
            factor_costo: Factor para estimar el costo (por defecto 60% del precio de venta)
        """
        costos = {}
        # Si tenemos datos reales, usamos sus precios
        if self.datos_reales:
            for row in self.datos_reales:
                precio = float(row[0])
                costo = precio * factor_costo
                costos[precio] = costo
        return costos
    
    def _normalizar_datos(self, X, y):
        """Normaliza los datos para mejorar la convergencia del modelo."""
        X_mean = np.mean(X)
        X_std = np.std(X) if np.std(X) > 0 else 1
        X_norm = (X - X_mean) / X_std
        
        y_mean = np.mean(y)
        y_std = np.std(y) if np.std(y) > 0 else 1
        y_norm = (y - y_mean) / y_std if y_std > 0 else y
        
        return X_norm, y_norm, X_mean, X_std, y_mean, y_std
    
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
    
    def _entrenar_modelo_ganancias(self, X, y, costos, learning_rate=0.01, epochs=200, 
                                 lambda_reg=0.01, early_stopping=True, 
                                 paciencia=20, verbose=False):
        """Entrena un modelo para maximizar ganancias considerando costos.
        
        Args:
            X: Array de precios
            y: Array de demanda (ventas)
            costos: Diccionario de costos por precio
            learning_rate: Tasa de aprendizaje
            epochs: Número de épocas
            lambda_reg: Factor de regularización
            early_stopping: Si se debe detener temprano
            paciencia: Épocas de paciencia para early stopping
            verbose: Si se debe mostrar información detallada
            
        Returns:
            Pesos del modelo, historial de ganancias
        """
        # Calcular características
        X_features = self._calcular_caracteristicas(X)
        
        # Inicializar pesos
        W = np.zeros(X_features.shape[1])
        ganancia_history = []
        best_W = W.copy()
        best_ganancia = float('-inf')  # Queremos maximizar ganancias
        counter_paciencia = 0
        
        # Convertir costos a array
        costos_array = np.array([costos.get(precio, precio * 0.6) for precio in X])
        
        # Entrenamiento con descenso de gradiente
        for epoch in range(epochs):
            # Predicción con el modelo actual
            y_pred = np.dot(X_features, W)
            
            # Calcular ganancias = ingresos - costos
            ingresos = X * y_pred  # precio * demanda_predicha
            gastos = costos_array * y_pred  # costo * demanda_predicha
            ganancias = ingresos - gastos
            ganancia_total = np.sum(ganancias)
            
            # Añadir término de regularización para penalizar pesos grandes
            ganancia_regularizada = ganancia_total - lambda_reg * np.sum(W**2)
            ganancia_history.append(ganancia_regularizada)
            
            # Guardar el mejor modelo hasta ahora (maximizando ganancias)
            if ganancia_regularizada > best_ganancia:
                best_ganancia = ganancia_regularizada
                best_W = W.copy()
                counter_paciencia = 0
            else:
                counter_paciencia += 1
            
            # Early stopping si la ganancia no mejora durante 'paciencia' épocas
            if early_stopping and counter_paciencia >= paciencia:
                if verbose:
                    print(f"Early stopping en época {epoch+1}")
                break
            
            # Calcular gradiente para maximizar ganancias
            # Derivada de ganancias respecto a W
            gradient = np.zeros_like(W)
            for j in range(len(W)):
                # Derivada de y_pred respecto a W[j]
                d_y_pred = X_features[:, j]
                # Derivada de ganancias respecto a y_pred
                d_ganancias = X - costos_array
                # Derivada de ganancias respecto a W[j]
                gradient[j] = np.mean(d_ganancias * d_y_pred)
            
            # Añadir término de regularización al gradiente
            gradient_reg = gradient - 2 * lambda_reg * W
            
            # Actualizar pesos (ascenso de gradiente para maximizar)
            W = W + learning_rate * gradient_reg
            
            # Mostrar progreso si verbose es True
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Época {epoch+1}/{epochs}, Ganancia: {ganancia_regularizada:.2f}, Pesos: {W}")
        
        # Usar los mejores pesos encontrados
        if best_ganancia > ganancia_history[0]:
            W = best_W
            if verbose:
                print(f"Usando los mejores pesos encontrados con ganancia {best_ganancia:.2f}")
        
        return W, ganancia_history
    
    def _importar_datos_mercado(self):
        """Importa datos de mercados reales usando el importador."""
        try:
            datos_mercado = importar_datos_mercado()
            print(f"Se importaron {len(datos_mercado)} registros de datos de mercados reales")
            return datos_mercado
        except Exception as e:
            print(f"Error al importar datos de mercado: {e}")
            return []
    
    def _obtener_factores_estacionales(self):
        """Obtiene factores estacionales de mercados reales."""
        try:
            return obtener_factores_estacionales_reales()
        except Exception as e:
            print(f"Error al obtener factores estacionales: {e}")
            return {}
    
    def optimizar_con_datos_simulados(self, num_precios=20, num_dias=365, 
                                     incluir_datos_reales=True, verbose=False):
        """Optimiza el modelo de ganancias utilizando datos simulados y opcionalmente datos reales."""
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
            for _ in range(5):  # Duplicar 5 veces para dar más peso (aumentado de 3)
                datos_combinados.extend(self.datos_reales)
        
        # Añadir datos de mercados reales si están disponibles
        if self.usar_datos_mercado and self.datos_mercado:
            if verbose:
                print(f"Añadiendo {len(self.datos_mercado)} registros de datos de mercados reales")
            
            # Dar más peso a los datos de mercado duplicándolos
            for _ in range(3):  # Duplicar 3 veces para dar peso significativo
                datos_combinados.extend(self.datos_mercado)
        
        # Preparar datos para entrenamiento
        X = np.array([float(row[0]) for row in datos_combinados])  # Precios
        y = np.array([float(row[2]) for row in datos_combinados])  # Ventas (demanda)
        
        # Asegurar que tenemos costos para todos los precios
        for precio in X:
            if precio not in self.costos_adquisicion:
                self.costos_adquisicion[precio] = precio * 0.6  # Estimación por defecto
        
        # Normalizar datos
        X_norm, y_norm, X_mean, X_std, y_mean, y_std = self._normalizar_datos(X, y)
        
        # Normalizar costos también
        costos_norm = {precio: (costo - X_mean) / X_std for precio, costo in self.costos_adquisicion.items()}
        
        # Entrenar modelo para maximizar ganancias
        if verbose:
            print("\nEntrenando modelo para maximizar ganancias...")
        
        W, ganancia_history = self._entrenar_modelo_ganancias(
            X_norm, y_norm, costos_norm,
            learning_rate=0.02,  # Aumentado para convergencia más rápida
            epochs=300,
            lambda_reg=0.005,  # Reducido para permitir más flexibilidad
            verbose=verbose
        )
        
        # Guardar el mejor modelo y métricas
        self.mejor_modelo = {
            'W': W,
            'X_mean': X_mean,
            'X_std': X_std,
            'y_mean': y_mean,
            'y_std': y_std,
            'ganancia_final': ganancia_history[-1] if ganancia_history else 0,
        }
        
        self.metricas_validacion = {
            'ganancia_history': ganancia_history
        }
        
        self.historial_optimizacion.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_datos': len(datos_combinados),
            'num_datos_reales': len(self.datos_reales),
            'num_datos_simulados': len(datos_simulados),
            'ganancia_final': self.mejor_modelo['ganancia_final'],
            'W': W
        })
        
        if verbose:
            print(f"\nOptimización completada con éxito!")
            print(f"Ganancia final: {self.mejor_modelo['ganancia_final']:.2f}")
            print(f"Pesos finales: {W}")
        
        return self.mejor_modelo, self.metricas_validacion
    
    def predecir_demanda_y_ganancia(self, precio):
        """Predice la demanda y ganancia para un precio dado usando el modelo optimizado."""
        if self.mejor_modelo is None:
            print("Error: El modelo no ha sido optimizado. Ejecute optimizar_con_datos_simulados primero.")
            return None, None
        
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
        
        # Calcular ganancia esperada
        costo = self.costos_adquisicion.get(precio, precio * 0.6)
        ganancia_pred = (precio - costo) * demanda_pred
        
        return demanda_pred, ganancia_pred
    
    def recomendar_precio_optimo(self, rango_precios=None):
        """Recomienda el precio que maximiza las ganancias."""
        if self.mejor_modelo is None:
            print("Error: El modelo no ha sido optimizado. Ejecute optimizar_con_datos_simulados primero.")
            return None
        
        # Si no se proporciona un rango, usar precios de los datos reales o un rango predeterminado
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
            demanda_pred, ganancia_pred = self.predecir_demanda_y_ganancia(precio)
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

# Función auxiliar para usar directamente
def optimizar_ganancias(archivo_datos='mondongos.csv', usar_datos_mercado=False, verbose=True):
    """Función auxiliar para optimizar ganancias directamente.
    
    Args:
        archivo_datos: Ruta al archivo CSV con los datos reales de ventas
        usar_datos_mercado: Si es True, importa datos de mercados reales
        verbose: Si es True, muestra información detallada
    """
    optimizador = OptimizadorGanancias(archivo_datos=archivo_datos, usar_datos_mercado=usar_datos_mercado)
    modelo, metricas = optimizador.optimizar_con_datos_simulados(verbose=verbose)
    recomendacion = optimizador.recomendar_precio_optimo()
    
    if verbose and recomendacion:
        print("\nRecomendación de precio óptimo:")
        print(f"Precio: {recomendacion['precio_optimo']:.2f}")
        print(f"Demanda esperada: {recomendacion['demanda_esperada']:.2f} kg")
        print(f"Ganancia esperada: {recomendacion['ganancia_esperada']:.2f}")
        print("\nTop 5 precios con mejores ganancias:")
        for i, (precio, demanda, ganancia) in enumerate(recomendacion['top_precios']):
            print(f"{i+1}. Precio: {precio:.2f}, Demanda: {demanda:.2f} kg, Ganancia: {ganancia:.2f}")
    
    return optimizador, recomendacion

# Si se ejecuta como script principal
if __name__ == "__main__":
    optimizador, recomendacion = optimizar_ganancias(verbose=True)