import os
import json
import pickle
import numpy as np
import argparse
from datetime import datetime

# Intentar importar matplotlib para visualización
try:
    import matplotlib.pyplot as plt
    matplotlib_disponible = True
except ImportError:
    matplotlib_disponible = False
    print("Error: matplotlib no está instalado. La visualización no estará disponible.")
    print("Para instalar matplotlib, ejecuta: pip install matplotlib")

class VisualizadorConvergencia:
    """
    Clase para visualizar la convergencia del error durante el entrenamiento masivo
    y comparar diferentes modelos entrenados.
    """
    
    def __init__(self, directorio_modelos='modelos_entrenados'):
        """
        Inicializa el visualizador con el directorio donde se encuentran los modelos.
        
        Args:
            directorio_modelos: Directorio donde se guardan los modelos entrenados
        """
        self.directorio_modelos = directorio_modelos
        self.modelos_disponibles = self._buscar_modelos_disponibles()
    
    def _buscar_modelos_disponibles(self):
        """
        Busca todos los modelos disponibles en el directorio de modelos.
        
        Returns:
            Diccionario con información de los modelos disponibles
        """
        modelos = {}
        
        if not os.path.exists(self.directorio_modelos):
            print(f"El directorio {self.directorio_modelos} no existe.")
            return modelos
        
        # Buscar archivos de configuración
        for archivo in os.listdir(self.directorio_modelos):
            if archivo.startswith('config_') and archivo.endswith('.json'):
                ruta_completa = os.path.join(self.directorio_modelos, archivo)
                try:
                    with open(ruta_completa, 'r') as f:
                        config = json.load(f)
                    
                    # Extraer ID de comerciante y tipo de producto del nombre del archivo
                    # Formato: config_COMERCIANTE_PRODUCTO.json
                    partes = archivo[7:-5].split('_')
                    if len(partes) >= 2:
                        id_comerciante = partes[0]
                        tipo_producto = '_'.join(partes[1:])  # Por si el producto tiene guiones bajos
                        
                        # Guardar información del modelo
                        modelos[f"{id_comerciante}_{tipo_producto}"] = {
                            'id_comerciante': id_comerciante,
                            'tipo_producto': tipo_producto,
                            'iteraciones': config.get('iteraciones_totales', 0),
                            'error': config.get('mejor_error', float('inf')),
                            'fecha': config.get('fecha_ultimo_entrenamiento', 'Desconocida'),
                            'tiempo_total': config.get('tiempo_total_entrenamiento', 0),
                            'ruta_config': ruta_completa,
                            'ruta_modelo': os.path.join(self.directorio_modelos, f'modelo_{id_comerciante}_{tipo_producto}.pkl')
                        }
                except Exception as e:
                    print(f"Error al leer {archivo}: {e}")
        
        return modelos
    
    def listar_modelos(self):
        """
        Muestra una lista de todos los modelos disponibles con su información.
        """
        if not self.modelos_disponibles:
            print("No se encontraron modelos entrenados.")
            return
        
        print("\nModelos entrenados disponibles:")
        print("-" * 100)
        print(f"{'ID':<20} {'Comerciante':<15} {'Producto':<15} {'Iteraciones':<12} {'Error':<15} {'Fecha':<20}")
        print("-" * 100)
        
        for id_modelo, info in self.modelos_disponibles.items():
            print(f"{id_modelo:<20} {info['id_comerciante']:<15} {info['tipo_producto']:<15} "
                  f"{info['iteraciones']:<12,} {info['error']:<15.8f} {info['fecha']:<20}")
    
    def cargar_modelo(self, id_comerciante, tipo_producto):
        """
        Carga un modelo específico para visualización.
        
        Args:
            id_comerciante: ID del comerciante
            tipo_producto: Tipo de producto
            
        Returns:
            Diccionario con el modelo cargado o None si no se encuentra
        """
        id_modelo = f"{id_comerciante}_{tipo_producto}"
        
        if id_modelo not in self.modelos_disponibles:
            print(f"No se encontró el modelo para comerciante '{id_comerciante}' y producto '{tipo_producto}'")
            return None
        
        ruta_modelo = self.modelos_disponibles[id_modelo]['ruta_modelo']
        
        try:
            with open(ruta_modelo, 'rb') as f:
                modelo = pickle.load(f)
            return modelo
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return None
    
    def visualizar_convergencia(self, id_comerciante, tipo_producto, historial_archivo=None):
        """
        Visualiza la convergencia del error durante el entrenamiento.
        
        Args:
            id_comerciante: ID del comerciante
            tipo_producto: Tipo de producto
            historial_archivo: Archivo con historial de errores (opcional)
        """
        if not matplotlib_disponible:
            print("Error: matplotlib no está instalado. No se puede visualizar la convergencia.")
            return
        
        # Cargar modelo
        modelo = self.cargar_modelo(id_comerciante, tipo_producto)
        if modelo is None:
            return
        
        # Cargar historial de errores si se proporciona un archivo
        errores = []
        if historial_archivo and os.path.exists(historial_archivo):
            try:
                errores = np.loadtxt(historial_archivo)
            except Exception as e:
                print(f"Error al cargar historial de errores: {e}")
        
        # Si no hay historial, mostrar solo información del modelo
        if not errores:
            print("\nInformación del modelo:")
            print(f"  • Comerciante: {id_comerciante}")
            print(f"  • Producto: {tipo_producto}")
            print(f"  • Iteraciones totales: {self.modelos_disponibles[f'{id_comerciante}_{tipo_producto}']['iteraciones']:,}")
            print(f"  • Error final: {modelo['error_final']:.8f}")
            print(f"  • Fecha de entrenamiento: {modelo['fecha']}")
            print("\nNo se encontró historial de errores para visualizar la convergencia.")
            return
        
        # Crear gráfico
        plt.figure(figsize=(12, 8))
        
        # Gráfico de error por iteración
        plt.subplot(2, 1, 1)
        plt.plot(errores)
        plt.title(f"Convergencia del Error - {tipo_producto} ({id_comerciante})")
        plt.xlabel('Iteración')
        plt.ylabel('Error (MSE)')
        plt.grid(True)
        
        # Gráfico de error suavizado (media móvil)
        plt.subplot(2, 1, 2)
        window_size = min(50, len(errores))
        if window_size > 0:
            errores_suavizados = np.convolve(errores, np.ones(window_size)/window_size, mode='valid')
            plt.plot(errores_suavizados)
            plt.title(f"Error Suavizado (Media Móvil de {window_size} iteraciones)")
            plt.xlabel('Iteración')
            plt.ylabel('Error (MSE) - Media Móvil')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def comparar_modelos(self, modelos_ids):
        """
        Compara múltiples modelos visualizando sus curvas de demanda y ganancia.
        
        Args:
            modelos_ids: Lista de tuplas (id_comerciante, tipo_producto)
        """
        if not matplotlib_disponible:
            print("Error: matplotlib no está instalado. No se puede visualizar la comparación.")
            return
        
        # Cargar modelos
        modelos = []
        nombres = []
        for id_comerciante, tipo_producto in modelos_ids:
            modelo = self.cargar_modelo(id_comerciante, tipo_producto)
            if modelo is not None:
                modelos.append(modelo)
                nombres.append(f"{tipo_producto} ({id_comerciante})")
        
        if not modelos:
            print("No se pudieron cargar los modelos para comparación.")
            return
        
        # Crear rango de precios común
        rango_precios = np.linspace(5, 25, 100)  # 100 puntos entre 5 y 25
        
        # Crear gráfico
        plt.figure(figsize=(12, 8))
        
        # Curvas de demanda
        plt.subplot(2, 1, 1)
        for i, modelo in enumerate(modelos):
            # Extraer parámetros del modelo
            W = modelo['W']
            X_mean = modelo['X_mean']
            X_std = modelo['X_std']
            y_mean = modelo['y_mean']
            y_std = modelo['y_std']
            
            # Calcular demandas
            demandas = []
            for precio in rango_precios:
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
                demanda_pred = max(0.1, demanda_pred)  # Asegurar un valor mínimo razonable
                
                demandas.append(demanda_pred)
            
            plt.plot(rango_precios, demandas, label=nombres[i])
        
        plt.title("Comparación de Curvas de Demanda")
        plt.xlabel("Precio ($)")
        plt.ylabel("Demanda (kg)")
        plt.grid(True)
        plt.legend()
        
        # Curvas de ganancia
        plt.subplot(2, 1, 2)
        for i, modelo in enumerate(modelos):
            # Extraer parámetros del modelo
            W = modelo['W']
            X_mean = modelo['X_mean']
            X_std = modelo['X_std']
            y_mean = modelo['y_mean']
            y_std = modelo['y_std']
            
            # Calcular ganancias
            ganancias = []
            for precio in rango_precios:
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
                demanda_pred = max(0.1, demanda_pred)  # Asegurar un valor mínimo razonable
                
                # Calcular ganancia (asumiendo costo = 60% del precio)
                ganancia = (precio - precio * 0.6) * demanda_pred
                ganancias.append(ganancia)
            
            plt.plot(rango_precios, ganancias, label=nombres[i])
            
            # Marcar precio óptimo
            indice_optimo = np.argmax(ganancias)
            precio_optimo = rango_precios[indice_optimo]
            ganancia_optima = ganancias[indice_optimo]
            plt.scatter([precio_optimo], [ganancia_optima], marker='o', s=100)
            plt.annotate(f"${precio_optimo:.2f}", 
                        (precio_optimo, ganancia_optima),
                        xytext=(5, 10), textcoords='offset points')
        
        plt.title("Comparación de Curvas de Ganancia")
        plt.xlabel("Precio ($)")
        plt.ylabel("Ganancia ($)")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Función principal para ejecutar el visualizador desde línea de comandos.
    """
    parser = argparse.ArgumentParser(description='Visualizador de convergencia de modelos entrenados')
    parser.add_argument('--listar', action='store_true',
                        help='Listar todos los modelos disponibles')
    parser.add_argument('--comerciante', type=str, default=None,
                        help='ID del comerciante para visualizar')
    parser.add_argument('--producto', type=str, default=None,
                        help='Tipo de producto para visualizar')
    parser.add_argument('--historial', type=str, default=None,
                        help='Archivo con historial de errores para visualizar convergencia')
    parser.add_argument('--comparar', action='store_true',
                        help='Comparar múltiples modelos (requiere --modelos)')
    parser.add_argument('--modelos', type=str, default=None,
                        help='Lista de modelos a comparar en formato "comerciante1:producto1,comerciante2:producto2,..."')
    
    args = parser.parse_args()
    
    # Crear visualizador
    visualizador = VisualizadorConvergencia()
    
    # Listar modelos disponibles
    if args.listar:
        visualizador.listar_modelos()
        return
    
    # Comparar modelos
    if args.comparar and args.modelos:
        modelos_ids = []
        for modelo_str in args.modelos.split(','):
            partes = modelo_str.split(':')
            if len(partes) == 2:
                modelos_ids.append((partes[0], partes[1]))
        
        if modelos_ids:
            visualizador.comparar_modelos(modelos_ids)
        else:
            print("Error: Formato incorrecto para --modelos. Debe ser 'comerciante1:producto1,comerciante2:producto2,...'")
        return
    
    # Visualizar convergencia de un modelo específico
    if args.comerciante and args.producto:
        visualizador.visualizar_convergencia(args.comerciante, args.producto, args.historial)
        return
    
    # Si no se especifica ninguna acción, mostrar ayuda
    parser.print_help()

# Si se ejecuta como script principal
if __name__ == "__main__":
    if not matplotlib_disponible:
        print("Error: matplotlib no está instalado. La visualización no estará disponible.")
        print("Para instalar matplotlib, ejecuta: pip install matplotlib")
    else:
        main()