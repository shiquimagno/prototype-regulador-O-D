import os
import numpy as np
import json
from datetime import datetime

class GestorHistorialEntrenamiento:
    """
    Clase para gestionar el historial de entrenamiento de modelos, permitiendo
    guardar y cargar el historial de errores, así como analizar la convergencia.
    """
    
    def __init__(self, directorio_historial='historiales_error'):
        """
        Inicializa el gestor de historial con el directorio donde se guardarán los historiales.
        
        Args:
            directorio_historial: Directorio donde se guardarán los historiales de error
        """
        self.directorio_historial = directorio_historial
        
        # Crear directorio si no existe
        if not os.path.exists(self.directorio_historial):
            os.makedirs(self.directorio_historial)
    
    def guardar_historial(self, errores, id_comerciante, tipo_producto, metadata=None):
        """
        Guarda el historial de errores en un archivo.
        
        Args:
            errores: Lista o array con los errores por iteración
            id_comerciante: ID del comerciante
            tipo_producto: Tipo de producto
            metadata: Diccionario con metadatos adicionales (opcional)
            
        Returns:
            Ruta del archivo donde se guardó el historial
        """
        # Generar nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        id_comerciante_str = id_comerciante if id_comerciante else 'general'
        tipo_producto_str = tipo_producto if tipo_producto else 'mondongo'
        
        # Archivo para los errores
        archivo_errores = os.path.join(
            self.directorio_historial, 
            f"historial_{id_comerciante_str}_{tipo_producto_str}_{timestamp}.txt"
        )
        
        # Guardar errores
        np.savetxt(archivo_errores, errores)
        
        # Si hay metadatos, guardarlos en un archivo JSON
        if metadata:
            archivo_metadata = os.path.join(
                self.directorio_historial, 
                f"metadata_{id_comerciante_str}_{tipo_producto_str}_{timestamp}.json"
            )
            with open(archivo_metadata, 'w') as f:
                json.dump(metadata, f, indent=4)
        
        return archivo_errores
    
    def cargar_historial(self, id_comerciante, tipo_producto, archivo_especifico=None):
        """
        Carga el historial de errores de un modelo.
        
        Args:
            id_comerciante: ID del comerciante
            tipo_producto: Tipo de producto
            archivo_especifico: Ruta específica al archivo (opcional)
            
        Returns:
            Tuple con (errores, metadata) o (None, None) si no se encuentra
        """
        # Si se proporciona un archivo específico, intentar cargarlo
        if archivo_especifico and os.path.exists(archivo_especifico):
            try:
                errores = np.loadtxt(archivo_especifico)
                
                # Intentar cargar metadatos si existen
                archivo_metadata = archivo_especifico.replace('historial_', 'metadata_').replace('.txt', '.json')
                metadata = None
                if os.path.exists(archivo_metadata):
                    with open(archivo_metadata, 'r') as f:
                        metadata = json.load(f)
                
                return errores, metadata
            except Exception as e:
                print(f"Error al cargar historial desde {archivo_especifico}: {e}")
                return None, None
        
        # Buscar el archivo más reciente para este comerciante y producto
        id_comerciante_str = id_comerciante if id_comerciante else 'general'
        tipo_producto_str = tipo_producto if tipo_producto else 'mondongo'
        patron = f"historial_{id_comerciante_str}_{tipo_producto_str}_"
        
        archivos = [f for f in os.listdir(self.directorio_historial) if f.startswith(patron) and f.endswith('.txt')]
        
        if not archivos:
            print(f"No se encontró historial para comerciante '{id_comerciante_str}' y producto '{tipo_producto_str}'")
            return None, None
        
        # Ordenar por fecha (más reciente primero)
        archivos.sort(reverse=True)
        archivo_mas_reciente = os.path.join(self.directorio_historial, archivos[0])
        
        try:
            errores = np.loadtxt(archivo_mas_reciente)
            
            # Intentar cargar metadatos si existen
            archivo_metadata = archivo_mas_reciente.replace('historial_', 'metadata_').replace('.txt', '.json')
            metadata = None
            if os.path.exists(archivo_metadata):
                with open(archivo_metadata, 'r') as f:
                    metadata = json.load(f)
            
            return errores, metadata
        except Exception as e:
            print(f"Error al cargar historial desde {archivo_mas_reciente}: {e}")
            return None, None
    
    def listar_historiales(self):
        """
        Lista todos los historiales disponibles.
        
        Returns:
            Lista de diccionarios con información de cada historial
        """
        if not os.path.exists(self.directorio_historial):
            print(f"El directorio {self.directorio_historial} no existe.")
            return []
        
        historiales = []
        for archivo in os.listdir(self.directorio_historial):
            if archivo.startswith('historial_') and archivo.endswith('.txt'):
                # Extraer información del nombre del archivo
                # Formato: historial_COMERCIANTE_PRODUCTO_TIMESTAMP.txt
                partes = archivo[10:-4].split('_')  # Quitar 'historial_' y '.txt'
                if len(partes) >= 3:
                    # El timestamp son los últimos 2 elementos (fecha y hora)
                    timestamp = '_'.join(partes[-2:])  # YYYYMMDD_HHMMSS
                    # El producto puede tener guiones bajos, así que unimos todo lo que no es timestamp
                    id_comerciante = partes[0]
                    tipo_producto = '_'.join(partes[1:-2])
                    
                    # Intentar convertir timestamp a fecha legible
                    try:
                        fecha = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        fecha = timestamp
                    
                    # Obtener tamaño del archivo
                    ruta_completa = os.path.join(self.directorio_historial, archivo)
                    tamaño_kb = os.path.getsize(ruta_completa) / 1024
                    
                    # Intentar obtener número de iteraciones (líneas en el archivo)
                    try:
                        with open(ruta_completa, 'r') as f:
                            num_iteraciones = sum(1 for _ in f)
                    except:
                        num_iteraciones = "Desconocido"
                    
                    historiales.append({
                        'id_comerciante': id_comerciante,
                        'tipo_producto': tipo_producto,
                        'fecha': fecha,
                        'archivo': archivo,
                        'ruta': ruta_completa,
                        'tamaño_kb': tamaño_kb,
                        'iteraciones': num_iteraciones
                    })
        
        # Ordenar por fecha (más reciente primero)
        historiales.sort(key=lambda x: x['fecha'], reverse=True)
        return historiales
    
    def analizar_convergencia(self, errores):
        """
        Analiza la convergencia del error y proporciona estadísticas.
        
        Args:
            errores: Lista o array con los errores por iteración
            
        Returns:
            Diccionario con estadísticas de convergencia
        """
        if errores is None or len(errores) == 0:
            return None
        
        # Convertir a numpy array si no lo es
        if not isinstance(errores, np.ndarray):
            errores = np.array(errores)
        
        # Calcular estadísticas básicas
        error_inicial = errores[0]
        error_final = errores[-1]
        error_minimo = np.min(errores)
        error_maximo = np.max(errores)
        error_promedio = np.mean(errores)
        error_mediana = np.median(errores)
        error_std = np.std(errores)
        
        # Calcular mejora porcentual
        if error_inicial > 0:
            mejora_porcentual = (error_inicial - error_final) / error_inicial * 100
        else:
            mejora_porcentual = 0
        
        # Calcular tasa de convergencia (promedio de mejora por iteración)
        num_iteraciones = len(errores)
        if num_iteraciones > 1 and error_inicial > error_final:
            tasa_convergencia = (error_inicial - error_final) / (num_iteraciones - 1)
        else:
            tasa_convergencia = 0
        
        # Detectar estancamiento (si el error no mejora significativamente en las últimas iteraciones)
        ventana_estancamiento = min(100, num_iteraciones // 10)  # 10% de las iteraciones o 100, lo que sea menor
        if ventana_estancamiento > 0 and num_iteraciones > ventana_estancamiento:
            error_reciente_min = np.min(errores[-ventana_estancamiento:])
            error_anterior_min = np.min(errores[-(2*ventana_estancamiento):-ventana_estancamiento])
            if error_anterior_min > 0:
                mejora_reciente = (error_anterior_min - error_reciente_min) / error_anterior_min * 100
            else:
                mejora_reciente = 0
            
            estancado = mejora_reciente < 1.0  # Menos del 1% de mejora en la última ventana
        else:
            mejora_reciente = 0
            estancado = False
        
        return {
            'error_inicial': error_inicial,
            'error_final': error_final,
            'error_minimo': error_minimo,
            'error_maximo': error_maximo,
            'error_promedio': error_promedio,
            'error_mediana': error_mediana,
            'error_std': error_std,
            'mejora_porcentual': mejora_porcentual,
            'tasa_convergencia': tasa_convergencia,
            'num_iteraciones': num_iteraciones,
            'mejora_reciente': mejora_reciente,
            'estancado': estancado
        }

# Función para mostrar estadísticas de convergencia
def mostrar_estadisticas_convergencia(estadisticas):
    """
    Muestra estadísticas de convergencia en un formato legible.
    
    Args:
        estadisticas: Diccionario con estadísticas de convergencia
    """
    if not estadisticas:
        print("No hay estadísticas disponibles.")
        return
    
    print("\nEstadísticas de Convergencia:")
    print(f"  • Iteraciones totales: {estadisticas['num_iteraciones']:,}")
    print(f"  • Error inicial: {estadisticas['error_inicial']:.8f}")
    print(f"  • Error final: {estadisticas['error_final']:.8f}")
    print(f"  • Error mínimo: {estadisticas['error_minimo']:.8f}")
    print(f"  • Mejora porcentual: {estadisticas['mejora_porcentual']:.2f}%")
    print(f"  • Tasa de convergencia: {estadisticas['tasa_convergencia']:.8f} por iteración")
    
    if estadisticas['estancado']:
        print(f"  • Estado: ESTANCADO (mejora reciente: {estadisticas['mejora_reciente']:.2f}%)")
        print("    Se recomienda ajustar hiperparámetros o aumentar datos de entrenamiento.")
    else:
        print(f"  • Estado: Convergiendo (mejora reciente: {estadisticas['mejora_reciente']:.2f}%)")

# Función principal para usar desde línea de comandos
def main():
    """
    Función principal para usar el gestor de historial desde línea de comandos.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Gestor de historial de entrenamiento')
    parser.add_argument('--listar', action='store_true',
                        help='Listar todos los historiales disponibles')
    parser.add_argument('--analizar', action='store_true',
                        help='Analizar convergencia de un historial')
    parser.add_argument('--comerciante', type=str, default=None,
                        help='ID del comerciante')
    parser.add_argument('--producto', type=str, default=None,
                        help='Tipo de producto')
    parser.add_argument('--archivo', type=str, default=None,
                        help='Archivo específico de historial')
    
    args = parser.parse_args()
    
    # Crear gestor
    gestor = GestorHistorialEntrenamiento()
    
    # Listar historiales
    if args.listar:
        historiales = gestor.listar_historiales()
        if historiales:
            print("\nHistoriales de entrenamiento disponibles:")
            print("-" * 100)
            print(f"{'Comerciante':<15} {'Producto':<15} {'Fecha':<20} {'Iteraciones':<12} {'Tamaño (KB)':<12} {'Archivo':<30}")
            print("-" * 100)
            
            for h in historiales:
                print(f"{h['id_comerciante']:<15} {h['tipo_producto']:<15} {h['fecha']:<20} "
                      f"{h['iteraciones']:<12,} {h['tamaño_kb']:<12.2f} {h['archivo']:<30}")
        return
    
    # Analizar convergencia
    if args.analizar:
        if not args.comerciante and not args.producto and not args.archivo:
            print("Error: Debe especificar comerciante y producto, o un archivo específico.")
            return
        
        # Cargar historial
        errores, _ = gestor.cargar_historial(args.comerciante, args.producto, args.archivo)
        if errores is not None:
            # Analizar convergencia
            estadisticas = gestor.analizar_convergencia(errores)
            mostrar_estadisticas_convergencia(estadisticas)
            
            # Preguntar si desea visualizar
            try:
                import matplotlib.pyplot as plt
                visualizar = input("\n¿Desea visualizar la convergencia? (s/n): ")
                if visualizar.lower() == 's':
                    plt.figure(figsize=(12, 6))
                    
                    # Gráfico de error por iteración
                    plt.subplot(1, 2, 1)
                    plt.plot(errores)
                    plt.title("Error por Iteración")
                    plt.xlabel('Iteración')
                    plt.ylabel('Error (MSE)')
                    plt.grid(True)
                    
                    # Gráfico de error suavizado (media móvil)
                    plt.subplot(1, 2, 2)
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
            except ImportError:
                print("Nota: matplotlib no está instalado. No se puede visualizar la convergencia.")
        return
    
    # Si no se especifica ninguna acción, mostrar ayuda
    parser.print_help()

# Si se ejecuta como script principal
if __name__ == "__main__":
    main()