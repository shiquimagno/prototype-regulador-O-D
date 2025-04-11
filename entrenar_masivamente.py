import os
import sys
import time
import numpy as np
from datetime import datetime

# Importar módulos necesarios
from entrenamiento_masivo import EntrenadorMasivo, entrenar_masivamente

# Verificar si matplotlib está disponible
try:
    import matplotlib.pyplot as plt
    matplotlib_disponible = True
except ImportError:
    matplotlib_disponible = False
    print("Nota: matplotlib no está instalado. La visualización no estará disponible.")
    print("Para instalar matplotlib, ejecuta: pip install matplotlib")

def mostrar_menu_principal():
    """
    Muestra el menú principal de la aplicación.
    """
    print("\n" + "=" * 80)
    print("SISTEMA DE ENTRENAMIENTO MASIVO DE MODELOS DE PREDICCIÓN DE DEMANDA")
    print("=" * 80)
    print("\nOpciones disponibles:")
    print("  1. Entrenar modelo personalizado (configuración básica)")
    print("  2. Entrenar modelo avanzado (configuración detallada)")
    print("  3. Visualizar modelos entrenados")
    print("  4. Comparar modelos")
    print("  5. Entrenar múltiples productos")
    print("  0. Salir")
    
    opcion = input("\nSeleccione una opción (0-5): ")
    return opcion

def entrenar_modelo_basico():
    """
    Entrena un modelo con configuración básica.
    """
    print("\n" + "=" * 80)
    print("ENTRENAMIENTO BÁSICO DE MODELO PERSONALIZADO")
    print("=" * 80 + "\n")
    
    # Solicitar parámetros básicos
    id_comerciante = input("ID del comerciante (Enter para 'general'): ") or None
    tipo_producto = input("Tipo de producto (Enter para 'mondongo'): ") or None
    num_iteraciones = int(input("Número de iteraciones (recomendado 10000+): ") or "10000")
    
    # Crear directorio para guardar historial de errores
    directorio_historial = 'historiales_error'
    if not os.path.exists(directorio_historial):
        os.makedirs(directorio_historial)
    
    # Nombre de archivo para guardar historial de errores
    id_comerciante_str = id_comerciante if id_comerciante else 'general'
    tipo_producto_str = tipo_producto if tipo_producto else 'mondongo'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo_historial = os.path.join(
        directorio_historial, 
        f"historial_{id_comerciante_str}_{tipo_producto_str}_{timestamp}.txt"
    )
    
    print(f"\nIniciando entrenamiento con {num_iteraciones:,} iteraciones...")
    print(f"El historial de errores se guardará en: {archivo_historial}")
    
    # Crear entrenador
    entrenador = EntrenadorMasivo(
        id_comerciante=id_comerciante,
        tipo_producto=tipo_producto
    )
    
    # Verificar si hay un modelo previo
    modelo_previo = entrenador._cargar_modelo()
    if modelo_previo:
        print(f"\nModelo previo cargado con error: {entrenador.configuracion['mejor_error']:.6f}")
        print(f"Continuando entrenamiento desde {entrenador.configuracion['iteraciones_totales']:,} iteraciones previas")
    
    # Iniciar entrenamiento
    tiempo_inicio = time.time()
    errores_por_iteracion = []
    
    try:
        # Entrenar masivamente
        resultado = entrenador.entrenar_masivamente(
            num_iteraciones=num_iteraciones,
            verbose=True,
            guardar_cada=1000,
            visualizar_progreso=matplotlib_disponible
        )
        
        # Guardar historial de errores
        errores_por_iteracion = resultado['errores_por_iteracion']
        np.savetxt(archivo_historial, errores_por_iteracion)
        
        # Mostrar resultados
        tiempo_total = time.time() - tiempo_inicio
        print("\n" + "=" * 80)
        print("RESULTADOS DEL ENTRENAMIENTO")
        print("=" * 80)
        print(f"\nEntrenamiento completado en {tiempo_total/60:.2f} minutos")
        print(f"Iteraciones totales acumuladas: {entrenador.configuracion['iteraciones_totales']:,}")
        print(f"Error final: {entrenador.configuracion['mejor_error']:.8f}")
        print(f"Modelo guardado en: {entrenador._obtener_ruta_modelo()}")
        print(f"Historial de errores guardado en: {archivo_historial}")
        
        # Obtener recomendación de precio óptimo
        recomendacion = entrenador.recomendar_precio_optimo()
        if recomendacion:
            print("\nRecomendación de precio óptimo:")
            print(f"  • Precio: ${recomendacion['precio_optimo']:.2f}")
            print(f"  • Demanda esperada: {recomendacion['demanda_esperada']:.2f} kg")
            print(f"  • Ganancia esperada: ${recomendacion['ganancia_esperada']:.2f}")
        
        # Visualizar curva de demanda final
        if matplotlib_disponible:
            entrenador.visualizar_curva_demanda()
        
        return entrenador, recomendacion
        
    except KeyboardInterrupt:
        tiempo_total = time.time() - tiempo_inicio
        print("\n\nEntrenamiento interrumpido por el usuario.")
        print(f"Tiempo transcurrido: {tiempo_total/60:.2f} minutos")
        print(f"Iteraciones completadas: {entrenador.configuracion['iteraciones_totales']:,}")
        print(f"Mejor error alcanzado: {entrenador.configuracion['mejor_error']:.8f}")
        
        # Guardar historial parcial
        if errores_por_iteracion:
            np.savetxt(archivo_historial, errores_por_iteracion)
            print(f"Historial parcial de errores guardado en: {archivo_historial}")
        
        return entrenador, None
    except Exception as e:
        print(f"\n\nError durante el entrenamiento: {e}")
        return None, None

def visualizar_modelos():
    """
    Visualiza los modelos entrenados disponibles.
    """
    # Importar el visualizador
    try:
        from visualizador_convergencia import VisualizadorConvergencia
        visualizador = VisualizadorConvergencia()
        visualizador.listar_modelos()
        
        # Preguntar si desea visualizar algún modelo específico
        visualizar = input("\n¿Desea visualizar algún modelo específico? (s/n): ")
        if visualizar.lower() == 's':
            id_comerciante = input("ID del comerciante: ")
            tipo_producto = input("Tipo de producto: ")
            
            # Buscar historial de errores
            directorio_historial = 'historiales_error'
            archivo_historial = None
            if os.path.exists(directorio_historial):
                archivos = [f for f in os.listdir(directorio_historial) 
                           if f.startswith(f"historial_{id_comerciante}_{tipo_producto}")]
                if archivos:
                    # Usar el archivo más reciente
                    archivos.sort(reverse=True)
                    archivo_historial = os.path.join(directorio_historial, archivos[0])
                    print(f"Usando historial de errores: {archivo_historial}")
            
            visualizador.visualizar_convergencia(id_comerciante, tipo_producto, archivo_historial)
    except ImportError:
        print("Error: No se pudo importar el visualizador de convergencia.")
        print("Asegúrate de que el archivo visualizador_convergencia.py esté en el mismo directorio.")

def comparar_modelos():
    """
    Compara múltiples modelos entrenados.
    """
    # Importar el visualizador
    try:
        from visualizador_convergencia import VisualizadorConvergencia
        visualizador = VisualizadorConvergencia()
        visualizador.listar_modelos()
        
        print("\nComparación de modelos:")
        print("Ingrese los modelos a comparar (deje en blanco para terminar)")
        
        modelos_ids = []
        while True:
            id_comerciante = input("\nID del comerciante (Enter para terminar): ")
            if not id_comerciante:
                break
            
            tipo_producto = input("Tipo de producto: ")
            modelos_ids.append((id_comerciante, tipo_producto))
        
        if modelos_ids:
            visualizador.comparar_modelos(modelos_ids)
        else:
            print("No se seleccionaron modelos para comparar.")
    except ImportError:
        print("Error: No se pudo importar el visualizador de convergencia.")
        print("Asegúrate de que el archivo visualizador_convergencia.py esté en el mismo directorio.")

def entrenar_multiples_productos():
    """
    Entrena modelos para múltiples productos del mismo comerciante.
    """
    print("\n" + "=" * 80)
    print("ENTRENAMIENTO DE MÚLTIPLES PRODUCTOS")
    print("=" * 80 + "\n")
    
    # Solicitar parámetros
    id_comerciante = input("ID del comerciante (Enter para 'general'): ") or 'general'
    
    productos = []
    print("\nIngrese los tipos de productos a entrenar (deje en blanco para terminar):")
    while True:
        producto = input("Tipo de producto: ")
        if not producto:
            break
        productos.append(producto)
    
    if not productos:
        print("No se ingresaron productos para entrenar.")
        return
    
    num_iteraciones = int(input("\nNúmero de iteraciones por producto (recomendado 5000+): ") or "5000")
    
    # Crear directorio para guardar historial de errores
    directorio_historial = 'historiales_error'
    if not os.path.exists(directorio_historial):
        os.makedirs(directorio_historial)
    
    # Entrenar cada producto
    entrenadores = {}
    for producto in productos:
        print(f"\n{'='*80}")
        print(f"ENTRENANDO MODELO PARA: {producto.upper()}")
        print(f"{'='*80}\n")
        
        # Nombre de archivo para guardar historial de errores
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo_historial = os.path.join(
            directorio_historial, 
            f"historial_{id_comerciante}_{producto}_{timestamp}.txt"
        )
        
        # Crear entrenador
        entrenador = EntrenadorMasivo(
            id_comerciante=id_comerciante,
            tipo_producto=producto
        )
        
        # Verificar si hay un modelo previo
        modelo_previo = entrenador._cargar_modelo()
        if modelo_previo:
            print(f"\nModelo previo cargado con error: {entrenador.configuracion['mejor_error']:.6f}")
            print(f"Continuando entrenamiento desde {entrenador.configuracion['iteraciones_totales']:,} iteraciones previas")
        
        # Entrenar
        try:
            resultado = entrenador.entrenar_masivamente(
                num_iteraciones=num_iteraciones,
                verbose=True,
                guardar_cada=1000,
                visualizar_progreso=False  # No visualizar durante el entrenamiento
            )
            
            # Guardar historial de errores
            errores_por_iteracion = resultado['errores_por_iteracion']
            np.savetxt(archivo_historial, errores_por_iteracion)
            
            entrenadores[producto] = entrenador
            
            # Mostrar resultado
            print(f"\nModelo para {producto} entrenado con error: {entrenador.configuracion['mejor_error']:.6f}")
            print(f"Historial de errores guardado en: {archivo_historial}")
            
        except KeyboardInterrupt:
            print(f"\nEntrenamiento interrumpido para {producto}")
            continue
        except Exception as e:
            print(f"\nError durante el entrenamiento de {producto}: {e}")
            continue
    
    # Comparar modelos
    if len(entrenadores) > 1 and matplotlib_disponible:
        print("\n" + "=" * 80)
        print("COMPARACIÓN DE MODELOS ENTRENADOS")
        print("=" * 80 + "\n")
        
        # Crear gráfico
        plt.figure(figsize=(12, 8))
        
        # Rango de precios común
        rango_precios = np.linspace(5, 25, 100)  # 100 puntos entre 5 y 25
        
        # Curvas de demanda
        plt.subplot(2, 1, 1)
        for producto, entrenador in entrenadores.items():
            demandas = [entrenador.predecir_demanda(precio) for precio in rango_precios]
            plt.plot(rango_precios, demandas, label=producto)
        
        plt.title("Comparación de Curvas de Demanda")
        plt.xlabel("Precio ($)")
        plt.ylabel("Demanda (kg)")
        plt.grid(True)
        plt.legend()
        
        # Curvas de ganancia
        plt.subplot(2, 1, 2)
        for producto, entrenador in entrenadores.items():
            ganancias = []
            for precio in rango_precios:
                demanda = entrenador.predecir_demanda(precio)
                ganancia = (precio - precio * 0.6) * demanda  # Asumiendo costo = 60% del precio
                ganancias.append(ganancia)
            plt.plot(rango_precios, ganancias, label=producto)
            
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
    
    return entrenadores

def main():
    """
    Función principal que ejecuta el menú interactivo.
    """
    while True:
        opcion = mostrar_menu_principal()
        
        if opcion == '0':
            print("\nSaliendo del sistema. ¡Hasta pronto!")
            break
        elif opcion == '1':
            entrenar_modelo_basico()
        elif opcion == '2':
            # Usar el script avanzado
            try:
                from entrenar_modelo_personalizado import entrenar_modelo_personalizado
                entrenar_modelo_personalizado()
            except ImportError:
                print("Error: No se pudo importar el módulo de entrenamiento personalizado.")
                print("Asegúrate de que el archivo entrenar_modelo_personalizado.py esté en el mismo directorio.")
        elif opcion == '3':
            visualizar_modelos()
        elif opcion == '4':
            comparar_modelos()
        elif opcion == '5':
            entrenar_multiples_productos()
        else:
            print("\nOpción no válida. Por favor, seleccione una opción del 0 al 5.")
        
        # Pausa antes de volver al menú principal
        input("\nPresione Enter para continuar...")

# Si se ejecuta como script principal
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrograma interrumpido por el usuario.")
    except Exception as e:
        print(f"\n\nError inesperado: {e}")