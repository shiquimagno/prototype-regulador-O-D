#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script ejecutable para iniciar el entrenamiento masivo de modelos de predicción de demanda.
Permite entrenar modelos personalizados por comerciante y tipo de producto con miles de iteraciones.
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Verificar si matplotlib está disponible
try:
    import matplotlib.pyplot as plt
    matplotlib_disponible = True
except ImportError:
    matplotlib_disponible = False
    print("Nota: matplotlib no está instalado. La visualización no estará disponible.")
    print("Para instalar matplotlib, ejecuta: pip install matplotlib")

# Importar el entrenador masivo
try:
    from entrenamiento_masivo import EntrenadorMasivo
except ImportError:
    print("Error: No se pudo importar el módulo EntrenadorMasivo.")
    print("Asegúrate de que el archivo entrenamiento_masivo.py esté en el mismo directorio.")
    sys.exit(1)

def main():
    """
    Función principal que procesa los argumentos de línea de comandos y ejecuta el entrenamiento.
    """
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(
        description='Entrenamiento masivo de modelos de predicción de demanda',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--iteraciones', type=int, default=50000, 
                        help='Número de iteraciones para el entrenamiento')
    parser.add_argument('--comerciante', type=str, default=None,
                        help='ID del comerciante para personalizar el modelo')
    parser.add_argument('--producto', type=str, default=None,
                        help='Tipo de producto para personalizar el modelo')
    parser.add_argument('--datos', type=str, default='mondongos.csv',
                        help='Archivo CSV con datos de ventas')
    parser.add_argument('--guardar_cada', type=int, default=1000,
                        help='Guardar el modelo cada N iteraciones')
    parser.add_argument('--datos_simulados', type=int, default=200,
                        help='Número de datos simulados por iteración')
    parser.add_argument('--usar_mercado', action='store_true',
                        help='Usar datos de mercado para mejorar el entrenamiento')
    parser.add_argument('--silencioso', action='store_true',
                        help='Ejecutar en modo silencioso (menos información en consola)')
    parser.add_argument('--sin_visualizacion', action='store_true',
                        help='Desactivar visualización de gráficos durante el entrenamiento')
    parser.add_argument('--menu', action='store_true',
                        help='Mostrar menú interactivo en lugar de iniciar entrenamiento directo')
    
    args = parser.parse_args()
    
    # Si se solicita el menú interactivo, ejecutar el script correspondiente
    if args.menu:
        try:
            from entrenar_masivamente import main as menu_main
            menu_main()
            return
        except ImportError:
            print("Error: No se pudo importar el menú interactivo.")
            print("Asegúrate de que el archivo entrenar_masivamente.py esté en el mismo directorio.")
            sys.exit(1)
    
    # Mostrar banner
    print("\n" + "=" * 80)
    print("ENTRENAMIENTO MASIVO DE MODELOS DE PREDICCIÓN DE DEMANDA")
    print("=" * 80 + "\n")
    
    # Mostrar configuración
    print(f"Configuración del entrenamiento:")
    print(f"  • Iteraciones: {args.iteraciones:,}")
    print(f"  • Comerciante: {args.comerciante if args.comerciante else 'general'}")
    print(f"  • Producto: {args.producto if args.producto else 'mondongo'}")
    print(f"  • Archivo de datos: {args.datos}")
    print(f"  • Datos simulados por iteración: {args.datos_simulados}")
    print(f"  • Usar datos de mercado: {'Sí' if args.usar_mercado else 'No'}")
    print(f"  • Visualización durante entrenamiento: {'No' if args.sin_visualizacion else 'Sí'}")
    print("\n" + "-" * 80 + "\n")
    
    # Crear directorio para guardar historial de errores
    directorio_historial = 'historiales_error'
    if not os.path.exists(directorio_historial):
        os.makedirs(directorio_historial)
    
    # Nombre de archivo para guardar historial de errores
    id_comerciante_str = args.comerciante if args.comerciante else 'general'
    tipo_producto_str = args.producto if args.producto else 'mondongo'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo_historial = os.path.join(
        directorio_historial, 
        f"historial_{id_comerciante_str}_{tipo_producto_str}_{timestamp}.txt"
    )
    
    # Crear entrenador
    entrenador = EntrenadorMasivo(
        archivo_datos=args.datos,
        id_comerciante=args.comerciante,
        tipo_producto=args.producto
    )
    
    # Verificar si hay un modelo previo
    modelo_previo = entrenador._cargar_modelo()
    if modelo_previo:
        print(f"\nModelo previo cargado con error: {entrenador.configuracion['mejor_error']:.6f}")
        print(f"Continuando entrenamiento desde {entrenador.configuracion['iteraciones_totales']:,} iteraciones previas")
    
    # Iniciar entrenamiento
    print(f"\nIniciando entrenamiento masivo con {args.iteraciones:,} iteraciones...\n")
    tiempo_inicio = time.time()
    
    try:
        # Entrenar masivamente
        resultado = entrenador.entrenar_masivamente(
            num_iteraciones=args.iteraciones,
            datos_simulados_por_iteracion=args.datos_simulados,
            usar_datos_mercado=args.usar_mercado,
            verbose=not args.silencioso,
            guardar_cada=args.guardar_cada,
            visualizar_progreso=not args.sin_visualizacion
        )
        
        # Guardar historial de errores
        import numpy as np
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
            print("\nTop 5 precios con mejores ganancias:")
            for i, (precio, demanda, ganancia) in enumerate(recomendacion['top_precios']):
                print(f"  {i+1}. Precio: ${precio:.2f}, Demanda: {demanda:.2f} kg, Ganancia: ${ganancia:.2f}")
        
        # Visualizar curva de demanda final
        if not args.sin_visualizacion and matplotlib_disponible:
            entrenador.visualizar_curva_demanda()
        
        # Analizar convergencia del error
        try:
            from gestor_historial_entrenamiento import GestorHistorialEntrenamiento, mostrar_estadisticas_convergencia
            gestor = GestorHistorialEntrenamiento()
            estadisticas = gestor.analizar_convergencia(errores_por_iteracion)
            print("\n" + "-" * 80)
            print("ANÁLISIS DE CONVERGENCIA DEL ERROR")
            mostrar_estadisticas_convergencia(estadisticas)
        except ImportError:
            pass
        
    except KeyboardInterrupt:
        tiempo_total = time.time() - tiempo_inicio
        print("\n\nEntrenamiento interrumpido por el usuario.")
        print(f"Tiempo transcurrido: {tiempo_total/60:.2f} minutos")
        print(f"Iteraciones completadas: {entrenador.configuracion['iteraciones_totales']:,}")
        print(f"Mejor error alcanzado: {entrenador.configuracion['mejor_error']:.8f}")
    except Exception as e:
        print(f"\n\nError durante el entrenamiento: {e}")

if __name__ == "__main__":
    main()