import os
import time
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

# Importar el entrenador masivo
from entrenamiento_masivo import EntrenadorMasivo

def entrenar_modelo_personalizado():
    """
    Función principal para entrenar masivamente un modelo personalizado por comerciante y producto.
    Permite configurar parámetros avanzados para minimizar el error promedio.
    """
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(description='Entrenamiento masivo de modelos personalizados')
    parser.add_argument('--iteraciones', type=int, default=50000, 
                        help='Número de iteraciones para el entrenamiento (recomendado: 50000+)')
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
    parser.add_argument('--modo_silencioso', action='store_true',
                        help='Ejecutar en modo silencioso (menos información en consola)')
    parser.add_argument('--sin_visualizacion', action='store_true',
                        help='Desactivar visualización de gráficos durante el entrenamiento')
    
    args = parser.parse_args()
    
    # Mostrar banner
    print("\n" + "=" * 80)
    print("ENTRENAMIENTO MASIVO DE MODELOS DE PREDICCIÓN DE DEMANDA PERSONALIZADOS")
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
    
    # Confirmar inicio
    confirmacion = input("¿Iniciar entrenamiento con esta configuración? (s/n): ")
    if confirmacion.lower() != 's':
        print("Entrenamiento cancelado.")
        return
    
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
            verbose=not args.modo_silencioso,
            guardar_cada=args.guardar_cada,
            visualizar_progreso=not args.sin_visualizacion
        )
        
        # Mostrar resultados
        tiempo_total = time.time() - tiempo_inicio
        print("\n" + "=" * 80)
        print("RESULTADOS DEL ENTRENAMIENTO")
        print("=" * 80)
        print(f"\nEntrenamiento completado en {tiempo_total/60:.2f} minutos")
        print(f"Iteraciones totales acumuladas: {entrenador.configuracion['iteraciones_totales']:,}")
        print(f"Error final: {entrenador.configuracion['mejor_error']:.8f}")
        print(f"Modelo guardado en: {entrenador._obtener_ruta_modelo()}")
        
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
        if not args.sin_visualizacion:
            entrenador.visualizar_curva_demanda()
        
        # Mostrar información sobre cómo usar el modelo
        print("\n" + "-" * 80)
        print("CÓMO USAR EL MODELO ENTRENADO:")
        print(f"1. Importar el modelo en tu código:")
        print(f"   from entrenamiento_masivo import EntrenadorMasivo")
        print(f"   entrenador = EntrenadorMasivo(id_comerciante='{args.comerciante if args.comerciante else 'general'}', tipo_producto='{args.producto if args.producto else 'mondongo'}')")
        print(f"   # Cargar modelo entrenado")
        print(f"   entrenador.mejor_modelo = entrenador._cargar_modelo()")
        print(f"\n2. Predecir demanda para un precio específico:")
        print(f"   demanda = entrenador.predecir_demanda(precio=15.0)  # Cambia el precio según necesites")
        print(f"\n3. Obtener recomendación de precio óptimo:")
        print(f"   recomendacion = entrenador.recomendar_precio_optimo()")
        print("-" * 80)
        
        return entrenador, recomendacion
        
    except KeyboardInterrupt:
        tiempo_total = time.time() - tiempo_inicio
        print("\n\nEntrenamiento interrumpido por el usuario.")
        print(f"Tiempo transcurrido: {tiempo_total/60:.2f} minutos")
        print(f"Iteraciones completadas: {entrenador.configuracion['iteraciones_totales']:,}")
        print(f"Mejor error alcanzado: {entrenador.configuracion['mejor_error']:.8f}")
        return entrenador, None
    except Exception as e:
        print(f"\n\nError durante el entrenamiento: {e}")
        return None, None

def visualizar_comparacion_modelos(entrenadores, nombres=None):
    """
    Visualiza una comparación entre múltiples modelos entrenados.
    
    Args:
        entrenadores: Lista de objetos EntrenadorMasivo con modelos entrenados
        nombres: Lista de nombres para identificar cada modelo en la leyenda
    """
    if not plt:
        print("Error: matplotlib no está instalado. No se puede visualizar la comparación.")
        return
    
    if nombres is None:
        nombres = [f"Modelo {i+1}" for i in range(len(entrenadores))]
    
    # Crear rango de precios común
    rango_precios = np.linspace(5, 25, 100)  # 100 puntos entre 5 y 25
    
    # Crear gráfico
    plt.figure(figsize=(12, 8))
    
    # Curvas de demanda
    plt.subplot(2, 1, 1)
    for i, entrenador in enumerate(entrenadores):
        if entrenador.mejor_modelo is not None:
            demandas = [entrenador.predecir_demanda(precio) for precio in rango_precios]
            plt.plot(rango_precios, demandas, label=nombres[i])
    
    plt.title("Comparación de Curvas de Demanda")
    plt.xlabel("Precio ($)")
    plt.ylabel("Demanda (kg)")
    plt.grid(True)
    plt.legend()
    
    # Curvas de ganancia
    plt.subplot(2, 1, 2)
    for i, entrenador in enumerate(entrenadores):
        if entrenador.mejor_modelo is not None:
            ganancias = []
            for precio in rango_precios:
                demanda = entrenador.predecir_demanda(precio)
                ganancia = (precio - precio * 0.6) * demanda  # Asumiendo costo = 60% del precio
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

def entrenar_multiples_productos(productos, iteraciones=10000, comerciante=None):
    """
    Entrena modelos para múltiples productos del mismo comerciante.
    
    Args:
        productos: Lista de nombres de productos
        iteraciones: Número de iteraciones por producto
        comerciante: ID del comerciante
    
    Returns:
        Diccionario con los entrenadores para cada producto
    """
    entrenadores = {}
    
    for producto in productos:
        print(f"\n{'='*80}")
        print(f"ENTRENANDO MODELO PARA: {producto.upper()}")
        print(f"{'='*80}\n")
        
        entrenador = EntrenadorMasivo(
            id_comerciante=comerciante,
            tipo_producto=producto
        )
        
        # Entrenar
        resultado = entrenador.entrenar_masivamente(
            num_iteraciones=iteraciones,
            verbose=True,
            guardar_cada=1000,
            visualizar_progreso=False  # No visualizar durante el entrenamiento
        )
        
        entrenadores[producto] = entrenador
        
        # Mostrar resultado
        print(f"\nModelo para {producto} entrenado con error: {entrenador.configuracion['mejor_error']:.6f}")
    
    # Comparar modelos
    if len(entrenadores) > 1:
        visualizar_comparacion_modelos(
            list(entrenadores.values()),
            nombres=list(entrenadores.keys())
        )
    
    return entrenadores

# Si se ejecuta como script principal
if __name__ == "__main__":
    entrenar_modelo_personalizado()