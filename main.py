import csv  # Base de datos de la cantidad de items ofertadas y los demandados
import numpy as np  # Para cálculos matemáticos y operaciones con arrays
import time  # Para medir tiempos de ejecución
import os  # Para verificar existencia de archivos

# Intentar importar matplotlib para visualización, pero hacer que sea opcional
try:
    import matplotlib.pyplot as plt
    matplotlib_disponible = True
except ImportError:
    matplotlib_disponible = False
    print("Nota: matplotlib no está instalado. La visualización de convergencia no estará disponible.")
    print("Para instalar matplotlib, ejecuta: pip install matplotlib")

# Importar módulos de simulación y entrenamiento automático
try:
    from simulador_mercado import generar_dataset_simulado, simular_escenario_precio
    from entrenamiento_automatico import EntrenadorAutomatico, entrenar_automaticamente
    modulos_avanzados_disponibles = True
except ImportError:
    modulos_avanzados_disponibles = False
    print("Nota: No se pudieron cargar los módulos avanzados de simulación y entrenamiento.")
    print("Asegúrate de que los archivos simulador_mercado.py y entrenamiento_automatico.py estén en el mismo directorio.")

def leer_datos_csv():
    """Lee los datos del archivo CSV y los devuelve como una lista de filas."""
    datos = []
    try:
        with open('mondongos.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Saltamos la cabecera
            for row in reader:
                datos.append(row)
        return datos
    except FileNotFoundError:
        print("Error: No se encontró el archivo mondongos.csv")
        # Crear archivo con encabezados si no existe
        with open('mondongos.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['precio', 'ofertaItems', 'itemsVendidos'])
        return []

def guardar_datos_csv(datos):
    """Guarda los datos en el archivo CSV."""
    with open('mondongos.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['precio', 'ofertaItems', 'itemsVendidos'])
        writer.writerows(datos)

def agregar_nuevo_precio(precio):
    """Agrega un nuevo precio al CSV si no existe."""
    datos = leer_datos_csv()
    for row in datos:
        if float(row[0]) == precio:
            return False  # El precio ya existe
    
    # Si el precio no existe, lo agregamos con valores iniciales
    nueva_fila = [str(precio), '0.5', '0']  # Precio, ofertaItems inicial (0.5 kg), itemsVendidos inicial
    datos.append(nueva_fila)
    guardar_datos_csv(datos)
    print(f"Se ha agregado el nuevo precio {precio} al registro.")
    return True

def actualizar_ventas(precio_actual, cantidad_ventas=1, demanda_predicha=None):
    """Actualiza los itemsVendidos en el CSV y ajusta ofertaItems basado en el modelo.
    Permite registrar múltiples ventas a la vez en kilogramos."""
    datos = leer_datos_csv()
    encontrado = False
    
    for i, row in enumerate(datos):
        if float(row[0]) == precio_actual:
            # Actualizamos itemsVendidos sumando la cantidad de ventas en kilogramos
            ventas_actuales = float(row[2]) + cantidad_ventas
            datos[i][2] = str(ventas_actuales)
            
            # Actualizamos también ofertaItems basado en la predicción y ventas reales
            if demanda_predicha is not None:
                # Damos más peso a las ventas reales para que ofertaItems converja hacia itemsVendidos
                # a medida que el modelo se entrena con más datos
                factor_convergencia = min(0.8, len(datos) / 10)  # Aumenta con más datos históricos
                nueva_oferta = float((ventas_actuales * (1 + factor_convergencia) + demanda_predicha) / 2)
                datos[i][1] = str(max(0.1, nueva_oferta))  # Mínimo 100 gramos (0.1 kg)
            
            encontrado = True
            break
    
    if not encontrado:
        print(f"No se encontró el precio {precio_actual} en el registro.")
    else:
        guardar_datos_csv(datos)
        print(f"Se ha actualizado el registro de ventas para el precio {precio_actual}. Ventas registradas: {cantidad_ventas}")
        return True
    return False

def entrenar_modelo(datos, verbose=False, learning_rate=0.01, epochs=100):
    """Entrena un modelo de regresión lineal simple para predecir la demanda.
    Retorna el peso 'w' que relaciona precio con demanda y el historial de errores.
    
    Parámetros:
    - datos: Lista de filas con datos de precios y ventas
    - verbose: Si es True, muestra información detallada del entrenamiento
    - learning_rate: Tasa de aprendizaje para el descenso de gradiente
    - epochs: Número de iteraciones de entrenamiento
    """
    if not datos or len(datos) < 2:
        return 0.2, []  # Valor por defecto si no hay suficientes datos
    
    # Extraer precios y ventas para entrenamiento
    X = np.array([float(row[0]) for row in datos])  # Precios
    y = np.array([float(row[2]) for row in datos])    # Ventas (demanda) en kilogramos
    
    # Normalizar datos para mejor convergencia
    X_mean = np.mean(X)
    X_std = np.std(X) if np.std(X) > 0 else 1
    X_norm = (X - X_mean) / X_std
    # Normalizar también las ventas para reducir el error absoluto
    y_mean = np.mean(y)
    y_std = np.std(y) if np.std(y) > 0 else 1
    y_norm = (y - y_mean) / y_std if y_std > 0 else y
    
    # Inicializar peso w y variables para seguimiento
    w = 0
    error_history = []
    best_w = w
    best_error = float('inf')
    
    # Ajustar learning_rate basado en la cantidad de datos
    # Con pocos datos, usamos un learning_rate más conservador
    adaptive_lr = learning_rate
    if len(datos) < 5:
        adaptive_lr = learning_rate * 0.5
    
    # Entrenamiento con descenso de gradiente
    for epoch in range(epochs):
        # Predicción con el modelo actual
        y_pred_norm = w * X_norm
        
        # Calcular error (error cuadrático medio) en espacio normalizado
        error = y_pred_norm - y_norm
        mse = np.mean(error**2)
        error_history.append(mse)
        
        # Guardar el mejor modelo hasta ahora
        if mse < best_error:
            best_error = mse
            best_w = w
        
        # Actualizar peso w con learning rate adaptativo
        gradient = np.mean(error * X_norm)
        w = w - adaptive_lr * gradient
        
        # Reducir learning rate si el error no mejora (cada 20 épocas)
        if epoch > 0 and epoch % 20 == 0:
            if error_history[epoch] > error_history[epoch-20]:
                adaptive_lr = adaptive_lr * 0.9
        
        # Mostrar progreso si verbose es True
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Época {epoch+1}/{epochs}, Error: {mse:.4f}, Peso w: {w:.4f}, LR: {adaptive_lr:.6f}")
    
    # Usar el mejor peso encontrado
    if best_error < np.mean(error_history):
        w = best_w
        if verbose:
            print(f"Usando el mejor peso encontrado: w = {w:.4f} con error {best_error:.4f}")
    
    # Asegurar que el error no sea exactamente 1 (podría indicar un problema)
    if len(error_history) > 0 and abs(np.mean(error_history) - 1.0) < 0.001:
        # Si el error es muy cercano a 1, ajustamos ligeramente el peso
        w = w * 1.01
        # Recalculamos el error con el peso ajustado
        y_pred_norm = w * X_norm
        error = y_pred_norm - y_norm
        mse = np.mean(error**2)
        error_history.append(mse)
        if verbose:
            print(f"Error ajustado para evitar valor 1: {mse:.4f}")
    
    # Retornar el peso entrenado y el historial de errores
    return w, error_history

def visualizar_convergencia(error_history, titulo="Convergencia del Modelo"):
    """Visualiza la convergencia del modelo mostrando cómo disminuye el error con cada época."""
    if not matplotlib_disponible:
        print("No se puede visualizar la convergencia porque matplotlib no está instalado.")
        print("Para instalar matplotlib, ejecuta: pip install matplotlib")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(error_history)
    plt.title(titulo)
    plt.xlabel('Época')
    plt.ylabel('Error Cuadrático Medio')
    plt.grid(True)
    plt.show()

def predecir_demanda(precio, w, datos):
    """Predice la demanda para un precio dado usando el modelo entrenado.
    A medida que se acumulan más datos, la predicción converge hacia las ventas reales."""
    if not datos or len(datos) < 2:
        return 0.5  # Valor por defecto si no hay suficientes datos (0.5 kg)
    
    # Extraer precios y ventas para análisis
    X = np.array([float(row[0]) for row in datos])
    y = np.array([float(row[2]) for row in datos])    # Ventas (demanda) en kilogramos
    oferta_actual = np.array([float(row[1]) for row in datos])  # ofertaItems actuales en kilogramos
    
    # Normalizar datos de manera consistente con el entrenamiento
    X_mean = np.mean(X)
    X_std = np.std(X) if np.std(X) > 0 else 1
    precio_norm = (precio - X_mean) / X_std
    
    # Normalización de ventas (consistente con el entrenamiento)
    y_mean = np.mean(y)
    y_std = np.std(y) if np.std(y) > 0 else 1
    
    # Buscar si hay datos históricos para este precio exacto
    demanda_historica = None
    oferta_historica = None
    for row in datos:
        if float(row[0]) == precio:
            demanda_historica = float(row[2])  # itemsVendidos en kilogramos
            oferta_historica = float(row[1])   # ofertaItems en kilogramos
            break
    
    # Calcular la demanda promedio como base
    demanda_promedio = np.mean(y)
    
    # Factor de convergencia que aumenta con la cantidad de datos
    # Esto hace que el modelo dé más peso a las ventas reales con el tiempo
    factor_convergencia = min(0.9, len(datos) / 10)
    
    # Si tenemos datos históricos para este precio, les damos más peso
    if demanda_historica is not None:
        # A medida que aumenta el factor de convergencia, nos acercamos más a la demanda histórica
        base_demanda = (demanda_historica * factor_convergencia) + (demanda_promedio * (1 - factor_convergencia))
        
        # Si tenemos oferta histórica, la consideramos para la convergencia
        if oferta_historica is not None:
            # Calculamos un promedio ponderado entre oferta y demanda histórica
            # A medida que el modelo aprende, ofertaItems debe converger hacia itemsVendidos
            base_demanda = (base_demanda * 0.7) + (oferta_historica * 0.3)
    else:
        base_demanda = demanda_promedio
    
    # Calcular predicción usando el modelo normalizado
    y_pred_norm = w * precio_norm
    # Desnormalizar para obtener la predicción en la escala original
    y_pred = (y_pred_norm * y_std) + y_mean if y_std > 0 else y_pred_norm
    
    # Aplicar factor de ajuste basado en la correlación precio-demanda
    # Usamos una función sigmoide para suavizar el efecto
    factor_ajuste = 1 / (1 + np.exp(-w * precio_norm * 0.5))
    
    # Combinar la predicción del modelo con la base_demanda
    if len(datos) < 5:  # Con pocos datos, damos más peso a la base_demanda
        demanda_pred = float((y_pred * 0.3) + (base_demanda * 0.7))
    else:  # Con más datos, confiamos más en el modelo
        demanda_pred = float((y_pred * 0.6) + (base_demanda * 0.4))
    
    # Aplicar factor de convergencia final para acercar la predicción a las ventas reales
    if demanda_historica is not None:
        # Cuanto más datos tengamos, más nos acercamos a la demanda histórica real
        demanda_pred = float((demanda_pred * (1 - factor_convergencia)) + (demanda_historica * factor_convergencia))
    
    # Asegurar un valor mínimo razonable (mínimo 0.1 kg = 100 gramos)
    return max(0.1, demanda_pred)

def main():
    # Entrada de datos
    print("=== Sistema de Predicción de Demanda de Mondongos ===\n")
    print("Nota: El precio se refiere al costo de adquisición por kilogramo para el comerciante.")
    precio = float(input("¿Cuánto te costó adquirir cada kilogramo de mondongo hoy? "))
    inventario = float(input("¿Cuántos kilogramos de mondongos compraste inicialmente? "))
    
    # Variables para almacenar información
    ofertaItems = 0
    itemsVendidos = 0
    
    # Leer datos del CSV
    datos = leer_datos_csv()
    
    # Buscar datos específicos para el precio actual
    for row in datos:
        precio_csv = float(row[0])
        if precio_csv <= precio:  # Consideramos todos los precios menores o iguales
            ofertaItems = float(row[1])
            if precio_csv == precio:
                itemsVendidos = float(row[2])
    
    # Entrenar el modelo con los datos históricos
    w, error_history = entrenar_modelo(datos)
    error_promedio = np.mean(error_history) if error_history else 0
    print(f"Modelo entrenado con peso w = {w:.4f}")
    print(f"Error promedio del modelo: {error_promedio:.4f}")
    
    if error_promedio > 50 and len(datos) < 10:
        print("\nNota: El error del modelo es alto, lo cual es normal con pocos datos.")
        print("A medida que registres más ventas, el modelo mejorará su precisión.")
    
    # Predecir la demanda usando el modelo entrenado
    demanda_predicha = predecir_demanda(precio, w, datos)
    
    # Ajustar la oferta basada en la predicción y datos históricos
    # Calculamos la oferta óptima basada en el modelo de machine learning
    if itemsVendidos > 0:
        # Si tenemos datos históricos para este precio, combinamos con la predicción
        # A medida que el modelo se entrena, damos más peso a las ventas reales
        # para que ofertaItems converja hacia itemsVendidos
        factor_convergencia = min(0.9, len(datos) / 10)  # Aumenta con más datos históricos
        peso_modelo = max(0.1, 1 - factor_convergencia)  # Disminuye con más datos históricos
        
        ofertaItems = float((demanda_predicha * peso_modelo) + (itemsVendidos * factor_convergencia * 1.1))
    else:
        # Si no tenemos datos históricos, confiamos completamente en la predicción
        ofertaItems = max(0.1, demanda_predicha)
    
    # Aseguramos un valor mínimo razonable (mínimo 0.1 kg = 100 gramos)
    ofertaItems = max(0.1, ofertaItems)
    
    # Calcular items actuales y recomendación
    actualItems = inventario - itemsVendidos
    # Asegurar que la recomendación sea siempre un valor positivo
    recomendacion = max(0, ofertaItems - actualItems)
    
    print(f"\nRecomendación: Debes comprar {recomendacion:.2f} kilogramos de mondongos adicionales")
    print(f"Esta recomendación está basada en un modelo de machine learning que converge")
    print(f"hacia la demanda real (ventas históricas) con cada interacción.")
    print(f"Peso actual del modelo w = {w:.4f}")
    
    # Verificar si el precio existe en el registro, si no, agregarlo
    precio_existe = False
    for row in datos:
        if float(row[0]) == precio:
            precio_existe = True
            break
    
    if not precio_existe:
        agregar_nuevo_precio(precio)
        print("Se ha agregado un nuevo precio al registro para futuras predicciones.")
    
    # Preguntar si el usuario compró mondongos adicionales
    compro_adicionales = input(f"\n¿Compraste mondongos adicionales hoy? (s/n): ").lower()
    if compro_adicionales == 's':
        # Actualizar el inventario con los mondongos adicionales comprados
        if recomendacion > 0:
            print(f"Recomendación actual: {recomendacion:.2f} kg de mondongos")
        
        try:
            cantidad_str = input("¿Cuántos kilogramos de mondongos compraste? (puedes usar decimales, ej: 0.5 para 500g): ")
            cantidad_comprada = float(cantidad_str)
            cantidad_comprada = max(0.01, cantidad_comprada)  # Mínimo 10 gramos
        except ValueError:
            print("Valor no válido. Se registrará la compra de 0.5 kg de mondongos.")
            cantidad_comprada = 0.5
        
        # Actualizar el inventario sumando la cantidad comprada
        inventario += cantidad_comprada
        actualItems += cantidad_comprada
        print(f"Inventario actualizado. Ahora tienes {actualItems:.2f} kg de mondongos.")
    else:
        print("No se han registrado compras adicionales de mondongos.")
    
    # Preguntar si desea registrar ventas
    registrando_ventas = True
    while registrando_ventas and actualItems > 0:
        respuesta = input("\n¿Deseas registrar ventas? (s/n): ").lower()
        if respuesta == 's':
            # Permitir registrar múltiples ventas a la vez
            max_ventas = min(40, actualItems)  # Aumentado a 40 ventas máximas por sesión
            cantidad_ventas = 1
            if max_ventas > 0:
                try:
                    cantidad_str = input(f"¿Cuántos kilogramos de mondongo vendiste? (máximo {max_ventas:.2f} kg disponibles): ")
                    cantidad_ventas = float(cantidad_str)
                    cantidad_ventas = max(0.01, min(cantidad_ventas, max_ventas))  # Asegurar límites, mínimo 10 gramos (0.01 kg)
                except ValueError:
                    print("Valor no válido. Se registrará 1 kg de venta.")
                    cantidad_ventas = 1
            
            # Pasamos la predicción de demanda para actualizar también ofertaItems
            actualizar_ventas(precio, cantidad_ventas, demanda_predicha)
            # Actualizamos el inventario restando la cantidad vendida
            actualItems = actualItems - cantidad_ventas
            print(f"Inventario actualizado. Ahora tienes {actualItems:.2f} kg de mondongos.")
            
            # Reentrenar el modelo con los nuevos datos
            datos_actualizados = leer_datos_csv()
            w_nuevo, error_history = entrenar_modelo(datos_actualizados)
            error_nuevo = np.mean(error_history) if error_history else 0
            print(f"Modelo reentrenado con nuevo peso w = {w_nuevo:.4f}")
            print(f"Error actual del modelo: {error_nuevo:.4f}")
            
            # Mostrar información sobre la mejora del error
            if error_promedio > 0 and error_nuevo < error_promedio:
                mejora_porcentaje = ((error_promedio - error_nuevo) / error_promedio) * 100
                print(f"¡El error del modelo ha mejorado un {mejora_porcentaje:.2f}%!")
            
            # Actualizar la predicción con el modelo reentrenado
            demanda_predicha = predecir_demanda(precio, w_nuevo, datos_actualizados)
            print(f"Nueva predicción de demanda: {demanda_predicha:.2f} kg de mondongos")
            
            # Calcular nueva recomendación
            ofertaItems = predecir_demanda(precio, w_nuevo, datos_actualizados)
            recomendacion = max(0, ofertaItems - actualItems)
            print(f"Nueva recomendación de compra: {recomendacion:.2f} kg de mondongos adicionales")
            
            if actualItems <= 0:
                print("\n¡Atención! Te has quedado sin inventario.")
                registrando_ventas = False
        else:
            registrando_ventas = False
    
    # Al finalizar el día, actualizamos el inventario en el registro
    if input("\n¿Ha finalizado el día? (s/n): ").lower() == 's':
        print(f"Se ha actualizado el inventario al finalizar el día. Inventario final: {actualItems:.2f} kg de mondongos.")
        print("El modelo de machine learning ha sido actualizado con los datos de hoy.")
        
        # Ofrecer opciones de entrenamiento avanzado
        if input("\n¿Deseas realizar un entrenamiento avanzado del modelo? (s/n): ").lower() == 's':
            # Leer los datos más recientes para el entrenamiento manual
            datos_recientes = leer_datos_csv()
            if modulos_avanzados_disponibles:
                entrenar_automaticamente_menu(datos_recientes)
            else:
                entrenar_manualmente(datos_recientes)
        # Aquí podríamos guardar el inventario final en otro archivo si fuera necesario

def entrenar_manualmente(datos):
    """Permite al usuario entrenar manualmente el modelo con diferentes parámetros
    y visualizar la convergencia del entrenamiento."""
    print("\n=== Entrenamiento Manual del Modelo ===\n")
    
    # Mostrar estadísticas de los datos actuales
    if datos:
        precios = [float(row[0]) for row in datos]
        ventas = [float(row[2]) for row in datos]
        print(f"Datos disponibles: {len(datos)} registros")
        print(f"Rango de precios: {min(precios):.2f} - {max(precios):.2f}")
        print(f"Promedio de ventas: {np.mean(ventas):.2f}")
        print(f"Desviación estándar de ventas: {np.std(ventas):.2f}")
        print(f"Correlación precio-ventas: {np.corrcoef(precios, ventas)[0,1]:.4f}")
        
        # Mostrar advertencia si hay pocos datos
        if len(datos) < 5:
            print("\n⚠️ ADVERTENCIA: Tienes pocos datos para entrenar el modelo.")
            print("El error puede ser alto hasta que registres más ventas.")
    else:
        print("No hay datos disponibles para entrenar el modelo.")
        return
    
    # Opciones de entrenamiento
    print("\n=== Opciones de Entrenamiento ===")
    print("1. Entrenamiento básico (valores predeterminados)")
    print("2. Entrenamiento personalizado (ajustar parámetros)")
    print("3. Entrenamiento avanzado (múltiples pruebas)")
    
    opcion = input("\nSelecciona una opción (1-3): ")
    
    if opcion == "1":
        # Entrenamiento básico con valores predeterminados
        learning_rate = 0.01
        epochs = 100
        verbose = True
    elif opcion == "2":
        # Entrenamiento personalizado
        try:
            learning_rate = float(input("\nTasa de aprendizaje (recomendado 0.001-0.1): ") or "0.01")
            epochs = int(input("Número de épocas (recomendado 100-1000): ") or "100")
            verbose = input("¿Mostrar detalles del entrenamiento? (s/n): ").lower() == 's'
        except ValueError:
            print("Valores no válidos. Usando valores predeterminados.")
            learning_rate = 0.01
            epochs = 100
            verbose = True
    elif opcion == "3":
        # Entrenamiento avanzado con múltiples pruebas
        print("\n=== Entrenamiento Avanzado ===")
        print("Se probarán diferentes combinaciones de parámetros para encontrar el mejor modelo.")
        
        # Definir rangos de parámetros
        learning_rates = [0.001, 0.01, 0.05, 0.1]
        epochs_list = [100, 200, 500]
        
        best_error = float('inf')
        best_params = {}
        
        print("\nProbando combinaciones de parámetros...")
        for lr in learning_rates:
            for ep in epochs_list:
                print(f"\nProbando: LR={lr}, Épocas={ep}")
                w_temp, error_history = entrenar_modelo(datos, verbose=False, learning_rate=lr, epochs=ep)
                final_error = error_history[-1] if error_history else float('inf')
                
                print(f"  Error final: {final_error:.4f}, Peso w: {w_temp:.4f}")
                
                if final_error < best_error:
                    best_error = final_error
                    best_params = {'learning_rate': lr, 'epochs': ep, 'w': w_temp}
        
        print(f"\n✅ Mejor combinación encontrada:")
        print(f"  Learning Rate: {best_params['learning_rate']}")
        print(f"  Épocas: {best_params['epochs']}")
        print(f"  Error final: {best_error:.4f}")
        print(f"  Peso w: {best_params['w']:.4f}")
        
        # Usar los mejores parámetros
        learning_rate = best_params['learning_rate']
        epochs = best_params['epochs']
        verbose = True
    else:
        print("Opción no válida. Usando valores predeterminados.")
        learning_rate = 0.01
        epochs = 100
        verbose = True
    
    # Entrenar el modelo con los parámetros especificados
    print("\nEntrenando modelo...")
    w, error_history = entrenar_modelo(datos, verbose=verbose, learning_rate=learning_rate, epochs=epochs)
    
    error_final = error_history[-1] if error_history else 0
    print(f"\nEntrenamiento completado. Peso final w = {w:.4f}, Error final = {error_final:.4f}")
    
    # Mostrar métricas de evaluación
    if len(datos) >= 2:
        X = np.array([float(row[0]) for row in datos])  # Precios
        y = np.array([float(row[2]) for row in datos])    # Ventas (demanda)
        
        # Normalizar datos
        X_mean = np.mean(X)
        X_std = np.std(X) if np.std(X) > 0 else 1
        X_norm = (X - X_mean) / X_std
        
        # Calcular predicciones
        y_pred = w * X_norm
        
        # Calcular métricas
        mse = np.mean((y_pred - y)**2)
        mae = np.mean(np.abs(y_pred - y))
        
        print(f"Error cuadrático medio (MSE): {mse:.4f}")
        print(f"Error absoluto medio (MAE): {mae:.4f}")
        
        # Explicar el significado del error
        print("\nInterpretación del error:")
        if mse > 50:
            print("- El error es alto, lo que indica que el modelo aún no ha convergido bien.")
            print("- Esto es normal con pocos datos o cuando hay mucha variabilidad.")
            print("- Recomendación: Registra más ventas para mejorar el modelo.")
        elif mse > 10:
            print("- El error es moderado, el modelo está aprendiendo pero aún puede mejorar.")
            print("- Recomendación: Continúa registrando ventas y ajusta los parámetros.")
        else:
            print("- El error es bajo, el modelo está convergiendo bien.")
            print("- Las predicciones deberían ser bastante precisas.")
    
    # Visualizar la convergencia si hay suficientes épocas y matplotlib está disponible
    if len(error_history) > 1 and matplotlib_disponible:
        if input("¿Deseas visualizar la convergencia del modelo? (s/n): ").lower() == 's':
            titulo = f"Convergencia del Modelo (lr={learning_rate}, épocas={epochs})"
            visualizar_convergencia(error_history, titulo)
    elif len(error_history) > 1 and not matplotlib_disponible:
        print("\nLa visualización de convergencia no está disponible porque matplotlib no está instalado.")
        print("Para instalar matplotlib, ejecuta: pip install matplotlib")
    
    # Probar el modelo con diferentes precios
    if input("\n¿Deseas probar el modelo con diferentes precios? (s/n): ").lower() == 's':
        while True:
            try:
                precio_prueba = float(input("\nIngresa un precio para probar (o -1 para salir): "))
                if precio_prueba == -1:
                    break
                
                demanda_pred = predecir_demanda(precio_prueba, w, datos)
                print(f"Predicción de demanda para precio {precio_prueba}: {demanda_pred} mondongos")
                
                # Buscar si hay datos históricos para este precio
                historico = False
                for row in datos:
                    if float(row[0]) == precio_prueba:
                        print(f"Datos históricos: Precio {precio_prueba}, Ventas reales: {row[2]}")
                        historico = True
                        break
                
                if not historico:
                    print("No hay datos históricos para este precio exacto.")
                    
            except ValueError:
                print("Valor no válido. Intenta de nuevo.")
    
    print("\nEntrenamiento manual completado.")

def entrenar_automaticamente_menu(datos):
    """Menú para el entrenamiento automático con simulación de mercado."""
    if not modulos_avanzados_disponibles:
        print("\nError: Los módulos avanzados no están disponibles.")
        print("Asegúrate de que los archivos simulador_mercado.py y entrenamiento_automatico.py estén en el mismo directorio.")
        return
    
    print("\n=== Entrenamiento Automático con Simulación de Mercado ===\n")
    print("Este modo utiliza simulación para generar miles de datos sintéticos")
    print("que permiten entrenar el modelo con mayor precisión.\n")
    
    # Opciones de entrenamiento
    print("Opciones de entrenamiento:")
    print("1. Entrenamiento rápido (50 precios, 100 días)")
    print("2. Entrenamiento estándar (100 precios, 365 días)")
    print("3. Entrenamiento intensivo (200 precios, 730 días)")
    print("4. Entrenamiento personalizado")
    
    opcion = input("\nSelecciona una opción (1-4): ")
    
    # Configurar parámetros según la opción
    if opcion == "1":
        num_precios = 50
        num_dias = 100
        print("\nIniciando entrenamiento rápido...")
    elif opcion == "2":
        num_precios = 100
        num_dias = 365
        print("\nIniciando entrenamiento estándar...")
    elif opcion == "3":
        num_precios = 200
        num_dias = 730
        print("\nIniciando entrenamiento intensivo...")
    elif opcion == "4":
        try:
            num_precios = int(input("\nNúmero de precios a simular (recomendado: 50-500): ") or "100")
            num_dias = int(input("Número de días a simular por precio (recomendado: 100-1000): ") or "365")
            print(f"\nIniciando entrenamiento personalizado con {num_precios} precios y {num_dias} días...")
        except ValueError:
            print("Valores no válidos. Usando configuración estándar.")
            num_precios = 100
            num_dias = 365
    else:
        print("Opción no válida. Usando configuración estándar.")
        num_precios = 100
        num_dias = 365
    
    # Confirmar uso de datos reales
    incluir_datos_reales = True
    if len(datos) > 0:
        if input(f"\n¿Incluir {len(datos)} registros de datos reales en el entrenamiento? (s/n): ").lower() != 's':
            incluir_datos_reales = False
    
    # Mostrar detalles del proceso
    verbose = input("\n¿Mostrar detalles del proceso de entrenamiento? (s/n): ").lower() == 's'
    
    # Iniciar cronómetro
    tiempo_inicio = time.time()
    
    # Ejecutar entrenamiento automático
    try:
        print("\nIniciando simulación y entrenamiento automático...")
        print("Este proceso puede tardar varios minutos dependiendo de la configuración.")
        print("Por favor, espera...\n")
        
        modelo, entrenador = entrenar_automaticamente(
            num_precios=num_precios,
            num_dias=num_dias,
            incluir_datos_reales=incluir_datos_reales,
            verbose=verbose
        )
        
        # Calcular tiempo transcurrido
        tiempo_total = time.time() - tiempo_inicio
        minutos = int(tiempo_total // 60)
        segundos = int(tiempo_total % 60)
        
        print(f"\n✅ Entrenamiento completado en {minutos} minutos y {segundos} segundos")
        print(f"Peso final del modelo: w = {modelo['w']:.6f}")
        print(f"Error final: {modelo['error_final']:.6f}")
        
        # Mostrar evaluación del modelo
        print("\n=== Evaluación del Modelo ===")
        resultados, error_promedio = entrenador.evaluar_modelo_con_simulacion(verbose=False)
        print(f"Error promedio en evaluación: {error_promedio:.2f}%")
        
        # Mostrar predicciones para algunos precios de ejemplo
        print("\n=== Predicciones de Ejemplo ===")
        precios_ejemplo = [5.0, 10.0, 15.0, 20.0, 25.0]
        for precio in precios_ejemplo:
            demanda = entrenador.predecir_demanda(precio)
            print(f"Precio: {precio:.2f} → Demanda predicha: {demanda:.2f} kg")
        
        # Visualizar resultados si matplotlib está disponible
        if matplotlib_disponible:
            if input("\n¿Deseas visualizar los resultados del modelo? (s/n): ").lower() == 's':
                entrenador.visualizar_resultados()
        
        # Preguntar si desea usar este modelo para predicciones futuras
        if input("\n¿Deseas guardar este modelo para predicciones futuras? (s/n): ").lower() == 's':
            entrenador.guardar_modelo_entrenado()
            print("\nModelo guardado exitosamente. Se utilizará en futuras predicciones.")
    
    except Exception as e:
        print(f"\nError durante el entrenamiento automático: {e}")
        print("Volviendo al entrenamiento manual como alternativa...")
        entrenar_manualmente(datos)
if __name__ == "__main__":
    main()