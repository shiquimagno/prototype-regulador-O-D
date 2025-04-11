import numpy as np
import random
import math
from datetime import datetime, timedelta

class SimuladorMercado:
    """Clase para simular el comportamiento de un mercado local o supermercado.
    Genera datos sintéticos para entrenar el modelo de predicción de demanda."""
    
    def __init__(self, semilla=None):
        """Inicializa el simulador con parámetros base.
        
        Args:
            semilla: Valor para inicializar el generador de números aleatorios (para reproducibilidad)
        """
        # Establecer semilla para reproducibilidad si se proporciona
        if semilla is not None:
            np.random.seed(semilla)
            random.seed(semilla)
        
        # Parámetros base del mercado
        self.elasticidad_precio = -0.7  # Elasticidad precio-demanda (negativa: a mayor precio, menor demanda)
        self.precio_base = 10.0  # Precio base de referencia
        self.demanda_base = 20.0  # Demanda base en kg a precio base
        
        # Factores de variabilidad
        self.variabilidad_diaria = 0.15  # Variabilidad aleatoria diaria (15%)
        self.variabilidad_estacional = 0.25  # Variabilidad estacional (25%)
        self.tendencia_anual = 0.05  # Tendencia anual de crecimiento/decrecimiento (5%)
        
        # Factores de mercado
        self.dias_semana_factor = {
            0: 0.7,  # Lunes
            1: 0.8,  # Martes
            2: 0.9,  # Miércoles
            3: 1.0,  # Jueves
            4: 1.3,  # Viernes
            5: 1.5,  # Sábado
            6: 0.5   # Domingo
        }
        
        # Factores estacionales por mes (1-12)
        self.mes_factor = {
            1: 0.9,   # Enero
            2: 0.85,  # Febrero
            3: 0.9,   # Marzo
            4: 1.0,   # Abril
            5: 1.05,  # Mayo
            6: 1.1,   # Junio
            7: 1.15,  # Julio
            8: 1.2,   # Agosto
            9: 1.1,   # Septiembre
            10: 1.0,  # Octubre
            11: 0.95, # Noviembre
            12: 1.2   # Diciembre (fiestas)
        }
        
        # Eventos especiales (fechas con alta demanda)
        self.eventos_especiales = {
            # Formato: "MM-DD": factor_multiplicador
            "12-24": 1.8,  # Nochebuena
            "12-31": 1.7,  # Nochevieja
            "01-01": 0.6,  # Año Nuevo (baja demanda)
            "05-01": 1.3,  # Día del Trabajo
            "05-05": 1.4,  # Cinco de Mayo
            # Añadir más eventos según necesidades
        }
        
        # Parámetros de shock de mercado
        self.prob_shock = 0.01  # Probabilidad diaria de un shock de mercado (1%)
        self.shock_duracion_max = 7  # Duración máxima de un shock en días
        self.shock_actual = 0  # Contador de días restantes del shock actual
        self.shock_factor = 1.0  # Factor multiplicador del shock actual
        
        # Fecha inicial para simulación
        self.fecha_actual = datetime.now()
    
    def _calcular_factor_estacional(self, fecha):
        """Calcula el factor estacional basado en el día de la semana y mes."""
        # Factor del día de la semana
        dia_semana = fecha.weekday()  # 0-6 (lunes-domingo)
        factor_dia = self.dias_semana_factor[dia_semana]
        
        # Factor del mes
        mes = fecha.month  # 1-12
        factor_mes = self.mes_factor[mes]
        
        # Verificar si es un evento especial
        fecha_str = fecha.strftime("%m-%d")
        factor_evento = self.eventos_especiales.get(fecha_str, 1.0)
        
        # Combinar factores
        return factor_dia * factor_mes * factor_evento
    
    def _aplicar_tendencia(self, fecha):
        """Aplica una tendencia a largo plazo basada en la fecha."""
        # Calcular años desde la fecha inicial
        dias_diff = (fecha - self.fecha_actual).days
        años_diff = dias_diff / 365.0
        
        # Aplicar tendencia exponencial
        return math.pow(1 + self.tendencia_anual, años_diff)
    
    def _actualizar_shock(self):
        """Actualiza el estado de los shocks de mercado."""
        if self.shock_actual > 0:
            # Continuar con el shock actual
            self.shock_actual -= 1
        elif random.random() < self.prob_shock:
            # Iniciar un nuevo shock
            self.shock_actual = random.randint(1, self.shock_duracion_max)
            # El shock puede ser positivo (aumento de demanda) o negativo (disminución)
            shock_magnitud = random.uniform(0.6, 1.8)
            if random.random() < 0.5:  # 50% probabilidad de shock negativo
                shock_magnitud = 1.0 / shock_magnitud
            self.shock_factor = shock_magnitud
        else:
            # Sin shock
            self.shock_factor = 1.0
    
    def generar_demanda(self, precio, fecha=None):
        """Genera una demanda simulada para un precio dado en una fecha específica.
        
        Args:
            precio: Precio del producto en la fecha dada
            fecha: Fecha para la simulación (datetime). Si es None, usa la fecha actual.
            
        Returns:
            Demanda simulada en kilogramos
        """
        if fecha is None:
            fecha = self.fecha_actual
        
        # Actualizar estado de shocks
        self._actualizar_shock()
        
        # Calcular elasticidad precio-demanda
        # Fórmula: demanda = demanda_base * (precio/precio_base)^elasticidad
        factor_precio = math.pow(precio / self.precio_base, self.elasticidad_precio)
        
        # Aplicar factores estacionales
        factor_estacional = self._calcular_factor_estacional(fecha)
        
        # Aplicar tendencia a largo plazo
        factor_tendencia = self._aplicar_tendencia(fecha)
        
        # Calcular demanda base ajustada
        demanda_ajustada = self.demanda_base * factor_precio * factor_estacional * factor_tendencia * self.shock_factor
        
        # Añadir variabilidad aleatoria diaria
        ruido = np.random.normal(1.0, self.variabilidad_diaria)
        demanda_final = demanda_ajustada * ruido
        
        # Asegurar que la demanda no sea negativa y redondear a 2 decimales
        return round(max(0.1, demanda_final), 2)
    
    def generar_datos_entrenamiento(self, rango_precios, num_dias=365, fecha_inicio=None):
        """Genera un conjunto de datos sintéticos para entrenar el modelo.
        
        Args:
            rango_precios: Lista de precios para generar datos
            num_dias: Número de días a simular
            fecha_inicio: Fecha de inicio de la simulación (datetime). Si es None, usa la fecha actual.
            
        Returns:
            Lista de filas con formato [precio, ofertaItems, itemsVendidos]
        """
        if fecha_inicio is None:
            fecha_inicio = self.fecha_actual
        
        datos = []
        
        # Para cada precio en el rango
        for precio in rango_precios:
            # Simular varios días para este precio
            fecha_sim = fecha_inicio
            ventas_acumuladas = 0
            oferta_acumulada = 0
            
            for _ in range(num_dias):
                # Generar demanda para este día
                demanda = self.generar_demanda(precio, fecha_sim)
                ventas_acumuladas += demanda
                
                # La oferta inicial es un poco mayor que la demanda (simulando predicción)
                oferta = demanda * random.uniform(0.9, 1.2)
                oferta_acumulada += oferta
                
                # Avanzar al siguiente día
                fecha_sim += timedelta(days=1)
            
            # Calcular promedios
            ventas_promedio = ventas_acumuladas / num_dias
            oferta_promedio = oferta_acumulada / num_dias
            
            # Añadir ruido final para simular imperfecciones en datos reales
            ventas_final = ventas_promedio * random.uniform(0.95, 1.05)
            oferta_final = oferta_promedio * random.uniform(0.95, 1.05)
            
            # Añadir a los datos de entrenamiento
            datos.append([str(precio), str(oferta_final), str(ventas_final)])
        
        return datos

    def generar_datos_avanzados(self, min_precio=5.0, max_precio=25.0, num_precios=20, num_dias=365):
        """Genera un conjunto de datos avanzado con precios distribuidos y múltiples días.
        
        Args:
            min_precio: Precio mínimo a simular
            max_precio: Precio máximo a simular
            num_precios: Número de precios diferentes a simular
            num_dias: Número de días a simular para cada precio
            
        Returns:
            Lista de filas con formato [precio, ofertaItems, itemsVendidos]
        """
        # Generar rango de precios (distribución logarítmica para más detalle en precios bajos)
        precios_log = np.logspace(np.log10(min_precio), np.log10(max_precio), num_precios)
        precios = [round(p, 2) for p in precios_log]  # Redondear a 2 decimales
        
        return self.generar_datos_entrenamiento(precios, num_dias)

# Funciones de utilidad para usar con el simulador

def generar_dataset_simulado(num_precios=20, num_dias=365, semilla=None):
    """Genera un dataset simulado completo para entrenar el modelo.
    
    Args:
        num_precios: Número de precios diferentes a simular
        num_dias: Número de días a simular para cada precio
        semilla: Semilla para reproducibilidad
        
    Returns:
        Lista de filas con formato [precio, ofertaItems, itemsVendidos]
    """
    simulador = SimuladorMercado(semilla=semilla)
    return simulador.generar_datos_avanzados(num_precios=num_precios, num_dias=num_dias)

def simular_escenario_precio(precio, num_simulaciones=100, semilla=None):
    """Simula la demanda para un precio específico múltiples veces.
    
    Args:
        precio: Precio a simular
        num_simulaciones: Número de simulaciones a realizar
        semilla: Semilla para reproducibilidad
        
    Returns:
        Estadísticas de la demanda simulada (media, mediana, min, max, desviación estándar)
    """
    simulador = SimuladorMercado(semilla=semilla)
    demandas = []
    
    for _ in range(num_simulaciones):
        # Simular una fecha aleatoria en el próximo año
        dias_aleatorios = random.randint(0, 364)
        fecha_sim = simulador.fecha_actual + timedelta(days=dias_aleatorios)
        demanda = simulador.generar_demanda(precio, fecha_sim)
        demandas.append(demanda)
    
    return {
        'media': np.mean(demandas),
        'mediana': np.median(demandas),
        'min': min(demandas),
        'max': max(demandas),
        'desviacion': np.std(demandas)
    }

# Ejemplo de uso:
# datos_simulados = generar_dataset_simulado(num_precios=20, num_dias=365)
# estadisticas_precio_10 = simular_escenario_precio(10.0, num_simulaciones=100)