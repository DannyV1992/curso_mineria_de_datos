import numpy as np
import matplotlib.pyplot as plt


class MonteCarloSimulator:
    def __init__(self, n=10_000, seed=42):
        # Sección inicial del notebook: np.random.seed(42), n = 10_000
        # seed=None omite el reset para no interferir con el estado aleatorio del notebook
        self.n = n
        if seed is not None:
            np.random.seed(seed)
        self.muestras = {}
        self.resultados = {}

    # ── Sección 2 del notebook: Identificar variables inciertas ──────────────

    def agregar_variable_triangular(self, nombre, low, mode, high):
        # Distribución triangular definida en el dict variables_inciertas del notebook
        self.muestras[nombre] = np.random.triangular(low, mode, high, size=self.n)

    def agregar_variable_discreta(self, nombre, valores, probabilidades):
        # Extensión propia — no existe en el notebook de clase
        # Permite variables con distribución discreta (ej. precios con prob. fijas)
        self.muestras[nombre] = np.random.choice(valores, size=self.n, p=probabilidades)

    def agregar_variable_uniforme(self, nombre, low, high):
        # Extensión propia — no existe en el notebook de clase
        self.muestras[nombre] = np.random.uniform(low, high, size=self.n)

    def agregar_resultado(self, nombre, valores):
        # Almacena un array de resultados simulados (ej. margen, VAN)
        self.resultados[nombre] = np.asarray(valores)

    # ── Sección 3 del notebook: Distribución triangular (gráfico) ────────────

    def plot_distribucion_triangular(self, nombre, low, mode, high):
        # Bloque de visualización de distribución triangular del notebook Clase 11
        x = [low, mode, high]
        y = [0, 1, 0]
        plt.figure(figsize=(7, 4))
        plt.plot(x, y, marker="o")
        plt.fill_between(x, y, alpha=0.3)
        plt.title(f"Distribución triangular de {nombre}")
        plt.xlabel(nombre)
        plt.ylabel("Densidad (esquema)")
        plt.tight_layout()
        plt.show()

    # ── Sección 6 del notebook: Analizar los resultados ──────────────────────

    def resumen(self, nombre):
        # Bloque print() de la Sección 6 del notebook Clase 11
        sim = self.resultados[nombre]
        print(f"=== Resumen: {nombre} ===")
        print(f"  Media:              {sim.mean():>15,.2f}")
        print(f"  Desviación estándar:{sim.std():>15,.2f}")
        print(f"  Mínimo:             {sim.min():>15,.2f}")
        print(f"  Máximo:             {sim.max():>15,.2f}")
        print(f"  Percentil  5%:      {np.percentile(sim, 5):>15,.2f}")
        print(f"  Percentil 50%:      {np.percentile(sim, 50):>15,.2f}")
        print(f"  Percentil 95%:      {np.percentile(sim, 95):>15,.2f}")
        print(f"  P({nombre} < 0):    {(sim < 0).mean():>14.1%}")

    def plot_histograma(self, nombre, xlabel=None, titulo=None):
        # Bloque plt.hist() de la Sección 6 del notebook Clase 11
        sim = self.resultados[nombre]
        plt.figure(figsize=(8, 4))
        plt.hist(sim, bins=40, edgecolor="white")
        plt.axvline(sim.mean(), color="red", label="Media")
        plt.axvline(0, color="black", linestyle="--", label=f"{nombre} = 0")
        plt.title(titulo or f"Distribución de {nombre} (Monte Carlo)")
        plt.xlabel(xlabel or nombre)
        plt.ylabel("Frecuencia")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ── Sección 7 del notebook: Gráficos extras de análisis ──────────────────

    def plot_dispersion_vs_resultado(self, nombre_resultado):
        # Bloque scatter "variable incierta vs margen" del notebook Clase 11
        sim = self.resultados[nombre_resultado]
        idx = np.random.choice(self.n, size=min(2000, self.n), replace=False)
        for nombre_var, valores in self.muestras.items():
            plt.figure(figsize=(7, 4))
            plt.scatter(valores[idx], sim[idx], alpha=0.3, s=10)
            plt.title(f"{nombre_var} vs {nombre_resultado}")
            plt.xlabel(nombre_var)
            plt.ylabel(nombre_resultado)
            plt.tight_layout()
            plt.show()

    def plot_correlacion(self, nombre_resultado):
        # Bloque barh de correlaciones del notebook Clase 11
        sim = self.resultados[nombre_resultado]
        nombres = list(self.muestras.keys())
        correlaciones = [
            np.corrcoef(self.muestras[n], sim)[0, 1] for n in nombres
        ]
        print(f"Correlación con {nombre_resultado}:")
        for n, c in zip(nombres, correlaciones):
            print(f"  {n}: {c:.3f}")
        plt.figure(figsize=(7, 4))
        plt.barh(nombres, correlaciones)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlim(-1, 1)
        plt.title(f"Correlación de cada variable incierta con {nombre_resultado}")
        plt.xlabel("Correlación de Pearson")
        plt.tight_layout()
        plt.show()

    def plot_impacto_por_rangos(self, nombre_resultado):
        # Bloque "Impacto por rangos" del notebook Clase 11
        sim = self.resultados[nombre_resultado]
        for nombre_var, valores in self.muestras.items():
            p33, p66 = np.percentile(valores, [33, 66])
            grupos = np.where(valores <= p33, "Bajo",
                     np.where(valores <= p66, "Medio", "Alto"))
            orden = ["Bajo", "Medio", "Alto"]
            medias = [sim[grupos == g].mean() for g in orden]

            print(f"\nImpacto de {nombre_var} sobre {nombre_resultado}:")
            for g, m in zip(orden, medias):
                print(f"  {g}: {m:,.2f}")
            print(f"  Impacto Bajo → Medio: {medias[1] - medias[0]:,.2f}")
            print(f"  Impacto Medio → Alto: {medias[2] - medias[1]:,.2f}")

            plt.figure(figsize=(7, 4))
            plt.bar(orden, medias)
            plt.title(f"{nombre_resultado} promedio por rango de {nombre_var}")
            plt.ylabel(f"{nombre_resultado} promedio")
            plt.tight_layout()
            plt.show()

            datos_box = [sim[grupos == g] for g in orden]
            plt.figure(figsize=(7, 4))
            plt.boxplot(datos_box, labels=orden)
            plt.title(f"Distribución de {nombre_resultado} por rango de {nombre_var}")
            plt.ylabel(nombre_resultado)
            plt.tight_layout()
            plt.show()

    def plot_probabilidad_pct(self, nombre_resultado, nombre_referencia,
                              umbrales=None, xlabel_pct=None):
        # Bloque "¿Qué tan probable es cada nivel de margen %?"
        sim       = self.resultados[nombre_resultado]
        referencia = self.resultados[nombre_referencia]

        # margen_pct = margen_sim / ingresos_sim * 100
        resultado_pct = sim / referencia * 100

        if umbrales is None:
            umbrales = [20, 30, 40, 50, 100]

        print(f"Probabilidades de {nombre_resultado} como % de {nombre_referencia}:\n")
        for u in umbrales:
            p_menor_igual = (resultado_pct <= u).mean()
            p_mayor_igual = (resultado_pct >= u).mean()
            print(f"  P({nombre_resultado} % ≤ {u}%) = {p_menor_igual:.1%}")
            print(f"  P({nombre_resultado} % ≥ {u}%) = {p_mayor_igual:.1%}")
            print()

        # Curva acumulada
        pct_ord   = np.sort(resultado_pct)
        prob_acum = np.arange(1, self.n + 1) / self.n

        umbrales_marca = [u for u in umbrales if u <= pct_ord.max()][:3]

        plt.figure(figsize=(9, 5))
        plt.plot(pct_ord, prob_acum, linewidth=2, label="Probabilidad acumulada")

        for u in umbrales_marca:
            p = (resultado_pct <= u).mean()
            plt.axvline(u, color="gray", linestyle=":", alpha=0.7)
            plt.scatter([u], [p], zorder=5)
            plt.annotate(
                f"P(≤ {u}%) = {p:.0%}",
                xy=(u, p),
                xytext=(u + 1.5, p - 0.08 if p > 0.15 else p + 0.08),
                fontsize=9,
            )

        etiqueta_x = xlabel_pct or f"{nombre_resultado} (% de {nombre_referencia})"
        plt.title(f"Cómo leer el {nombre_resultado} %: probabilidad acumulada")
        plt.xlabel(etiqueta_x)
        plt.ylabel("Probabilidad acumulada")
        plt.ylim(0, 1.05)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

        # Barras: resumen directo de las preguntas
        etiquetas = [f"≤ {u}%" for u in umbrales]
        valores   = [(resultado_pct <= u).mean() for u in umbrales]

        plt.figure(figsize=(8, 4))
        barras = plt.bar(etiquetas, valores)
        plt.ylim(0, 1.15)
        plt.ylabel("Probabilidad")
        plt.title(f"Probabilidad de tener un {nombre_resultado} % menor o igual a…")
        for barra, v in zip(barras, valores):
            plt.text(barra.get_x() + barra.get_width() / 2, v + 0.03,
                     f"{v:.0%}", ha="center", fontsize=10)
        plt.tight_layout()
        plt.show()
    
    def plot_probabilidad_acumulada(self, nombre_resultado, xlabel=None):
        # Bloque "Probabilidad acumulada del margen" del notebook Clase 11
        sim = self.resultados[nombre_resultado]
        ordenado = np.sort(sim)
        prob_acum = np.arange(1, self.n + 1) / self.n

        plt.figure(figsize=(8, 4))
        plt.plot(ordenado, prob_acum)
        plt.axvline(0, color="black", linestyle="--", label=f"{nombre_resultado} = 0")
        plt.axhline(0.05, color="gray", linestyle=":", label="5% y 95%")
        plt.axhline(0.95, color="gray", linestyle=":")
        plt.title(f"Probabilidad acumulada de {nombre_resultado}")
        plt.xlabel(xlabel or nombre_resultado)
        plt.ylabel("Probabilidad acumulada")
        plt.legend()
        plt.tight_layout()
        plt.show()
