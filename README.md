# Teoría y Formulación Variacional del Índice de Estabilidad Poblacional (PSI)

El **Índice de Estabilidad Poblacional (PSI)** es una métrica fundamental para cuantificar el _data drift_ (deriva de datos) entre una distribución de referencia $Q$ (ej. datos de entrenamiento) y una distribución actual $P$ (ej. datos en producción).

Tradicionalmente, el PSI se calcula discretizando el espacio en rangos (bins) y comparando frecuencias relativas. Sin embargo, en espacios continuos y de alta dimensión, este enfoque colapsa debido a la maldición de la dimensionalidad. Para resolver esto, recurrimos a la estimación mediante **Redes Neuronales y F-Divergencias**.

---

## 1. El PSI en la Teoría de la Información

Matemáticamente, el PSI no es más que una versión simétrica de la Divergencia de Kullback-Leibler (KL), conocida formalmente como la **Divergencia de Jeffreys**.

Dadas dos distribuciones $P$ y $Q$, el PSI exacto se define como:

$$PSI(P, Q) = D_{KL}(P || Q) + D_{KL}(Q || P)$$

Expresado en su forma integral sobre el dominio $\mathcal{X}$:

$$PSI(P, Q) = \int_{\mathcal{X}} (P(x) - Q(x)) \ln \left( \frac{P(x)}{Q(x)} \right) dx$$

---

## 2. La Formulación Variacional (Dual de Fenchel)

Para estimar esta divergencia a partir de muestras de datos sin conocer las densidades subyacentes exactas, utilizamos la representación variacional basada en el conjugado de Fenchel-Legendre (cota de Donsker-Varadhan / Nguyen-Wainwright-Jordan).

La cota variacional "centrada" (que introduce un $-1$ para reducir la varianza y asegurar que la divergencia sea $0$ cuando $P=Q$) para una divergencia KL individual es:

$$D_{KL}(P || Q) \geq \sup_{T} \left( \mathbb{E}_{x \sim P}[T(x)] - \mathbb{E}_{x \sim Q}[e^{T(x)} - 1] \right)$$

Donde $T(x)$ es una función escalar arbitraria (nuestro "crítico", modelado por una red neuronal). El crítico óptimo que alcanza el supremo es el log-ratio de las densidades: $T^*(x) = \ln\left(\frac{P(x)}{Q(x)}\right)$.

### El Truco Simétrico del "Crítico Único"

Dado que el PSI es la suma de dos divergencias KL en direcciones opuestas, necesitaríamos teóricamente dos críticos ($T_1$ y $T_2$). Sin embargo, aprovechando la propiedad del logaritmo:

$$T_2^*(x) = \ln\left(\frac{Q(x)}{P(x)}\right) = - \ln\left(\frac{P(x)}{Q(x)}\right) = -T_1^*(x)$$

Podemos sustituir $T_2$ por $-T_1$. Llamando simplemente $T$ a nuestra única red neuronal, la cota variacional completa para el PSI se convierte en:

$$PSI(P, Q) \geq \sup_{T} \left( \mathbb{E}_{x \sim P}[T(x) - e^{-T(x)} + 1] - \mathbb{E}_{x \sim Q}[e^{T(x)} + T(x) - 1] \right)$$

Esta ecuación final permite estimar el PSI simétrico entrenando un único modelo.

---

## 3. La Función de Pérdida (Loss Function)

En el aprendizaje automático, optimizamos parámetros $\theta$ de una red neuronal $T_\theta(x)$. Dado que los optimizadores estándar (como Adam o SGD) están diseñados para **minimizar** una función objetivo, y nuestra cota variacional busca un **supremo** (máximo), la Función de Pérdida ($\mathcal{L}$) se define como el negativo de la ecuación variacional.

Aproximando las esperanzas matemáticas mediante el promedio empírico sobre minilotes (mini-batches) de tamaño $N$, la pérdida es:

$$\mathcal{L}(\theta) = - \left( \frac{1}{N} \sum_{x \in P} \left[ T_\theta(x) - e^{-T_\theta(x)} + 1 \right] - \frac{1}{N} \sum_{x \in Q} \left[ e^{T_\theta(x)} + T_\theta(x) - 1 \right] \right)$$

Al minimizar $\mathcal{L}(\theta)$, la red neuronal es esculpida para aproximar el log-ratio real de las distribuciones, y el valor final de $-\mathcal{L}(\theta)$ convergente es nuestra estimación del PSI.

---

## 4. Consideraciones Prácticas y Regularización

La estimación de divergencias mediante redes neuronales es susceptible a inestabilidad numérica y sobreajuste (_overfitting_), especialmente cuando el soporte de las distribuciones empíricas no se solapa (separación perfecta).

Si la red encuentra un espacio vacío entre $P$ y $Q$, $T_\theta(x)$ tenderá al infinito. Para garantizar una estimación robusta, particularmente ante **derivas de covarianza**, es mandatario aplicar técnicas de regularización:

1. **Weight Decay (Regularización L2):** Penaliza magnitudes grandes en los pesos $\theta$. Permite que la red modele superficies cuadráticas suaves (necesarias para estimar cambios en matrices de covarianza) sin permitir que los gradientes exploten hacia el infinito.
2. **Early Stopping:** Monitorear la convergencia del PSI estimado y detener el entrenamiento cuando alcance una asíntota, evitando que la red comience a memorizar el ruido muestral (separación de puntos aislados).
