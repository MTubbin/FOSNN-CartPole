# MT25-tLab

Work completed in the teuscher.:Lab as a research intern - Summer 2025


continuous-time Grünwald–Letnikov Fractional Derivative:

$D^\alpha V(t) = \lim_{h \to 0} \frac{1}{h^\alpha} \sum_{k=0}^{N} (-1)^k \binom{\alpha}{k} V(t - kh)$

Converted to a time-discrete update formula for approximation of membrane voltage

$V_m(t_{n+1}) = \alpha V_m(t_n) + h^\alpha\cdot\Biggl(\frac{-(V_m(t_n)-V_{rest})+R_mI_{syn}(t_n)}{\tau_m}\Biggl) - \sum_{k=2}^{n+1}\omega_k^{\alpha}V_m(t_{n+1-k})$
    

