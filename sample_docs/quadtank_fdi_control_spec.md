# Quadruple-Tank Process Fault Detection and Isolation (QuadTank-FDI) Technical Specification

## 1. System Dynamics and Non-Minimum Phase Configuration
The quadruple-tank laboratory process consists of four interconnected water tanks ($T_1, T_2, T_3, T_4$) fed by two variable-speed gear pumps ($P_1, P_2$) through two three-way distribution valves ($\gamma_1, \gamma_2 \in [0, 1]$). Pump $P_1$ delivers flow $k_1 u_1$ split into $\gamma_1 k_1 u_1$ entering lower tank $T_1$ and $(1 - \gamma_1) k_1 u_1$ entering upper tank $T_4$. Symmetrically, pump $P_2$ delivers flow $k_2 u_2$ split into $\gamma_2 k_2 u_2$ entering lower tank $T_2$ and $(1 - \gamma_2) k_2 u_2$ entering upper tank $T_3$.

The system operates in a minimum-phase regime when $1 < \gamma_1 + \gamma_2 < 2$ and transitions into a non-minimum-phase regime with a multivariable right-half-plane (RHP) transmission zero when $0 < \gamma_1 + \gamma_2 < 1$. In the standard non-minimum-phase testbench configuration, valve ratios are calibrated at $\gamma_1 = 0.43$ and $\gamma_2 = 0.34$ (sum $\gamma_1 + \gamma_2 = 0.77$). Cross-sectional tank areas are $A_1 = A_3 = 28.0\text{ cm}^2$ and $A_2 = A_4 = 32.0\text{ cm}^2$, with outlet orifice areas $a_1 = a_3 = 0.071\text{ cm}^2$ and $a_2 = a_4 = 0.057\text{ cm}^2$.

## 2. Extended Kalman Filter (EKF) State Estimation and Residual Generation
To monitor unmeasured disturbances and sensor/actuator faults under process and measurement noise, a discrete-time Extended Kalman Filter (EKF) runs at a fixed sampling period of $T_s = 50\text{ ms}$ ($20\text{ Hz}$). The Jacobian matrix $F_k = \left.\frac{\partial f}{\partial x}\right|_{\hat{x}_{k|k-1}}$ linearizes the Bernoulli drain dynamics $\sqrt{2 g h_i}$ around the current state estimate at every step.

The EKF innovation vector $r_k = y_k - C \hat{x}_{k|k-1}$ with theoretical innovation covariance $S_k = C P_{k|k-1} C^\top + R$ serves as the primary analytical redundancy residual. Under nominal fault-free operation, the normalized innovation squared (NIS) statistic $\epsilon_k = r_k^\top S_k^{-1} r_k$ follows a central $\chi^2$ distribution with $m = 2$ degrees of freedom (corresponding to the two ultrasonic level sensors measuring $h_1$ and $h_2$).

## 3. Adaptive CUSUM and GLRT Fault Isolation Thresholds
Fault detection combines a Generalized Likelihood Ratio Test (GLRT) over a sliding window of $N_w = 25$ samples ($1.25\text{ seconds}$) with a two-sided Cumulative Sum (CUSUM) sequential detector. An alarm is triggered when the CUSUM decision function $g_k$ exceeds the detection threshold $J_{\text{th}} = 14.5$, which is calibrated to maintain a false alarm rate (FAR) below $0.2\%$ under nominal sensor noise ($\sigma_v = 0.08\text{ cm}$).

Fault isolation distinguishes between three primary fault modes using directional residual signatures:
- **Actuator Gain Degradation ($\Delta k_1, \Delta k_2$)**: Manifests as a persistent coupled drift in both $r_1$ and $r_2$ aligned with the input direction matrix $B(\gamma_1, \gamma_2)$. Mean isolation latency for a $15\%$ pump efficiency loss is $420\text{ ms}$.
- **Lower-Tank Leakage Fault ($d_{f,1}, d_{f,2}$)**: Produces a strictly negative innovation bias on the affected lower tank without an preceding upper-tank transient.
- **Ultrasonic Sensor Bias ($\Delta y_1, \Delta y_2$)**: Triggers an immediate step jump in the innovation $r_i$ followed by exponential decay as the EKF Kalman gain $K_k$ adapts, yielding an isolation latency under $150\text{ ms}$.

## 4. Actuator Saturation and Safe Fallback Interlocks
Pump control voltages are hardware-clamped to the safe operating range $[0.0\text{ V}, 10.0\text{ V}]$, with a high-level overflow interlock at $h_{\max} = 20.0\text{ cm}$. According to the v2.1 Legacy Plant Controller Manual, the emergency pump shutdown timeout is configured at $45\text{ ms}$ after an over-current fault signal.
