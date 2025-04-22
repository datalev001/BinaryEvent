# A Comparative Framework for Binary Event Forecasting in Financial Time Series
Evaluating CDF‑Based Trees, Calibrated GLMs, Genetic Sequence Models, and Traditional Benchmarks
Forecasting a continuous quantity - like tomorrow's total sales volume - is a well‑trodden path in time‑series analysis. By contrast, estimating the chance that a one‑off action occurs (for example, when a subscriber next reactivates a service or a site visitor completes a checkout) poses unique challenges. Such binary‑event forecasts demand accurate probability scores (AUC, Brier, log‑loss), not just point estimates, and must handle long runs of "no‑event" punctuated by infrequent spikes.
Traditional models (ARIMA, exponential smoothing) excel at capturing smooth trends and seasonality in continuous data but often struggle when the target is a rare, discrete occurrence. In binary forecasting, practitioners care both about the immediate probability of an event tomorrow and the joint probability of at least one event in the next K days - for instance, "What is the likelihood a user will engage at least once this week"? Addressing this requires sequence encoders (e.g. n‑gram attention), multiclass heads over counts 0…K, and strong calibration strategies to produce reliable uncertainty intervals.
In this paper, we implement and compare eight approaches on real engagement data:
Random Forest + Gaussian CDF exceedance
Elastic‑Net Logistic Regression + Platt / Isotonic calibration
Discrete n‑gram Attention + Softmax Count Head
XGBoost classifier
Markov Chain model
ARIMA & Prophet + CDF exceedance
Recurrent Neural Network
Lightweight State‑Space binary event model
Attention based model

Each method is evaluated via out‑of‑time log‑loss, Brier score, AUC. We demonstrate that architectures tailored to binary events - and proper calibration - significantly outperform generic continuous‑series techniques, offering both theoretical insights and practical guidance for data scientists tackling discrete‑action forecasting.
