
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.special import expit

from sklearn.ensemble       import RandomForestRegressor
from sklearn.metrics        import log_loss, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler, PolynomialFeatures
from sklearn.linear_model   import LogisticRegressionCV
from sklearn.calibration    import CalibratedClassifierCV

import pygad

import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1) DATA FETCH & LABELING
# ——————————————————————————————
# 1) DOWNLOAD & LABEL SPY
def fetch_spy(days=800):
    df = yf.Ticker("SPY").history(period=f"{days}d")[["Close"]].reset_index()
    df.rename(columns={"Date": "date", "Close": "close"}, inplace=True)
    return df

def label_event_onset(df, threshold=0.00):
    """
    Labels an event at time t if the maximum of the next two days' closes
    exceeds today's close by at least the given threshold.
    """
    # look one and two days ahead, take the elementwise max
    future_max = df["close"].shift(-1).combine(df["close"].shift(-2), max)
    df["event"] = (future_max > df["close"] * (1 + threshold)).astype(int)
    df.dropna(inplace=True)
    return df

# 2) FEATURE ENGINEERING (no leakage)
def make_features(df, n_lags=7, roll_windows=(5, 21)):
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df.close.pct_change(lag).shift(1)
    for w in roll_windows:
        df[f"roll_{w}"] = (
            df.close.pct_change()
              .rolling(w)
              .mean()
              .shift(1)
        )
    df.dropna(inplace=True)
    return df


# 2) RF + Gaussian‑CDF BENCHMARK
# ——————————————————————————————
def benchmark_rf_cdf(train, test):
    Xtr = train.filter(like="lag_").join(train.filter(like="roll_"))
    ytr = train.event.values
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(Xtr, ytr)

    Xte = test.filter(like="lag_").join(test.filter(like="roll_"))
    pf = rf.predict(Xte)
    sigma = ytr.std()
    p1 = norm.sf(0.5, loc=pf, scale=sigma)

    return {
        "LogLoss": log_loss(test.event, p1),
        "Brier":   brier_score_loss(test.event, p1),
        "AUC":     roc_auc_score(test.event, p1)
    }, p1


# 3) DYNAMIC BASELINE (Logistic→Poly→ElasticNet→Isotonic)
# ——————————————————————————————
def dynamic_baseline_full(df):
    """
    1) Build lag_*/roll_* matrix and binary event y.
    2) Expand to interaction features (degree=2, no bias).
    3) Scale everything.
    4) Fit an Elastic‑Net LogisticRegressionCV (time‑series CV, class_weight balanced).
    5) Wrap in isotonic CalibratedClassifierCV (same splits).
    6) Return calibrated P(event=1) on entire df.
    """
    # 1) raw X,y
    X = df.filter(like="lag_").join(df.filter(like="roll_"))
    y = df.event.values

    # 2) time‑series splits
    tscv = TimeSeriesSplit(n_splits=5)

    # 3) build pipeline
    pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=2,
                                     interaction_only=True,
                                     include_bias=False)),
        ("scale",  StandardScaler()),
        ("lrcv",   LogisticRegressionCV(
                       Cs=[0.01, 0.1, 1.0, 10.0],
                       cv=tscv,
                       penalty="elasticnet",
                       solver="saga",
                       l1_ratios=[0.5, 0.7, 1.0],
                       class_weight="balanced",
                       scoring="roc_auc",
                       max_iter=2000,
                       n_jobs=-1,
                   ))
    ])

    # 4) fit and report best hyperparams
    pipe.fit(X, y)
    best_lr = pipe.named_steps["lrcv"]
    #print(f"[dynamic_baseline] best C = {best_lr.C_[0]:.3f}, l1_ratio = {best_lr.l1_ratio_[0]:.2f}, CV AUC ≈ {best_lr.scores_[1].mean():.4f}")

    # 5) isotonic calibration
    iso = CalibratedClassifierCV(pipe, method="isotonic", cv=tscv)
    iso.fit(X, y)

    # 6) return calibrated probs on full set
    return iso.predict_proba(X)[:, 1]

def benchmark_xgb_clf(train, test):
    # 1) Feature matrix & labels
    Xtr = train.filter(like="lag_").join(train.filter(like="roll_"))
    ytr = train.event.values
    Xte = test.filter(like="lag_").join(test.filter(like="roll_"))

    # 2) Fit an XGBClassifier
    #    scale_pos_weight balances the positive vs negative class frequencies
    neg, pos = np.bincount(ytr)
    xgb_clf = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        scale_pos_weight=neg/pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    xgb_clf.fit(Xtr, ytr)

    # 3) Predict probabilities
    p1 = xgb_clf.predict_proba(Xte)[:,1]

    # 4) Compute standard metrics
    return {
        "LogLoss": log_loss(test.event, p1),
        "Brier":   brier_score_loss(test.event, p1),
        "AUC":     roc_auc_score(test.event, p1)
    }, p1

# 4) PURE GA–EVOLVED SOFTMAX BINARY FORECASTER
# ——————————————————————————————
def ga_binary_forecast(train, test,
                       pop_size=50, generations=100, mutation_percent=5):
    # build features & label arrays
    Xtr = train.filter(like="lag_").join(train.filter(like="roll_")).values
    ytr = train.event.values.astype(float)
    Xte = test.filter(like="lag_").join(test.filter(like="roll_")).values

    # append bias
    Xtr_ = np.hstack([Xtr, np.ones((len(Xtr),1))])
    Xte_ = np.hstack([Xte, np.ones((len(Xte),1))])
    n_genes = Xtr_.shape[1]

    # fitness must accept (ga, solution, sol_idx)
    def fitness_fn(ga_inst, solution, sol_idx):
        p = expit(Xtr_.dot(solution))
        # we return positive fitness = –log_loss
        return -log_loss(ytr, p)

    ga = pygad.GA(num_generations=generations,
                  num_parents_mating=pop_size//2,
                  fitness_func=fitness_fn,
                  sol_per_pop=pop_size,
                  num_genes=n_genes,
                  mutation_percent_genes=mutation_percent,
                  mutation_type="random",
                  crossover_type="single_point",
                  suppress_warnings=True)

    ga.run()
    w_opt, fitness, _ = ga.best_solution()
    print(f"[GA] best neg‑logloss = {fitness:.4f}")

    # test predictions
    p_te = expit(Xte_.dot(w_opt))
    return {
        "LogLoss": log_loss(test.event, p_te),
        "Brier":   brier_score_loss(test.event, p_te),
        "AUC":     roc_auc_score(test.event, p_te)
    }

def benchmark_markov(train, test):
    ev = train.event.values
    # count transitions
    c00 = ((ev[:-1]==0)&(ev[1:]==0)).sum()
    c01 = ((ev[:-1]==0)&(ev[1:]==1)).sum()
    c10 = ((ev[:-1]==1)&(ev[1:]==0)).sum()
    c11 = ((ev[:-1]==1)&(ev[1:]==1)).sum()
    p01 = c01 / (c00 + c01)  # P(1|0)
    p11 = c11 / (c10 + c11)  # P(1|1)

    # sequential one‐step‐ahead using true previous state
    prev = ev[-1]  # last train state
    p1 = []
    for actual in test.event.values:
        prob = p11 if prev==1 else p01
        p1.append(prob)
        prev = actual

    p1 = np.array(p1)
    return {
        "LogLoss": log_loss(test.event, p1),
        "Brier":   brier_score_loss(test.event, p1),
        "AUC":     roc_auc_score(test.event, p1)
    }, p1

# 5) ARIMA + Gaussian‑CDF
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.base.tsa_model import ValueWarning as StatsmodelsValueWarning

def arima_event_proba(train, test, threshold=0.005, order=(1,1,1)):
    """
    Fit an ARIMA on past returns and compute P(event=1) by
    approximating the one‐step ahead forecast error distribution.
    Suppresses statsmodels warnings about indexes & convergence.
    """
    # compute returns
    train_ret = train.close.pct_change().dropna()

    p1 = []
    for i in range(len(test)):
        # build history up to t–1
        history = pd.concat([train_ret, test.close.pct_change().iloc[:i]]).dropna()

        with warnings.catch_warnings():
            # ignore unsupported‐index & convergence warnings
            warnings.simplefilter("ignore", StatsmodelsValueWarning)
            warnings.simplefilter("ignore", ConvergenceWarning)

            model = ARIMA(history, order=order)
            fit   = model.fit()

            # one‐step forecast
            fc    = fit.get_forecast(steps=1)
            mu    = fc.predicted_mean.iloc[0]
            var   = fc.var_pred_mean.iloc[0]

        sigma = np.sqrt(var)
        # P(next_return > threshold)
        p1.append(1 - norm.cdf(threshold, loc=mu, scale=sigma))

    p1 = np.array(p1)
    mask = ~np.isnan(p1)
    return {
        "LogLoss": log_loss(test.event[mask], p1[mask]),
        "Brier":   brier_score_loss(test.event[mask], p1[mask]),
        "AUC":     roc_auc_score(test.event[mask], p1[mask])
    }, p1


# 6) Prophet + Gaussian‑CDF
from prophet import Prophet
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from scipy.stats import norm

def prophet_event_proba(train, test, threshold=0.003):
    """
    Fit a Prophet model on daily returns in `train` and produce P(event=1) on `test`.
    Event is defined as: max(next two days’ close) > today*(1+threshold),
    but here we forecast returns so threshold is a pct change.
    
    Returns (metrics_dict, p1_array).
    """
    # 1) build returns history for Prophet
    dfp = train[['date','close']].rename(columns={'date':'ds','close':'y'})
    # compute simple pct‐returns
    dfp['y'] = dfp['y'].pct_change().fillna(0)
    # strip any timezone
    if pd.api.types.is_datetime64tz_dtype(dfp['ds']):
        dfp['ds'] = dfp['ds'].dt.tz_convert(None).dt.tz_localize(None)
    else:
        try:
            dfp['ds'] = dfp['ds'].dt.tz_localize(None)
        except (TypeError, ValueError):
            pass

    # 2) fit Prophet on returns
    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05
    )
    m.fit(dfp)

    # 3) construct exactly the test‐set dates for forecasting
    future = test[['date']].rename(columns={'date':'ds'})
    # drop tz if any
    if pd.api.types.is_datetime64tz_dtype(future['ds']):
        future['ds'] = future['ds'].dt.tz_convert(None).dt.tz_localize(None)
    else:
        try:
            future['ds'] = future['ds'].dt.tz_localize(None)
        except (TypeError, ValueError):
            pass

    # 4) predict returns on those dates
    fc = m.predict(future)
    yhat   = fc['yhat'].values
    sigma  = ((fc['yhat_upper'] - fc['yhat_lower'])/(2*1.96)).values
    sigma  = np.maximum(sigma, 1e-6)   # guard against zero width

    # 5) compute P(return > threshold)
    p1 = norm.sf(threshold, loc=yhat, scale=sigma)
    p1 = np.clip(p1, 0, 1)

    # 6) evaluate against the binary event in `test`
    #    note: test.event must already be defined consistently
    mask = ~np.isnan(p1)
    y_true = test['event'].to_numpy()[mask]
    y_pred = p1[mask]

    metrics = {
        "LogLoss": log_loss(y_true, y_pred),
        "Brier":   brier_score_loss(y_true, y_pred),
        "AUC":     max(1- roc_auc_score(y_true, y_pred), roc_auc_score(y_true, y_pred))
    }

    return metrics, p1

# 5) RNN METHOD
def rnn_event_proba(train, test,
                    n_lags=7,
                    hidden_dim=32,
                    num_layers=1,
                    lr=1e-3,
                    epochs=10,
                    batch_size=32):
    """
    Train a simple LSTM on lagged returns to predict event[t], then
    output P(event=1) on `test`.
    """
    # prepare return series + targets
    tr_ret = train.close.pct_change().fillna(0).values
    te_ret = test.close.pct_change().fillna(0).values
    tr_y   = train.event.values
    te_y   = test.event.values

    # build sequences
    X_tr, y_tr = [], []
    for i in range(n_lags, len(tr_ret)):
        X_tr.append(tr_ret[i-n_lags:i])
        y_tr.append(tr_y[i])
    X_te, y_te = [], []
    for i in range(n_lags, len(te_ret)):
        X_te.append(te_ret[i-n_lags:i])
        y_te.append(te_y[i])

    X_tr = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(-1)
    y_tr = torch.tensor(y_tr, dtype=torch.float32)
    X_te = torch.tensor(X_te, dtype=torch.float32).unsqueeze(-1)
    y_te = torch.tensor(y_te, dtype=torch.float32)

    train_ds = TensorDataset(X_tr, y_tr)
    test_ds  = TensorDataset(X_te, y_te)
    tr_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    te_ld = DataLoader(test_ds,  batch_size=batch_size)

    # simple LSTM
    class RNNForecast(nn.Module):
        def __init__(self, inp_dim, h_dim, nlayers):
            super().__init__()
            self.lstm = nn.LSTM(inp_dim, h_dim, nlayers, batch_first=True)
            self.fc   = nn.Linear(h_dim, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            h_last = out[:,-1,:]
            logit  = self.fc(h_last).squeeze(-1)
            return torch.sigmoid(logit)

    model = RNNForecast(1, hidden_dim, num_layers).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    # train
    model.train()
    for _ in range(epochs):
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            p = model(xb)
            loss_fn(p, yb).backward()
            opt.step()

    # predict
    model.eval()
    all_p = []
    with torch.no_grad():
        for xb, _ in te_ld:
            xb = xb.to(device)
            all_p.extend(model(xb).cpu().numpy())
    all_p = np.array(all_p)
    y_true = y_te.numpy()

    return {
        "LogLoss": log_loss(y_true, all_p),
        "Brier":   brier_score_loss(y_true, all_p),
        "AUC":     roc_auc_score(y_true, all_p)
    }, all_p

# -----------------------------------------------------------------------------
#  ATTENTION‐BASED FORECAST FUNCTION
# -----------------------------------------------------------------------------
def attention_event_proba(train, test, n_lags=7,
                          embed_dim=32, num_heads=4,
                          hidden_dim=64, lr=1e-3,
                          epochs=10, batch_size=32,
                          device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # a) compute pct‐returns & align events
    tr_ret = train.close.pct_change().fillna(0).values
    te_ret = test.close.pct_change().fillna(0).values
    tr_evt = train.event.values
    te_evt = test.event.values

    # b) build sliding windows
    X_tr, y_tr = [], []
    for i in range(n_lags, len(tr_ret)):
        X_tr.append(tr_ret[i-n_lags:i])
        y_tr.append(tr_evt[i])
    X_te, y_te = [], []
    for i in range(n_lags, len(te_ret)):
        X_te.append(te_ret[i-n_lags:i])
        y_te.append(te_evt[i])

    X_tr = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(-1)  # (N, L, 1)
    y_tr = torch.tensor(y_tr, dtype=torch.float32)
    X_te = torch.tensor(X_te, dtype=torch.float32).unsqueeze(-1)
    y_te = torch.tensor(y_te, dtype=torch.float32)

    tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size)

    # c) model definition
    class AttnForecaster(nn.Module):
        def __init__(self, input_dim, embed_dim, num_heads, hidden_dim):
            super().__init__()
            self.project = nn.Linear(input_dim, embed_dim)
            self.attn    = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.mlp     = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, x):
            # x: (B, L, 1)
            h = self.project(x)                  # → (B, L, E)
            a, _ = self.attn(h, h, h)            # → (B, L, E)
            v = a.mean(dim=1)                    # → (B, E)
            return torch.sigmoid(self.mlp(v)).squeeze(-1)  # → (B,)

    model = AttnForecaster(1, embed_dim, num_heads, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.BCELoss()

    # d) training loop
    model.train()
    for _ in range(epochs):
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            p = model(xb)
            loss_fn(p, yb).backward()
            optimizer.step()

    # e) prediction
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in te_loader:
            xb = xb.to(device)
            preds.extend(model(xb).cpu().numpy())

    y_true = y_te.numpy()
    p1     = np.array(preds)

    # f) metrics
    return {
        "LogLoss": log_loss(y_true, p1),
        "Brier":   brier_score_loss(y_true, p1),
        "AUC":     max(roc_auc_score(y_true, p1), 1- roc_auc_score(y_true, p1))
    }, p1


# 5) MAIN
# ——————————————————————————————
if __name__=="__main__":
    df = fetch_spy(700)
    df = label_event_onset(df, threshold=0.003)
    df = make_features(df)

    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    bm_metrics, _ = benchmark_rf_cdf(train_df, test_df)
    print("=== RF+Gaussian‑CDF BENCHMARK ===", bm_metrics)

    dynp = dynamic_baseline_full(df)
    print("=== Dynamic Logistic+Platt AUC (full) ===",
          roc_auc_score(df.event, dynp))


    # 3) pure GA binary softmax forecaster
    print("=== GA‑Softmax Binary Forecast AUC ===")
    print(ga_binary_forecast(train_df, test_df))

    print("\n=== XGB Classifier Benchmark ===")
    print(benchmark_xgb_clf(train_df, test_df)[0])
    
    # Markov Chain
    mk_metrics, _ = benchmark_markov(train_df, test_df)
    print("=== Markov Chain Benchmark ===", mk_metrics)
    
    # Memba
    memb_metrics, _ = benchmark_hmm(train_df, test_df, n_states=2)
    print("=== Memba Model Benchmark ===", memb_metrics)
    
    # arima for binary target
    arima_m, _  = arima_event_proba(train_df, test_df)
    print("=== ARIMA+CDF ===", arima_m)
    
    prop_m, _   = prophet_event_proba(train_df, test_df)
    print("=== Prophet+CDF ===", prop_m)
    
    # RNN method
    rnn_m, _ = rnn_event_proba(train_df, test_df,
                              n_lags=7,hidden_dim=32,
                              num_layers=1, lr=1e-3,
                              epochs=10, batch_size=32)
    print("=== RNN Forecast AUC ===", rnn_m["AUC"])
    
    attn_metrics, attn_p1 = attention_event_proba(train_df, test_df)
    print("=== Attention Forecast AUC ===", attn_metrics)