import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse

# Simple LSTMForecaster (same architecture as notebook)
class LSTMForecaster(nn.Module):
    def __init__(self, n_assets, hidden=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_assets,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden, n_assets)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last))


def build_daily_to_monthly_dataset(daily_df, seq_len_days=252, train_start='2014-07-01', train_end='2020-06-30'):
    daily = daily_df.sort_index()
    daily_train = daily.loc[train_start:train_end]
    # monthly targets — handle pandas frequency alias changes ('M' may be unsupported)
    try:
        monthly_targets = daily_train.resample('M').apply(lambda g: (1 + g).prod() - 1).dropna(how='all')
    except ValueError:
        monthly_targets = daily_train.resample('ME').apply(lambda g: (1 + g).prod() - 1).dropna(how='all')

    idx = daily_train.index
    arr = daily_train.values.astype(np.float32)

    X_list, y_list, months = [], [], []
    for mt in monthly_targets.index:
        # last trading day for month
        le = idx[idx <= mt]
        if len(le) == 0:
            continue
        last_td = le[-1]
        pos = idx.get_loc(last_td)
        start = pos - seq_len_days + 1
        if start < 0:
            continue
        X_list.append(arr[start:pos+1])
        y_list.append(monthly_targets.loc[mt].values.astype(np.float32))
        months.append(mt)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, months


def train_model(X, y, device, hidden=64, n_layers=2, dropout=0.2, epochs=200, batch_size=16, lr=1e-3, patience=20):
    n_samples, seq_len, n_assets = X.shape
    split = int(0.8 * n_samples)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    X_tr_t = torch.tensor(X_tr).to(device)
    y_tr_t = torch.tensor(y_tr).to(device)
    X_val_t = torch.tensor(X_val).to(device)
    y_val_t = torch.tensor(y_val).to(device)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)

    model = LSTMForecaster(n_assets, hidden=hidden, n_layers=n_layers, dropout=dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=10, factor=0.5)

    best_val = float('inf'); best_state = None; no_imp = 0
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in loader:
            optim.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item() if len(X_val_t) > 0 else 0.0
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f'Early stop at epoch {ep} best_val={best_val:.6f}')
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    # preds for all samples
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
    return model, preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--daily_csv', default='data/raw/daily_returns.csv')
    parser.add_argument('--seq_days', type=int, default=252)
    parser.add_argument('--out_csv', default='data/processed/nse_pred_from_daily.csv')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    daily = pd.read_csv(args.daily_csv, index_col=0, parse_dates=True)
    daily.index = pd.to_datetime(daily.index)

    # Use same NSE tickers defined in notebook if available
    NSE_TICKERS = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS',
        'BAJFINANCE.NS', 'WIPRO.NS', 'MARUTI.NS', 'TITAN.NS', 'ULTRACEMCO.NS',
        'ASIANPAINT.NS', 'SUNPHARMA.NS', 'NESTLEIND.NS'
    ]
    available = [t for t in NSE_TICKERS if t in daily.columns]
    if len(available) < len(NSE_TICKERS):
        print(f'Warning: missing {len(NSE_TICKERS)-len(available)} tickers; using {len(available)} present.')
    daily_nse = daily[available]

    X, y, months = build_daily_to_monthly_dataset(daily_nse, seq_len_days=args.seq_days)
    if len(X) == 0:
        raise SystemExit('No training samples created (increase history or reduce seq_days).')
    device = torch.device(args.device)
    model, preds = train_model(X, y, device, hidden=64, n_layers=2, dropout=0.2, epochs=200, batch_size=16)

    cols = [f'S{i+1}' for i in range(preds.shape[1])]
    df_preds = pd.DataFrame(preds, index=pd.DatetimeIndex(months), columns=cols)
    df_preds.to_csv(args.out_csv)
    print('Saved monthly predictions to', args.out_csv)
    # also save model
    torch.save(model.state_dict(), os.path.splitext(args.out_csv)[0] + '_model.pt')
    print('Saved model to', os.path.splitext(args.out_csv)[0] + '_model.pt')
