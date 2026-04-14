import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# --- Configuration ---
FILE_PATHS = {
    '1D': 'GBPUSD_1D.csv',
    '1H': 'GBPUSD_1H.csv',
    '1M': 'GBPUSD_1M.csv',  # Note: MT5 treats 1M as 1-Minute. MN is Monthly.
    '1W': 'GBPUSD_1W.csv',
    '4H': 'GBPUSD_4H.csv',
    'MN': 'GBPUSD_MN.csv'   # Added in case you export Monthly data
}

def load_data(filepath):
    """Loads MT5 CSV data and formats dates correctly."""
    df = pd.read_csv(filepath)
    if len(df.columns) == 1: 
        df = pd.read_csv(filepath, sep='\t')
    df.columns = df.columns.str.replace(r'[<>]', '', regex=True).str.strip().str.lower()
    
    if 'date' in df.columns and 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    elif 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['time'])
    elif 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
    else:
        raise KeyError("Time data not found.")
        
    df.set_index('datetime', inplace=True)
    df.index = df.index + pd.Timedelta(hours=5, minutes=30) # IST
    
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['open', 'high', 'low', 'close'])

def process_logic(df):
    """Executes the exact logic extracted from the Pine Script."""
    stLen = 5
    itLen = 10
    
    df['STH'] = False
    df['STL'] = False
    df['ITH'] = False
    df['ITL'] = False

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    opens = df['open'].values

    lastIth = np.nan
    lastItl = np.nan
    structureBias = 0
    
    active_fvgs = []
    active_obs = []

    body_sizes = np.abs(closes - opens)
    avgBody = pd.Series(body_sizes).rolling(20).mean().fillna(0).values

    for i in range(20, len(df)):
        # 1. Fractals Detection
        if i >= stLen * 2:
            is_sth = True
            for j in range(i - stLen * 2, i + 1):
                if j != i - stLen and highs[j] > highs[i - stLen]:
                    is_sth = False
                    break
            if is_sth: df.iloc[i - stLen, df.columns.get_loc('STH')] = True
            
            is_stl = True
            for j in range(i - stLen * 2, i + 1):
                if j != i - stLen and lows[j] < lows[i - stLen]:
                    is_stl = False
                    break
            if is_stl: df.iloc[i - stLen, df.columns.get_loc('STL')] = True

        # ITH/ITL
        if i >= itLen * 2:
            is_ith = True
            for j in range(i - itLen * 2, i + 1):
                if j != i - itLen and highs[j] > highs[i - itLen]:
                    is_ith = False
                    break
            if is_ith:
                df.iloc[i - itLen, df.columns.get_loc('ITH')] = True
                lastIth = highs[i - itLen]
            
            is_itl = True
            for j in range(i - itLen * 2, i + 1):
                if j != i - itLen and lows[j] < lows[i - itLen]:
                    is_itl = False
                    break
            if is_itl:
                df.iloc[i - itLen, df.columns.get_loc('ITL')] = True
                lastItl = lows[i - itLen]

        # 2. Market Structure (BOS Logic)
        bosVal = closes[i]
        bosBull = False
        bosBear = False
        
        if not np.isnan(lastIth) and bosVal > lastIth:
            bosBull = True
            structureBias = 1
        if not np.isnan(lastItl) and bosVal < lastItl:
            bosBear = True
            structureBias = -1
            
        # 3. FVG Logic
        dispCandle = abs(closes[i-1] - opens[i-1]) > avgBody[i] * 1.5
        bullFvg = (highs[i-2] < lows[i]) and dispCandle
        bearFvg = (lows[i-2] > highs[i]) and dispCandle

        if bullFvg: active_fvgs.append({'idx': i-2, 'mid': i-1, 'type': 'Bullish', 'top': lows[i], 'bot': highs[i-2], 'active': True})
        if bearFvg: active_fvgs.append({'idx': i-2, 'mid': i-1, 'type': 'Bearish', 'top': lows[i-2], 'bot': highs[i], 'active': True})

        # 4. High Prob OB Logic
        if bosBull and bullFvg:
            lb = 0
            for j in range(1, 6):
                idx = i - j
                if closes[idx] < opens[idx]:
                    lb = j
                    break
            if lb != 0: active_obs.append({'idx': i - lb, 'type': 'Bullish', 'top': highs[i - lb], 'bot': lows[i - lb], 'active': True})
                
        if bosBear and bearFvg:
            lb = 0
            for j in range(1, 6):
                idx = i - j
                if closes[idx] > opens[idx]:
                    lb = j
                    break
            if lb != 0: active_obs.append({'idx': i - lb, 'type': 'Bearish', 'top': highs[i - lb], 'bot': lows[i - lb], 'active': True})

        # 5. Mitigations
        for f in active_fvgs:
            if f['active']:
                if f['type'] == 'Bullish' and lows[i] <= f['top']: f['active'] = False
                if f['type'] == 'Bearish' and highs[i] >= f['bot']: f['active'] = False

        for o in active_obs:
            if o['active'] and o['idx'] < i:
                if o['type'] == 'Bullish' and lows[i] <= o['bot']: o['active'] = False
                if o['type'] == 'Bearish' and highs[i] >= o['top']: o['active'] = False

    # Cleanup overlaps (if ITH, not STH)
    df.loc[df['ITH'] == True, 'STH'] = False
    df.loc[df['ITL'] == True, 'STL'] = False

    return df, active_fvgs, active_obs

def plot_chart(df, fvgs, obs, tf, output_dir):
    """Generates the Dark Theme Cyan/White aesthetic."""
    
    # Timeframe constraint: Keep all if Monthly ('MN' or '1MN'), otherwise last 250 candles
    if tf.upper() in ['MN', '1MN', 'MONTHLY']:
        plot_df = df
    else:
        plot_df = df.tail(250)
    
    # Exact Dark Theme Colors
    BG_COLOR = '#131722'      # TradingView Deep Dark
    BULL_COLOR = '#00BCD4'    # Neon Cyan
    BEAR_COLOR = '#FFFFFF'    # Pure White
    LINE_COLOR = '#A0A6B5'    # Soft Gray for Bracket Lines
    TEXT_COLOR = '#FFFFFF'    # White for text
    
    fig, ax = plt.subplots(figsize=(16, 7.5), dpi=200, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    # Remove all standard axes formatting
    ax.grid(False)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    candle_width = 0.65
    for i in range(len(plot_df)):
        o, c = plot_df['open'].iloc[i], plot_df['close'].iloc[i]
        h, l = plot_df['high'].iloc[i], plot_df['low'].iloc[i]
        
        color = BULL_COLOR if c >= o else BEAR_COLOR
        
        # Wicks
        ax.plot([i, i], [l, h], color=color, linewidth=1.0, zorder=2)
        
        # Bodies
        if abs(c - o) > 0:
            ax.add_patch(Rectangle((i - candle_width/2, min(o, c)), candle_width, abs(c - o), facecolor=color, edgecolor='none', zorder=3))
        else:
            ax.plot([i - candle_width/2, i + candle_width/2], [o, c], color=color, linewidth=1.5, zorder=3)

    # Calculate bracket line proportions based on the visible range
    price_range = plot_df['high'].max() - plot_df['low'].min()
    y_offset = price_range * 0.03  # Tighter offset since we have 250 candles
    x_offset = 4                   # Slightly wider horizontal line
    
    def draw_bracket(idx, y_val, label, is_top, is_intermediate):
        x_pos = np.where(plot_df.index == idx)[0][0]
        sign = 1 if is_top else -1
        y_end = y_val + (y_offset * sign)
        
        # Emphasize Intermediate Term Highs/Lows with a slightly thicker, brighter line
        lw = 1.5 if is_intermediate else 0.8
        l_col = TEXT_COLOR if is_intermediate else LINE_COLOR
        
        ax.plot([x_pos, x_pos], [y_val, y_end], color=l_col, lw=lw, zorder=4)
        ax.plot([x_pos, x_pos + x_offset], [y_end, y_end], color=l_col, lw=lw, zorder=4)
        ax.text(x_pos + x_offset + 0.5, y_end, label, color=l_col, fontsize=7, va='center', fontweight='bold', zorder=5)

    for idx, row in plot_df[plot_df['STH']].iterrows(): draw_bracket(idx, row['high'], 'STH', True, False)
    for idx, row in plot_df[plot_df['STL']].iterrows(): draw_bracket(idx, row['low'], 'STL', False, False)
    for idx, row in plot_df[plot_df['ITH']].iterrows(): draw_bracket(idx, row['high'], 'ITH', True, True)
    for idx, row in plot_df[plot_df['ITL']].iterrows(): draw_bracket(idx, row['low'], 'ITL', False, True)

    # Dark Mode FVGs and OBs
    for z in [f for f in fvgs if f['active'] and plot_df.index[0] <= df.index[f['idx']]]:
        try:
            x_start = np.where(plot_df.index == df.index[f['idx']])[0][0]
            z_color = BULL_COLOR if z['type'] == 'Bullish' else BEAR_COLOR
            ax.axhspan(z['bot'], z['top'], xmin=x_start/len(plot_df), xmax=1, color=z_color, alpha=0.15, lw=0, zorder=1)
            b_y = z['top'] if z['type'] == 'Bullish' else z['bot']
            ax.axhline(b_y, xmin=x_start/len(plot_df), xmax=1, color=z_color, linestyle='--', lw=0.8, alpha=0.5, zorder=2)
            ax.text(len(plot_df)-1, b_y, ' FVG', color=z_color, fontsize=7, va='center', alpha=0.8)
        except IndexError: pass

    for z in [o for o in obs if o['active'] and plot_df.index[0] <= df.index[o['idx']]]:
        try:
            x_start = np.where(plot_df.index == df.index[o['idx']])[0][0]
            z_color = BULL_COLOR if z['type'] == 'Bullish' else BEAR_COLOR
            ax.axhspan(z['bot'], z['top'], xmin=x_start/len(plot_df), xmax=1, color=z_color, alpha=0.25, lw=0, zorder=1)
            b_y = z['top'] if z['type'] == 'Bullish' else z['bot']
            ax.axhline(b_y, xmin=x_start/len(plot_df), xmax=1, color=z_color, linestyle='-', lw=1, alpha=0.6, zorder=2)
            ax.text(len(plot_df)-1, b_y, ' OB', color=z_color, fontsize=7, va='center', alpha=0.8)
        except IndexError: pass

    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(output_dir, f"{tf}_Chart.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

def export_excel(df, tf, output_dir):
    excel_path = os.path.join(output_dir, f"{tf}_Swings.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for name in ['STH', 'STL', 'ITH', 'ITL']:
            price_col = 'high' if 'H' in name else 'low'
            data = df[df[name]][[price_col]].tail(250)
            data.index = data.index.strftime('%Y-%m-%d %H:%M:%S')
            data.to_excel(writer, sheet_name=name)
        
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            worksheet.column_dimensions['A'].width = 25
            worksheet.column_dimensions['B'].width = 15

# --- Execution ---
if __name__ == "__main__":
    output_dir = 'Market_Structure_Output'
    os.makedirs(output_dir, exist_ok=True)
    
    for tf, path in FILE_PATHS.items():
        if os.path.exists(path):
            print(f"Processing {tf}...")
            df = load_data(path)
            df, fvgs, obs = process_logic(df)
            plot_chart(df, fvgs, obs, tf, output_dir)
            export_excel(df, tf, output_dir)
            print(f"[{tf}] Exported Data and Image successfully.")