"""舆情与市场特征管线。

从现有 CSV 衍生大盘/截面特征，并可选择接入 Tushare 新闻 + SnowNLP
进行个股情感分析。所有特征合并为长面板格式，与 PortfolioEnv 兼容。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 第一部分：从现有量价数据衍生大盘和截面特征（无需 API）
# ---------------------------------------------------------------------------


def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """在现有长面板 df 上计算大盘级别特征。

    对每个交易日，基于全部股票计算：
    - market_breadth: 当日上涨股票占比 (0~1)
    - market_return: 等权平均涨跌幅
    - market_vol: 截面涨跌幅标准差

    Parameters
    ----------
    df : pd.DataFrame
        长面板数据，必须包含 trade_date, pct_chg 列。

    Returns
    -------
    pd.DataFrame
        包含 trade_date 和市场特征的长面板（每交易日一行）。
    """
    daily = (
        df.groupby("trade_date")
        .agg(
            market_breadth=("pct_chg", lambda x: (x > 0).mean()),
            market_return=("pct_chg", "mean"),
            market_vol=("pct_chg", "std"),
        )
        .reset_index()
    )
    return daily


def compute_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算截面排名特征。

    对每个交易日，在股票池内计算各特征的百分位排名 (0~1)：
    - pct_chg_rank: 涨跌幅排名
    - turnover_rank: 换手率排名
    - pe_rank: 市盈率排名（越低排名越高）

    Parameters
    ----------
    df : pd.DataFrame
        长面板数据。

    Returns
    -------
    pd.DataFrame
        增加 _rank 列后的数据。
    """
    df = df.copy()

    for col, ascending in [("pct_chg", True), ("turnover_rate", True), ("pe_ttm", False)]:
        if col not in df.columns:
            continue
        rank_col = f"{col}_rank"
        df[rank_col] = df.groupby("trade_date")[col].rank(pct=True, ascending=ascending)

    return df


# ---------------------------------------------------------------------------
# 第二部分：Tushare 新闻情感（需 API，有频率限制）
# ---------------------------------------------------------------------------


def fetch_stock_news_sentiment(
    ts_code: str,
    start_date: str,
    end_date: str,
    pro,
    delay: float = 3.0,
) -> pd.DataFrame:
    """获取单只股票的重大新闻并计算每日情感均值。

    调用 Tushare major_news 获取新闻列表，使用 SnowNLP 对每条新闻
    标题进行情感评分 (0=负面, 1=正面)，按交易日聚合。

    Parameters
    ----------
    ts_code : str
        股票代码，如 '600519.SH'。
    start_date : str
        起始日期 YYYYMMDD。
    end_date : str
        结束日期 YYYYMMDD。
    pro : tushare.pro_api
        Tushare API 实例。
    delay : float
        每次 API 调用间隔秒数（避免频率限制）。

    Returns
    -------
    pd.DataFrame
        包含 trade_date, ts_code, news_count, sentiment_mean 的长面板。
    """
    from snownlp import SnowNLP

    try:
        df_news = pro.major_news(ts_code=ts_code, start_date=start_date, end_date=end_date)
    except Exception as e:
        print(f"  [Warn] Tushare major_news failed for {ts_code}: {e}")
        return pd.DataFrame()

    time.sleep(delay)

    if df_news is None or len(df_news) == 0:
        return pd.DataFrame()

    # 情感评分
    sentiments = []
    for _, row in df_news.iterrows():
        content = str(row.get("content", ""))
        if not content or content == "nan":
            continue
        try:
            s = SnowNLP(content)
            sentiments.append(float(s.sentiments))
        except Exception:
            continue

    if not sentiments:
        return pd.DataFrame()

    # 按日期聚合
    df_news["trade_date"] = pd.to_datetime(df_news["datetime"])
    df_news["sentiment"] = sentiments[: len(df_news)]  # 对齐

    daily = (
        df_news.groupby("trade_date")
        .agg(news_count=("sentiment", "count"), sentiment_mean=("sentiment", "mean"))
        .reset_index()
    )
    daily["ts_code"] = ts_code

    return daily[["trade_date", "ts_code", "news_count", "sentiment_mean"]]


def build_sentiment_dataset(
    stock_pool: list[str],
    start_date: str,
    end_date: str,
    pro,
    delay: float = 3.0,
) -> pd.DataFrame:
    """遍历股票池，构建完整的情感数据集。

    Parameters
    ----------
    stock_pool : list[str]
        股票代码列表。
    start_date, end_date : str
        日期范围 YYYYMMDD。
    pro : tushare.pro_api
        Tushare API 实例。
    delay : float
        每次调用间隔秒数。

    Returns
    -------
    pd.DataFrame
        长面板格式的情感数据（可能较稀疏）。
    """
    all_data = []
    n = len(stock_pool)

    for i, ts_code in enumerate(stock_pool):
        print(f"  [{i+1}/{n}] Fetching news for {ts_code}...")
        daily = fetch_stock_news_sentiment(ts_code, start_date, end_date, pro, delay=delay)
        if len(daily) > 0:
            all_data.append(daily)
            print(f"    -> {len(daily)} days with news")
        else:
            print(f"    -> no news found")

    if not all_data:
        print("  [Warn] No sentiment data collected from any stock.")
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    return result


# ---------------------------------------------------------------------------
# 第三部分：合并所有特征到主数据集
# ---------------------------------------------------------------------------


def merge_all_features(
    base_df: pd.DataFrame,
    market_df: pd.DataFrame | None = None,
    sentiment_df: pd.DataFrame | None = None,
    compute_cross_sectional: bool = True,
) -> pd.DataFrame:
    """将市场特征、截面特征和情感特征合并到基础数据集中。

    Parameters
    ----------
    base_df : pd.DataFrame
        主长面板数据（含量价和基本面列）。
    market_df : pd.DataFrame or None
        compute_market_features 的输出，或 None 则自动计算。
    sentiment_df : pd.DataFrame or None
        build_sentiment_dataset 的输出，或 None 则跳过。
    compute_cross_sectional : bool
        是否计算截面排名特征。

    Returns
    -------
    pd.DataFrame
        包含所有特征的长面板。
    """
    df = base_df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    # 截面排名特征（先用原始列计算，再合并大盘特征）
    if compute_cross_sectional:
        df = compute_cross_sectional_features(df)

    # 大盘特征
    if market_df is None:
        market_df = compute_market_features(df)
    market_df["trade_date"] = pd.to_datetime(market_df["trade_date"])
    df = df.merge(market_df, on="trade_date", how="left")

    # 情感特征
    if sentiment_df is not None and len(sentiment_df) > 0:
        sentiment_df["trade_date"] = pd.to_datetime(sentiment_df["trade_date"])
        df = df.merge(sentiment_df, on=["trade_date", "ts_code"], how="left")
        # 填充缺失：无新闻 = 中性情感
        if "news_count" in df.columns:
            df["news_count"] = df["news_count"].fillna(0).astype(int)
        if "sentiment_mean" in df.columns:
            df["sentiment_mean"] = df["sentiment_mean"].fillna(0.5)  # 中性
    else:
        # 即使无情感数据，也创建占位列保证列存在
        if "news_count" not in df.columns:
            df["news_count"] = 0
        if "sentiment_mean" not in df.columns:
            df["sentiment_mean"] = 0.5

    return df


# ---------------------------------------------------------------------------
# 便捷入口：一键增强数据集
# ---------------------------------------------------------------------------


def enhance_dataset(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    stock_pool: list[str] | None = None,
    tushare_token: str | None = None,
    fetch_sentiment: bool = False,
) -> pd.DataFrame:
    """加载 CSV，添加市场和截面特征，可选获取新闻情感，保存增强后的数据。

    Parameters
    ----------
    csv_path : str or Path
        输入 CSV 路径。
    output_path : str or Path or None
        输出路径。None 则不保存。
    stock_pool : list[str] or None
        股票池。仅 fetch_sentiment=True 时需要。
    tushare_token : str or None
        Tushare token。仅 fetch_sentiment=True 时需要。
    fetch_sentiment : bool
        是否调用 Tushare 获取新闻情感（耗时较长）。

    Returns
    -------
    pd.DataFrame
        增强后的长面板数据。
    """
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    # 确定日期范围
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    start = df["trade_date"].min().strftime("%Y%m%d")
    end = df["trade_date"].max().strftime("%Y%m%d")
    print(f"  Date range: {start} ~ {end}")

    # 情感数据（可选）
    sentiment_df = None
    if fetch_sentiment and stock_pool and tushare_token:
        import tushare as ts

        ts.set_token(tushare_token)
        pro = ts.pro_api()
        print(f"Fetching news sentiment for {len(stock_pool)} stocks...")
        sentiment_df = build_sentiment_dataset(stock_pool, start, end, pro, delay=3.0)

    # 合并所有特征
    df = merge_all_features(df, sentiment_df=sentiment_df, compute_cross_sectional=True)

    # 新增大盘特征列
    new_cols = ["market_breadth", "market_return", "market_vol",
                "pct_chg_rank", "turnover_rate_rank", "pe_ttm_rank",
                "news_count", "sentiment_mean"]

    added = [c for c in new_cols if c in df.columns]
    print(f"  Added features: {added}")
    print(f"  Final columns: {len(df.columns)}")
    print(f"  Final rows: {len(df)}")

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")

    return df
