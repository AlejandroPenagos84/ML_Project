"""Columnas y constantes compartidas; no hay ejercicios en este archivo."""

SEED = 42
CUTOFFS = {"train_feb": "2017-01-31", "test": "2017-02-28"}
TXN_COLUMNS = [
    "msno", "payment_method_id", "payment_plan_days", "plan_list_price",
    "actual_amount_paid", "is_auto_renew", "transaction_date",
    "membership_expire_date", "is_cancel", "_ord",
]
LOG_COUNTERS = ["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"]
MEMBER_FEATURES = ["city", "registered_via", "bd", "bd_missing", "tenure_days"]
TXN_FEATURES = [
    "transaction_count", "sum_amount_paid", "average_amount_paid",
    "last_is_cancel", "last_payment_method_id", "days_since_last_transaction",
    "days_to_expire",
]
LOG_FEATURES = ["active_days", "total_secs", "mean_daily_secs", "total_num_100", "days_since_last_log"]
FEATURES = MEMBER_FEATURES + TXN_FEATURES + LOG_FEATURES + ["has_transactions", "has_listening_logs"]
CATEGORICAL = ["city", "registered_via", "last_payment_method_id"]
NUMERIC = [name for name in FEATURES if name not in CATEGORICAL]
METRIC_COLUMNS = ["model", "auc", "log_loss", "improvement_vs_baseline", "beats_baseline"]
