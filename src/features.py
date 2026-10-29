"""Feature engineering, shared by notebook 04 and future modelling notebooks.

Two parts:
- add_features(df): row-wise features. Uses only the row itself, so it is safe before the split.
- make_preprocessor(): everything that learns from data (imputation, one-hot, frequency
  encoding, KMeans clusters, scaling). Lives in a ColumnTransformer -> fitted on train only.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "Severity"

ROAD = ["Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
        "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal"]
BAD_WEATHER = {"Rain", "Heavy rain / Thunderstorm", "Snow / Ice", "Fog / Haze / Smoke"}
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
# "I-95 N", "US-101", "CA-99 N" (state route), "Interstate 5", "Pacific Hwy", "Hollywood Fwy", "NJ Tpke" ...
# Route prefixes are case-sensitive (so "Rd-1" etc. don't match); words are case-insensitive.
# ponytail: name-based heuristic; plain "Pkwy" left out (mixes highways and suburban roads)
HIGHWAY = (r"\b(?:I|US|[A-Z]{2})-\s?\d"
           r"|(?i:\b(?:Interstate|Highway|Hwy|Freeway|Fwy|Expressway|Expy|Turnpike|Tpke|Beltway|State Pkwy|State Parkway|State Route|Route|Rte)\b)")

# Candidate input columns, grouped by how the preprocessor treats them
NUM_ALL = ["Start_Lat", "Start_Lng", "Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)",
           "Wind_Speed(mph)", "Precipitation(in)", "precip_missing", "hour", "weekday", "month", "year",
           "is_weekend", "is_rush_hour", "is_night", "civil_night", "nautical_night", "astro_night",
           "bad_weather", "road_feature_count", "is_highway", *ROAD]
CAT = ["State", "Source", "weather_group", "Wind_Direction", "season"]
FREQ = ["City", "County"]
GEO = ["Start_Lat", "Start_Lng"]
CANDIDATES = list(dict.fromkeys(NUM_ALL + CAT + FREQ))

# Feature selection (evidence computed on the training split in notebook 04)
NEAR_CONSTANT = 0.001  # binary flag that is 1 in < 0.1% of training rows
REDUNDANT = 0.85       # |correlation| above this with a feature that is kept
DROP = {
    "Roundabout": "near-constant (1 in < 0.1% of rows); still counted in road_feature_count",
    "Bump": "near-constant (1 in < 0.1% of rows); still counted in road_feature_count",
    "Traffic_Calming": "near-constant (1 in < 0.1% of rows); still counted in road_feature_count",
    "is_night": "redundant: |r| = 0.89 with civil_night, which has higher mutual information",
    "nautical_night": "redundant: |r| = 0.88 with astro_night, which has higher mutual information",
}
NUM = [c for c in NUM_ALL if c not in DROP]
FEATURES = [c for c in CANDIDATES if c not in DROP]

# Ablation feature sets (report Section 5.6): set -> (name, inputs removed from FEATURES)
LOCATION = ["Start_Lat", "Start_Lng", "State", "City", "County"]
ABLATION_SETS = {
    "A": ("Full set", []),
    "B": ("Without provider", ["Source"]),
    "C": ("Without provider and year", ["Source", "year"]),
    "D": ("Without location", LOCATION),
    "E": ("Context only: time, light, weather, road", LOCATION + ["Source", "year"]),
}


def ablation_features(key):
    """Input columns of ablation set `key` ("A".."E")."""
    removed = set(ABLATION_SETS[key][1])
    return [c for c in FEATURES if c not in removed]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """clean.parquet -> all candidate features + target (selection to FEATURES happens in notebook 04)."""
    t = df["Start_Time"]
    night = {c: df[c].astype(str).eq("Night").astype("int8") for c in
             ["Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight"]}
    out = df.assign(
        hour=t.dt.hour.astype("int8"),
        weekday=t.dt.dayofweek.astype("int8"),  # 0 = Monday
        month=t.dt.month.astype("int8"),
        year=t.dt.year.astype("int16"),
        is_night=night["Sunrise_Sunset"],
        civil_night=night["Civil_Twilight"],
        nautical_night=night["Nautical_Twilight"],
        astro_night=night["Astronomical_Twilight"],
        weather_group=df["Weather_Group"].astype(str),
        road_feature_count=df[ROAD].sum(axis=1).astype("int8"),
        is_highway=df["Street"].str.contains(HIGHWAY, regex=True, na=False).astype("int8"),
    )
    out["is_weekend"] = (out["weekday"] >= 5).astype("int8")
    out["is_rush_hour"] = ((out["weekday"] < 5) & (out["hour"].between(7, 9) | out["hour"].between(16, 19))).astype("int8")
    out["season"] = out["month"].map(SEASONS)
    out["bad_weather"] = out["weather_group"].isin(BAD_WEATHER).astype("int8")
    cat_cols = [c for c in CAT + FREQ if c in out]
    out[cat_cols] = out[cat_cols].astype(str)  # plain strings: same dtype in every split
    return out[CANDIDATES + [TARGET]]


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category by its share of training rows; unseen categories -> 0."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.freqs_ = {c: X[c].value_counts(normalize=True) for c in X}
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return np.column_stack([X[c].map(self.freqs_[c]).astype(float).fillna(0.0) for c in self.freqs_])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_freq" for c in self.feature_names_in_], dtype=object)


class KMeansCluster(BaseEstimator, TransformerMixin):
    """Lat/Lng -> id of the nearest of `n_clusters` KMeans centres (regions / hotspots)."""

    def __init__(self, n_clusters=50, random_state=SEED):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        self.km_ = KMeans(self.n_clusters, n_init="auto", random_state=self.random_state).fit(np.asarray(X))
        return self

    def transform(self, X):
        return self.km_.predict(np.asarray(X)).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["location_cluster"], dtype=object)


def make_preprocessor(scale=False, use_source=True, n_clusters=50, drop=()):
    """ColumnTransformer for all models. scale=True only for Logistic Regression.
    use_source=False drops `Source`; drop=ABLATION_SETS[k][1] builds ablation set k.
    The location clusters are only built when both Start_Lat and Start_Lng are kept."""
    drop = set(drop) | (set() if use_source else {"Source"})
    num = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else [])
    onehot = dict(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    blocks = [
        ("num", make_pipeline(*num), [c for c in NUM if c not in drop]),
        ("cat", OneHotEncoder(**onehot), [c for c in CAT if c not in drop]),
    ]
    freq = [c for c in FREQ if c not in drop]
    if freq:
        blocks.append(("freq", make_pipeline(FrequencyEncoder(), *([StandardScaler()] if scale else [])), freq))
    if not drop & set(GEO):
        blocks.append(("cluster", make_pipeline(KMeansCluster(n_clusters), OneHotEncoder(**onehot)), GEO))
    return ColumnTransformer(blocks, verbose_feature_names_out=False)
