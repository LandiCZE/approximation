"""Loading and harmonising the Czech Kraj / Okres datasets.

Both loaders return a :class:`Dataset` with the same shape, so every method and
evaluation function works on education and unemployment alike:

* ``okres`` - one row per district: ``kraj``, feature columns, ``weight``, ``y_true``
* ``kraj``  - one row per region: feature columns, ``weight``, ``y`` (the official value)

Features are always *shares* (0-1), never absolute counts, so that a model fitted
on 14 large regions can be applied to 77 much smaller districts without
extrapolating in size.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EDUCATION_DATA = REPO_ROOT / "education" / "DATA"
UNEMPLOYMENT_DATA = REPO_ROOT / "unemployment" / "DATA"


def normalize_name(name) -> str:
    """'Žďár nad Sázavou' -> 'zdarnadsazavou', 'Praha-východ' -> 'praha-vychod'."""
    text = unicodedata.normalize("NFKD", str(name).strip().lower())
    text = text.encode("ascii", errors="ignore").decode("ascii")
    return text.replace(" ", "")


# Spellings in the source files that differ from the canonical normalised name.
NAME_ALIASES = {
    "hlavnimestoprahavychod": "praha-vychod",
    "hlavnimestoprahazapad": "praha-zapad",
}


def canonical(name) -> str:
    key = normalize_name(name)
    return NAME_ALIASES.get(key, key)


OKRES_TO_KRAJ = {
    "hlavnimestopraha": "hlavnimestopraha",
    # Stredocesky
    "benesov": "stredoceskykraj", "beroun": "stredoceskykraj", "kladno": "stredoceskykraj",
    "kolin": "stredoceskykraj", "kutnahora": "stredoceskykraj", "melnik": "stredoceskykraj",
    "mladaboleslav": "stredoceskykraj", "nymburk": "stredoceskykraj",
    "praha-vychod": "stredoceskykraj", "praha-zapad": "stredoceskykraj",
    "pribram": "stredoceskykraj", "rakovnik": "stredoceskykraj",
    # Jihocesky
    "ceskebudejovice": "jihoceskykraj", "ceskykrumlov": "jihoceskykraj",
    "jindrichuvhradec": "jihoceskykraj", "pisek": "jihoceskykraj",
    "prachatice": "jihoceskykraj", "strakonice": "jihoceskykraj", "tabor": "jihoceskykraj",
    # Plzensky
    "domazlice": "plzenskykraj", "klatovy": "plzenskykraj", "plzen-jih": "plzenskykraj",
    "plzen-mesto": "plzenskykraj", "plzen-sever": "plzenskykraj",
    "rokycany": "plzenskykraj", "tachov": "plzenskykraj",
    # Karlovarsky
    "cheb": "karlovarskykraj", "karlovyvary": "karlovarskykraj", "sokolov": "karlovarskykraj",
    # Ustecky
    "decin": "usteckykraj", "chomutov": "usteckykraj", "litomerice": "usteckykraj",
    "louny": "usteckykraj", "most": "usteckykraj", "teplice": "usteckykraj",
    "ustinadlabem": "usteckykraj",
    # Liberecky
    "ceskalipa": "libereckykraj", "jablonecnadnisou": "libereckykraj",
    "liberec": "libereckykraj", "semily": "libereckykraj",
    # Kralovehradecky
    "hradeckralove": "kralovehradeckykraj", "jicin": "kralovehradeckykraj",
    "nachod": "kralovehradeckykraj", "rychnovnadkneznou": "kralovehradeckykraj",
    "trutnov": "kralovehradeckykraj",
    # Pardubicky
    "chrudim": "pardubickykraj", "pardubice": "pardubickykraj",
    "svitavy": "pardubickykraj", "ustinadorlici": "pardubickykraj",
    # Vysocina
    "havlickuvbrod": "krajvysocina", "jihlava": "krajvysocina", "pelhrimov": "krajvysocina",
    "trebic": "krajvysocina", "zdarnadsazavou": "krajvysocina",
    # Jihomoravsky
    "blansko": "jihomoravskykraj", "brno-mesto": "jihomoravskykraj",
    "brno-venkov": "jihomoravskykraj", "breclav": "jihomoravskykraj",
    "hodonin": "jihomoravskykraj", "vyskov": "jihomoravskykraj", "znojmo": "jihomoravskykraj",
    # Olomoucky
    "jesenik": "olomouckykraj", "olomouc": "olomouckykraj", "prostejov": "olomouckykraj",
    "prerov": "olomouckykraj", "sumperk": "olomouckykraj",
    # Zlinsky
    "kromeriz": "zlinskykraj", "uherskehradiste": "zlinskykraj",
    "vsetin": "zlinskykraj", "zlin": "zlinskykraj",
    # Moravskoslezsky
    "bruntal": "moravskoslezskykraj", "frydek-mistek": "moravskoslezskykraj",
    "karvina": "moravskoslezskykraj", "novyjicin": "moravskoslezskykraj",
    "opava": "moravskoslezskykraj", "ostrava-mesto": "moravskoslezskykraj",
}
KRAJE = sorted(set(OKRES_TO_KRAJ.values()))


@dataclass
class Dataset:
    name: str
    target_label: str
    okres: pd.DataFrame
    kraj: pd.DataFrame
    features: list[str]

    @property
    def eval_mask(self) -> pd.Series:
        """Districts used for evaluation.

        Prague is both a Kraj and its only Okres, so any anchored method gets it
        exactly right for free; it is excluded from every metric.
        """
        sizes = self.okres["kraj"].map(self.okres["kraj"].value_counts())
        return sizes > 1


def _check_complete(okres: pd.DataFrame, what: str) -> None:
    missing = sorted(set(OKRES_TO_KRAJ) - set(okres.index))
    extra = sorted(set(okres.index) - set(OKRES_TO_KRAJ))
    if missing or extra:
        raise ValueError(f"{what}: missing okresy {missing}, unknown okresy {extra}")


# --------------------------------------------------------------------------- #
# Households (shared: education features, population weights for both topics)
# --------------------------------------------------------------------------- #
HOUSEHOLD_COLUMNS = ["region", "total", "one_family", "multi_family",
                     "single_person", "multi_person_nonfamily"]
HOUSEHOLD_SHARES = ["share_one_family", "share_multi_family",
                    "share_single_person", "share_multi_person_nonfamily"]


def load_households(data_dir: Path = EDUCATION_DATA) -> pd.DataFrame:
    """Household counts per Okres (Census 2021), indexed by canonical Okres name."""
    raw = pd.read_excel(data_dir / "domacnosti.xlsx")
    raw.columns = HOUSEHOLD_COLUMNS
    raw["region"] = raw["region"].map(canonical)
    okres = raw[raw["region"].isin(OKRES_TO_KRAJ.keys())].set_index("region")
    _check_complete(okres, "domacnosti.xlsx")
    return okres


def _households_with_shares(households: pd.DataFrame) -> pd.DataFrame:
    out = households.copy()
    parts = ["one_family", "multi_family", "single_person", "multi_person_nonfamily"]
    for part, share in zip(parts, HOUSEHOLD_SHARES):
        out[share] = out[part] / out["total"]
    return out


def _aggregate_to_kraj(okres: pd.DataFrame, count_cols: list[str]) -> pd.DataFrame:
    return okres.groupby("kraj")[count_cols].sum()


# --------------------------------------------------------------------------- #
# Education
# --------------------------------------------------------------------------- #
def load_education(data_dir: Path = EDUCATION_DATA) -> Dataset:
    """Share of population with tertiary education (%), Census 2021.

    Features: household-composition shares. Weight: number of households.
    The 'multi-person non-family' share is left out of the model features
    because the four shares sum to one.
    """
    hh = _households_with_shares(load_households(data_dir))
    hh["kraj"] = hh.index.map(OKRES_TO_KRAJ)

    okres_y = pd.read_excel(data_dir / "vzdelani_with_percentage_only.xlsx")
    okres_y.columns = ["region", "y_true"]
    okres_y["region"] = okres_y["region"].map(canonical)
    okres_y = okres_y.set_index("region")["y_true"]

    kraj_y = pd.read_excel(data_dir / "krajske_vzdelani_percentage_only.xlsx")
    kraj_y.columns = ["region", "y"]
    kraj_y["region"] = kraj_y["region"].map(canonical)
    kraj_y = kraj_y.set_index("region")["y"]

    okres = hh.assign(weight=hh["total"], y_true=okres_y.reindex(hh.index))
    counts = ["total", "one_family", "multi_family", "single_person", "multi_person_nonfamily"]
    kraj = _households_with_shares(_aggregate_to_kraj(okres, counts))
    kraj["weight"] = kraj["total"]
    kraj["y"] = kraj_y.reindex(kraj.index)

    features = HOUSEHOLD_SHARES[:3]
    _validate(okres, kraj, features, "education")
    return Dataset("education", "% with tertiary education",
                   okres[["kraj", *features, "weight", "y_true"]],
                   kraj[[*features, "weight", "y"]], features)


# --------------------------------------------------------------------------- #
# Unemployment
# --------------------------------------------------------------------------- #
# The voting file holds the municipal elections: ~390 local lists and
# coalitions. They are grouped into political blocs by keyword, checked in
# this order (first match wins, e.g. 'Sdruzeni SPD, Trikolora, NK' -> spd).
VOTING_BLOCS = [
    ("ano", ["ano 2011"]),
    ("spd", ["spd", "trikolor", "pes", "svobodn", "prisaha"]),
    ("left", ["kscm", "komunist", "socdem", "socialni demokracie", "cssd", "levice"]),
    ("spolu", ["ods", "top 09", "kdu", "obcanska demokraticka"]),
    ("pirates_stan", ["pirat", "stan", "starostove"]),
    ("independents", ["nezavisl", "nestranic", "snk", "nk"]),
]
BLOC_NAMES = [name for name, _ in VOTING_BLOCS] + ["other"]


def party_to_bloc(party: str) -> str:
    text = unicodedata.normalize("NFKD", party.lower()).encode("ascii", "ignore").decode()
    tokens = text.replace(",", " ").replace("(", " ").replace(")", " ").split()
    for bloc, keys in VOTING_BLOCS:
        for key in keys:
            if (" " in key and key in text) or any(t.startswith(key) for t in tokens):
                return bloc
    return "other"


def load_voting_blocs(data_dir: Path = UNEMPLOYMENT_DATA) -> pd.DataFrame:
    """Votes per Okres and bloc (absolute), indexed by canonical Okres name."""
    raw = pd.read_excel(data_dir / "combined_with_kraj_okres.xlsx")
    raw["okres"] = raw["Okres"].map(canonical)
    raw["votes"] = (raw["Hlasy abs."].astype(str)
                    .str.replace("\xa0", "", regex=False)
                    .str.replace(" ", "", regex=False)
                    .astype(float))
    raw["bloc"] = raw["Volební strana"].map(party_to_bloc)
    votes = raw.pivot_table(index="okres", columns="bloc", values="votes",
                            aggfunc="sum", fill_value=0.0)
    votes = votes.reindex(columns=BLOC_NAMES, fill_value=0.0)
    _check_complete(votes, "combined_with_kraj_okres.xlsx")
    return votes


def load_unemployment(data_dir: Path = UNEMPLOYMENT_DATA,
                      households_dir: Path = EDUCATION_DATA) -> Dataset:
    """Share of unemployed persons (%).

    Features: vote shares of political blocs (municipal elections).
    Weight: number of households (Census 2021) as a population proxy - vote
    counts cannot serve as weights because in municipal elections each voter
    casts as many votes as there are seats (Prague alone has ~36M votes).
    The 'other' bloc is left out of the model features (shares sum to one).
    """
    votes = load_voting_blocs(data_dir)
    okres = votes.div(votes.sum(axis=1), axis=0).add_prefix("vote_")
    okres["kraj"] = okres.index.map(OKRES_TO_KRAJ)
    okres["weight"] = load_households(households_dir)["total"].reindex(okres.index)

    okres_y = pd.read_csv(data_dir / "Formatted_Okres_Data.csv")
    okres_y["Okres"] = okres_y["Okres"].map(canonical)
    okres["y_true"] = okres_y.set_index("Okres")["Podíl nezaměstnaných osob [%]"].reindex(okres.index)

    kraj_y = pd.read_csv(data_dir / "nezamestnanost.csv")
    kraj_y["Kraj"] = kraj_y["Kraj"].map(canonical)
    kraj_y = kraj_y.set_index("Kraj")["Podíl nezaměstnaných osob [%]"]

    votes["kraj"] = votes.index.map(OKRES_TO_KRAJ)
    kraj_votes = votes.groupby("kraj")[BLOC_NAMES].sum()
    kraj = kraj_votes.div(kraj_votes.sum(axis=1), axis=0).add_prefix("vote_")
    kraj["weight"] = okres.groupby("kraj")["weight"].sum()
    kraj["y"] = kraj_y.reindex(kraj.index)

    features = [f"vote_{b}" for b in BLOC_NAMES if b != "other"]
    _validate(okres, kraj, features, "unemployment")
    return Dataset("unemployment", "% unemployed",
                   okres[["kraj", *features, "weight", "y_true"]],
                   kraj[[*features, "weight", "y"]], features)


def _validate(okres: pd.DataFrame, kraj: pd.DataFrame, features: list[str], what: str) -> None:
    for frame, cols, label in [(okres, [*features, "weight", "y_true"], "okres"),
                               (kraj, [*features, "weight", "y"], "kraj")]:
        bad = frame[cols].isna().any(axis=1)
        if bad.any():
            raise ValueError(f"{what}: missing values for {label} rows {list(frame.index[bad])}")
    if sorted(kraj.index) != KRAJE:
        raise ValueError(f"{what}: unexpected kraje {sorted(kraj.index)}")
