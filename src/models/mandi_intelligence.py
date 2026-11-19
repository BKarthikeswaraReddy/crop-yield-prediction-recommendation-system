"""
Mandi Price Intelligence Engine
Machine Learning Based Crop Yield Prediction & Recommendation System

Provides historical and recent modal-price statistics by crop, market, district, and date.
Handles missing matches honestly and transparently with tiered fallback:
  1. District-level recent modal price (exact APMC market aggregations in the district)
  2. State-level benchmark (if district has no active mandi data for that crop)
  3. National-level benchmark (if state has no active mandi data for that crop)
  4. Unavailable (explicitly reported; never fabricated)
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


class MandiIntelligence:

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            # Default to crop-production-predictor/data
            self.data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)

        self.district_summary_path = self.data_dir / "mandi_crop_district_summary.parquet"
        self.market_summary_path = self.data_dir / "mandi_crop_market_summary.parquet"
        self.state_summary_path = self.data_dir / "mandi_crop_state_summary.parquet"
        self.national_summary_path = self.data_dir / "mandi_crop_national_summary.parquet"
        self.detailed_market_path = self.data_dir / "mandi_market_detailed_summary.parquet"
        
        # Robust crop mapping path resolution (data/, docs/, or root)
        self.mapping_path = self.data_dir / "crop_name_mapping.csv"
        for candidate in [self.data_dir / "crop_name_mapping.csv",
                          self.data_dir.parent / "crop_name_mapping.csv",
                          self.data_dir.parent / "docs" / "crop_name_mapping.csv"]:
            if candidate.exists():
                self.mapping_path = candidate
                break

        self.df_district = None
        self.df_market = None
        self.df_state = None
        self.df_national = None
        self.df_detailed = None
        self.crop_aliases = {}

        self._load_datasets()

    def _load_datasets(self):
        """Load pre-aggregated mandi summaries and crop name mapping."""
        if self.district_summary_path.exists():
            self.df_district = pd.read_parquet(self.district_summary_path)
            self.df_district["State_Upper"] = self.df_district["State"].astype(str).str.strip().str.upper()
            self.df_district["District_Upper"] = self.df_district["District"].astype(str).str.strip().str.upper()
            self.df_district["Crop_Upper"] = self.df_district["production_crop"].astype(str).str.strip().str.upper()

        if self.market_summary_path.exists():
            self.df_market = pd.read_parquet(self.market_summary_path)
            self.df_market["State_Upper"] = self.df_market["State"].astype(str).str.strip().str.upper()
            self.df_market["District_Upper"] = self.df_market["District"].astype(str).str.strip().str.upper()
            self.df_market["Crop_Upper"] = self.df_market["production_crop"].astype(str).str.strip().str.upper()

        if self.state_summary_path.exists():
            self.df_state = pd.read_parquet(self.state_summary_path)
            self.df_state["State_Upper"] = self.df_state["State"].astype(str).str.strip().str.upper()
            self.df_state["Crop_Upper"] = self.df_state["production_crop"].astype(str).str.strip().str.upper()

        if self.national_summary_path.exists():
            self.df_national = pd.read_parquet(self.national_summary_path)
            self.df_national["Crop_Upper"] = self.df_national["production_crop"].astype(str).str.strip().str.upper()

        if self.detailed_market_path.exists():
            self.df_detailed = pd.read_parquet(self.detailed_market_path)
            self.df_detailed["State_Upper"] = self.df_detailed["State"].astype(str).str.strip().str.upper()
            self.df_detailed["District_Upper"] = self.df_detailed["District"].astype(str).str.strip().str.upper()
            self.df_detailed["Crop_Upper"] = self.df_detailed["production_crop"].astype(str).str.strip().str.upper()

        if self.mapping_path.exists():
            try:
                df_map = pd.read_csv(self.mapping_path)
                for _, row in df_map.iterrows():
                    p_crop = str(row.get("production_crop", "")).strip()
                    m_comm = str(row.get("mandi_commodity", "")).strip()
                    if p_crop and m_comm:
                        self.crop_aliases[p_crop.upper()] = p_crop
                        self.crop_aliases[m_comm.upper()] = p_crop
            except Exception as e:
                print(f"Warning loading crop mapping: {e}")

    def normalize_crop_name(self, crop: str) -> str:
        """Map common crop names/aliases to production crop nomenclature."""
        clean = str(crop).strip()
        upper = clean.upper()
        if upper in self.crop_aliases:
            return self.crop_aliases[upper]
        return clean

    def get_price_intelligence(self, state: str, district: str, crop: str) -> Dict[str, Any]:
        """
        Calculate historical and recent modal-price statistics by crop, district, and market.
        
        Returns a structured dictionary with price per quintal, confidence level,
        source level, market breakdowns, and honest disclaimers.
        """
        norm_crop = self.normalize_crop_name(crop)
        state_u = str(state).strip().upper()
        dist_u = str(district).strip().upper()
        crop_u = norm_crop.upper()

        # 1. District Level Match
        district_match = None
        if self.df_district is not None:
            sub = self.df_district[
                (self.df_district["State_Upper"] == state_u) &
                (self.df_district["District_Upper"] == dist_u) &
                (self.df_district["Crop_Upper"] == crop_u)
            ]
            if len(sub) > 0:
                district_match = sub.iloc[0]

        # Markets in district
        markets_info = []
        if self.df_market is not None:
            sub_m = self.df_market[
                (self.df_market["State_Upper"] == state_u) &
                (self.df_market["District_Upper"] == dist_u) &
                (self.df_market["Crop_Upper"] == crop_u)
            ]
            for _, r in sub_m.iterrows():
                markets_info.append({
                    "market": r["Market"],
                    "recent_avg_modal_price": float(r["avg_modal_price"]),
                    "min_price": float(r["min_modal_price"]),
                    "max_price": float(r["max_modal_price"]),
                    "latest_arrival_date": str(r["latest_date"]),
                    "transactions_recorded": int(r["record_count"]),
                })

        if district_match is not None:
            recent_avg = district_match["recent_avg_modal_price"]
            hist_avg = district_match["historical_avg_modal_price"]
            # Prioritize recent if available, else historical
            price = recent_avg if pd.notna(recent_avg) else hist_avg
            is_recent = pd.notna(recent_avg)

            return {
                "crop": norm_crop,
                "state": state,
                "district": district,
                "price_per_quintal": round(float(price), 2),
                "price_level": "district",
                "is_estimate": False,
                "is_state_benchmark": False,
                "is_national_benchmark": False,
                "confidence": "high" if is_recent else "medium",
                "recent_avg_modal_price": round(float(recent_avg), 2) if pd.notna(recent_avg) else None,
                "recent_min_modal_price": round(float(district_match["recent_min_modal_price"]), 2) if pd.notna(district_match["recent_min_modal_price"]) else None,
                "recent_max_modal_price": round(float(district_match["recent_max_modal_price"]), 2) if pd.notna(district_match["recent_max_modal_price"]) else None,
                "historical_avg_modal_price": round(float(hist_avg), 2),
                "historical_records": int(district_match["historical_records"]),
                "latest_year": int(district_match["latest_year"]),
                "markets_covered": len(markets_info),
                "markets": markets_info,
                "explanation": f"Verified APMC mandi transactions in {district} district ({district_match['historical_records']:,} historical arrivals, latest year: {int(district_match['latest_year'])}).",
                "disclaimer": "Prices reflect APMC wholesale modal arrivals, not guaranteed farmgate selling prices."
            }

        # 2. State Level Match (Fallback with clear labeling)
        state_match = None
        if self.df_state is not None:
            sub_s = self.df_state[
                (self.df_state["State_Upper"] == state_u) &
                (self.df_state["Crop_Upper"] == crop_u)
            ]
            if len(sub_s) > 0:
                state_match = sub_s.iloc[0]

        if state_match is not None:
            recent_avg = state_match["recent_avg_modal_price"]
            hist_avg = state_match["historical_avg_modal_price"]
            price = recent_avg if pd.notna(recent_avg) else hist_avg

            return {
                "crop": norm_crop,
                "state": state,
                "district": district,
                "price_per_quintal": round(float(price), 2),
                "price_level": "state",
                "is_estimate": True,
                "is_state_benchmark": True,
                "is_national_benchmark": False,
                "confidence": "medium",
                "recent_avg_modal_price": round(float(recent_avg), 2) if pd.notna(recent_avg) else None,
                "recent_min_modal_price": round(float(state_match["recent_min_modal_price"]), 2) if pd.notna(state_match["recent_min_modal_price"]) else None,
                "recent_max_modal_price": round(float(state_match["recent_max_modal_price"]), 2) if pd.notna(state_match["recent_max_modal_price"]) else None,
                "historical_avg_modal_price": round(float(hist_avg), 2),
                "historical_records": int(state_match["historical_records"]),
                "latest_year": int(state_match["latest_year"]),
                "markets_covered": 0,
                "markets": [],
                "explanation": f"No active mandi records in {district} district. Price is estimated from state-level benchmark across {state} ({state_match['historical_records']:,} transactions).",
                "disclaimer": "State-level wholesale benchmark. Actual local realization will vary depending on transportation and local market demand."
            }

        # 3. National Level Match (Fallback with clear labeling)
        nat_match = None
        if self.df_national is not None:
            sub_n = self.df_national[self.df_national["Crop_Upper"] == crop_u]
            if len(sub_n) > 0:
                nat_match = sub_n.iloc[0]

        if nat_match is not None:
            recent_avg = nat_match["recent_avg_modal_price"]
            hist_avg = nat_match["historical_avg_modal_price"]
            price = recent_avg if pd.notna(recent_avg) else hist_avg

            return {
                "crop": norm_crop,
                "state": state,
                "district": district,
                "price_per_quintal": round(float(price), 2),
                "price_level": "national",
                "is_estimate": True,
                "is_state_benchmark": False,
                "is_national_benchmark": True,
                "confidence": "low",
                "recent_avg_modal_price": round(float(recent_avg), 2) if pd.notna(recent_avg) else None,
                "recent_min_modal_price": round(float(nat_match["recent_min_modal_price"]), 2) if pd.notna(nat_match["recent_min_modal_price"]) else None,
                "recent_max_modal_price": round(float(nat_match["recent_max_modal_price"]), 2) if pd.notna(nat_match["recent_max_modal_price"]) else None,
                "historical_avg_modal_price": round(float(hist_avg), 2),
                "historical_records": int(nat_match["historical_records"]),
                "latest_year": int(nat_match["latest_year"]),
                "markets_covered": 0,
                "markets": [],
                "explanation": f"No state or district mandi records for {norm_crop} in {state}. Using all-India national wholesale benchmark ({nat_match['historical_records']:,} transactions).",
                "disclaimer": "National wholesale benchmark only. Highly sensitive to interstate freight and regional market availability."
            }

        # 4. Completely Unavailable
        return {
            "crop": norm_crop,
            "state": state,
            "district": district,
            "price_per_quintal": None,
            "price_level": "unavailable",
            "is_estimate": False,
            "is_state_benchmark": False,
            "is_national_benchmark": False,
            "confidence": "none",
            "recent_avg_modal_price": None,
            "recent_min_modal_price": None,
            "recent_max_modal_price": None,
            "historical_avg_modal_price": None,
            "historical_records": 0,
            "latest_year": None,
            "markets_covered": 0,
            "markets": [],
            "explanation": f"No APMC mandi arrival price data available for {norm_crop} in Agmarknet dataset.",
            "disclaimer": "Market price is unobserved; revenue and profitability cannot be computed without verified market pricing."
        }

    def get_district_markets(self, state: str, district: str, crop: str,
                             max_markets: int = 20) -> Dict[str, Any]:
        """
        Return individual APMC market records for a given state, district and crop.

        Queries ``mandi_market_detailed_summary.parquet`` which stores the
        latest observed price transaction for each (State, District, Market,
        production_crop) combination.

        Returns
        -------
        dict with keys:
            found        - bool, True when at least one market record exists
            state        - str
            district     - str
            crop         - str (normalised)
            markets      - list of dicts, one per APMC market
            total_found  - int, total matching markets (may exceed max_markets)
            note         - str, mandatory disclaimer
        """
        norm_crop = self.normalize_crop_name(crop)
        state_u = str(state).strip().upper()
        dist_u = str(district).strip().upper()
        crop_u = norm_crop.upper()

        DISCLAIMER = (
            "These are historical APMC wholesale arrival prices (Agmarknet dataset), "
            "not guaranteed future farmgate selling prices. Actual realization will "
            "depend on local demand, transportation costs, and market conditions."
        )

        if self.df_detailed is None:
            return {
                "found": False,
                "state": state,
                "district": district,
                "crop": norm_crop,
                "markets": [],
                "total_found": 0,
                "note": DISCLAIMER,
                "error": "Detailed market dataset not loaded."
            }

        sub = self.df_detailed[
            (self.df_detailed["State_Upper"] == state_u) &
            (self.df_detailed["District_Upper"] == dist_u) &
            (self.df_detailed["Crop_Upper"] == crop_u)
        ]

        total_found = len(sub)

        if total_found == 0:
            return {
                "found": False,
                "state": state,
                "district": district,
                "crop": norm_crop,
                "markets": [],
                "total_found": 0,
                "note": DISCLAIMER,
                "error": f"No APMC market records found for {norm_crop} in {district}, {state}."
            }

        sub_sorted = sub.sort_values("Modal_Price", ascending=False).head(max_markets)

        markets_list = []
        for _, row in sub_sorted.iterrows():
            def safe_float(val):
                try:
                    return round(float(val), 2) if pd.notna(val) else None
                except (TypeError, ValueError):
                    return None

            def safe_int(val):
                try:
                    return int(val) if pd.notna(val) else None
                except (TypeError, ValueError):
                    return None

            markets_list.append({
                "market": str(row.get("Market", "")) if pd.notna(row.get("Market")) else "Unknown",
                "commodity": str(row.get("Commodity", "")) if pd.notna(row.get("Commodity")) else norm_crop,
                "variety": str(row.get("Variety", "")) if pd.notna(row.get("Variety")) else "—",
                "grade": str(row.get("Grade", "")) if pd.notna(row.get("Grade")) else "—",
                "min_price": safe_float(row.get("Min_Price")),
                "max_price": safe_float(row.get("Max_Price")),
                "modal_price": safe_float(row.get("Modal_Price")),
                "arrival_date": str(row.get("Arrival_Date", "")) if pd.notna(row.get("Arrival_Date")) else "—",
                "source_year": safe_int(row.get("source_year")),
                "historical_records": safe_int(row.get("record_count")),
                "historical_avg_modal_price": safe_float(row.get("avg_modal_price")),
            })

        return {
            "found": True,
            "state": state,
            "district": district,
            "crop": norm_crop,
            "markets": markets_list,
            "total_found": total_found,
            "shown": len(markets_list),
            "note": DISCLAIMER,
        }

