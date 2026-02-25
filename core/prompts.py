from typing import Dict


class StrucutredSystemPrompt:
    prompt = """
        You are a quantitative commodities analyst assistant.

        Your task is NOT to predict prices.
        Your task is ONLY to explain how ONE specific feature influenced an already-generated aluminum price forecast.

        You must strictly follow these rules:

        1. Use ONLY the provided inputs.
        2. Do NOT invent macroeconomic events, news, or external facts.
        3. Base reasoning ONLY on:
        - the feature's recent time-series behaviour
        - the SHAP contribution value
        - the direction and structure of the forecast
        4. Treat SHAP score as model influence strength:
        - positive SHAP → upward pressure on forecast
        - negative SHAP → downward pressure on forecast
        - magnitude → strength of influence
        5. Explain relationships logically, not mathematically.
        6. Never claim causality beyond model influence.
        7. If signal is weak or unclear, explicitly say uncertainty is high.

        Your goal is to produce a linguistic explanation suitable for a downstream senior analyst model.

        Be precise, neutral, and analytical.
        Avoid storytelling or speculation.
        """


class StructuredUserPrompt:
    def __init__(
        self,
        ssd_date: str,
        lme_al_forecast: str,
        feature_business_name: str,
        feature_asset_class: str,
        feature_macro_role: str,
        feature_relation_to_lme_al: str,
        feature_timeseries: str,
        shap_score: Dict,
        ssd_lme_price: float,
    ):
        self.ssd_date = ssd_date
        self.lme_al_forecast = lme_al_forecast
        self.ssd_lme_price = ssd_lme_price
        self.feature_business_name = feature_business_name
        self.feature_asset_class = feature_asset_class
        self.feature_macro_role = feature_macro_role
        self.feature_relation_to_lme_al = feature_relation_to_lme_al
        self.feature_timeseries = feature_timeseries
        self.shap_score = shap_score

    def get_prompt(self):
        return f"""TASK:
                        Explain the role of a single feature in influencing the **LME Aluminum** forecast.

                        INPUTS:
                        1. Forecast Generated On: {self.ssd_date}
                        2. Forecast (next 5 days):
                        {self.lme_al_forecast}
                        3. Feature Name: {self.feature_business_name}
                        4. Feature Asset Class: {self.feature_asset_class}
                        5. Feature Macro Role: {self.feature_macro_role}
                        6. Feature Relation to LME Aluminum: {self.feature_relation_to_lme_al}
                        6. Feature Timeseries Data (previous 10 days):
                        {self.feature_timeseries}
                        7. Feature SHAP Contribution:
                        {self.shap_score}
                        6. Current Date LME Aluminum Spot Price: {self.ssd_lme_price} usd/mt

                        SHAP Score Information:
                        1. The SHAP scores are provided separately for EACH forecasted date.
                        2. For every forecasted date:
                        - A list of 10 SHAP values is given.
                        - These 10 values represent the feature's contribution across the previous 10 days.
                        - The forecasting model was trained using a 10-day historical window.
                        - Therefore, each SHAP sequence shows how the feature’s influence evolved over those 10 input days leading to that specific forecast.
                        3. Interpretation Guidelines:
                        - Positive SHAP values indicate upward pressure on the forecast.
                        - Negative SHAP values indicate downward pressure.
                        - Larger magnitude indicates stronger influence.
                        - Focus on overall direction, consistency, and recent influence patterns rather than individual numbers.

                        ANALYSIS INSTRUCTIONS:
                        1. Step 1 — Identify feature behaviour: Describe whether the feature shows rising, falling, stable, or volatile behaviour.
                        2. Step 2 — Interpret model influence: Explain how the SHAP value indicates the feature pushed the forecast upward, downward, or had limited impact.
                        3. Step 3 — Relate behaviour to forecast shape: Explain how the feature's recent movement aligns or conflicts with the forecast trajectory.
                        4. Step 4 — Confidence assessment: Classify influence strength as:
                            LOW / MODERATE / STRONG

                        OUTPUT FORMAT (STRICT JSON — DO NOT ADD EXTRA TEXT):

                        JSON(
                        "feature_name": "",
                        "feature_behavior": "",
                        "shap_interpretation": "",
                        "forecast_alignment": "",
                        "influence_strength": "",
                        "analyst_explanation": "")"""
