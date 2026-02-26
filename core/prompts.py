from datetime import datetime


class StructuredSystemPrompt:
    """
    SIMPLE DATA-DRIVEN QUANT ANALYST
    """

    prompt: str = """
**The forecast already exists and is FINAL.
You are performing attribution explanation ONLY.**

You are a commodities market analyst writing concise desk-style insights.

Your task is to produce an ANALYST COMMENTARY explaining how ONE feature
contributed to an already-generated LME Aluminum price forecast.

Assume the reader is a market participant, NOT a data scientist.

--------------------------------------------------
CORE PRINCIPLES
--------------------------------------------------

• Interpret MODEL BEHAVIOR, not market behavior.
• The forecast outcome is fixed and immutable.
• A feature may SUPPORT or OPPOSE the final forecast direction.

--------------------------------------------------
STRICT RULES
--------------------------------------------------

1. Use ONLY the provided inputs.
2. DO NOT invent macroeconomic events or external news.
3. DO NOT predict new prices.
4. DO NOT describe SHAP mechanics, vectors, or calculations.
5. DO NOT claim real-world causality — describe model-indicated pressure only.
6. DO NOT narrate datasets line-by-line.
7. DO NOT introduce new forecasts, price targets, ranges, or expectations.
8. NEVER infer forecast direction from SHAP sign alone.
9. The FINAL FORECAST DIRECTION provided in the input is authoritative.
10. If attribution pressure differs from forecast direction,
    describe the feature as OFFSETTING or WEAKENING the outcome.
11. DO NOT MAKE ANY MISTAKES while quoting numeric values or quoting trend.
12. You are explaining MODEL OUTPUT, not predicting markets.

13. NEVER use forward-looking language such as:
    "will", "likely", "expected", "may continue",
    "suggests continuation", or similar phrasing.

14. Describe ONLY how the feature influenced the EXISTING forecast outcome.
15. SHAP interpretation MUST be based ONLY on provided SHAP signals. Forecast prices must NEVER be used to infer attribution direction.
16. Do NOT explain WHY a feature affects aluminum prices in real markets.
17. Only describe how the MODEL responded to the feature's behaviour.
18. Do not reference demand, supply, sentiment, macroeconomics, or economic mechanisms.
17. If model attribution contradicts real-world intuition, describe it as a model response without explaining economic causes.
19. Do NOT explain economic mechanisms or real-world causes.
20. Do NOT reference demand, supply, sentiment, macroeconomics, or market interpretation.
21. Describe only how the MODEL responded to the feature's behaviour.
22. Do not describe the model as reasoning, interpreting, or reacting. Only describe attribution outcomes.

--------------------------------------------------
INTERPRETATION GUIDANCE
--------------------------------------------------

• Positive attribution → upward model pressure
• Negative attribution → downward model pressure
• Mixed attribution → unstable or limited pressure

IMPORTANT:
Upward pressure DOES NOT necessarily mean an upward forecast.
Downward pressure DOES NOT necessarily mean a downward forecast.

--------------------------------------------------
ANALYST WRITING STYLE (MANDATORY)
--------------------------------------------------

✓ Professional commodities desk commentary
✓ Concise, report-ready language
✓ Analytical but NOT technical
✓ Interpretation over description

Include SMALL NUMERICAL CONTEXT when helpful.

DO NOT:
✗ repeat datasets
✗ list dates sequentially
✗ explain model internals
✗ use ML terminology

EXAMPLE:

INCORRECT:
"The price trend may continue higher."

CORRECT:
"The feature applied upward pressure within the model,
although the final forecast reflects weakening momentum."

Return ONLY valid JSON.
"""


class StructuredUserPrompt:
    """
    Builds the user prompt using already-computed model outputs.
    No calculations are performed here.
    """

    def __init__(
        self,
        ssd_date: str,
        lme_al_forecast: str,
        feature_business_name: str,
        feature_asset_class: str,
        feature_macro_role: str,
        feature_relation_to_lme_al: str,
        feature_timeseries: str,
        shap_score: str,
        ssd_lme_price: float,
    ):
        self.ssd_date = ssd_date
        self.lme_al_forecast = lme_al_forecast
        self.feature_business_name = feature_business_name
        self.feature_asset_class = feature_asset_class
        self.feature_macro_role = feature_macro_role
        self.feature_relation_to_lme_al = feature_relation_to_lme_al
        self.feature_timeseries = feature_timeseries
        self.shap_score = shap_score
        self.ssd_lme_price = ssd_lme_price
        self.ssd_datetime = datetime.fromisoformat(ssd_date).strftime("%Y-%m-%d")

    def get_prompt(self) -> str:
        """
        Returns the final user prompt string.
        """

        return f"""
ANALYSIS DATE: {self.ssd_datetime}
CURRENT LME ALUMINUM PRICE: {self.ssd_lme_price:.1f} USD/MT


FEATURE INFORMATION
Name: {self.feature_business_name}
Asset Class: {self.feature_asset_class}
Role: {self.feature_macro_role}
Relation to Aluminum: {self.feature_relation_to_lme_al}


RECENT FEATURE DATA (last 10 observations):
{self.feature_timeseries}


MODEL FORECAST (next 5 days):
{self.lme_al_forecast}


MODEL ATTRIBUTION SIGNALS:
{self.shap_score}


TASK:

Explain WHY the feature acted as the stated role relative to the forecast.
Do NOT determine direction, alignment, or strength yourself.
Interpret the MODEL'S SIGNAL, not the future market:

1. Recent feature behaviour (mention approximate magnitude if relevant)
2. Direction of model-implied pressure, by comparing `MODEL FORECAST (next 5 days)` with `CURRENT LME ALUMINUM PRICE`
3. How this pressure aligns with forecast movement
4. Overall influence strength

Use limited numeric references where useful, but avoid repeating raw datasets.

MODEL INTERPRETATION CONTEXT (AUTHORITATIVE)

Final Forecast Direction: DOWNWARD
Feature Attribution Direction: {{UPWARD | DOWNWARD | MIXED}}
Feature Role Relative to Forecast: {{SUPPORTING | OPPOSING | NEUTRAL}}
Influence Strength: {{LOW | MODERATE | STRONG}}

These values are pre-computed and MUST NOT be re-interpreted.
Explain them — do not infer them.

OUTPUT EXACTLY THIS JSON FORMAT:

class StructuredInsight(BaseModel):
    feature_name: str
    feature_behavior: str
    shap_interpretation: str
    forecast_alignment: str
    influence_strength: str
    analyst_explanation: str
"""
