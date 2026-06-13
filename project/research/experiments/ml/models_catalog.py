"""
Top 50 Models Catalog

This module implements a comprehensive catalog of the top 50 models
required for quantitative trading research and implementation.

Based on the Quant Research Intelligence System document.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCategory(Enum):
    """Model category types."""
    ML = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    TIME_SERIES = "time_series"
    OPTIMIZATION = "optimization"
    RISK = "risk"
    REGIME = "regime"
    MICROSTRUCTURE = "microstructure"
    EXECUTION = "execution"
    FACTOR = "factor"
    SIMULATION = "simulation"


class ModelComplexity(Enum):
    """Model complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class Model:
    """Model definition."""
    id: str
    name: str
    category: ModelCategory
    complexity: ModelComplexity
    description: str
    use_case: str
    data_requirements: List[str]
    computational_requirements: str


class ModelsCatalog:
    """
    Catalog of top 50 models.
    
    This class provides a comprehensive catalog of models
    with their characteristics and implementation requirements.
    """
    
    def __init__(self):
        """Initialize models catalog."""
        self.models: Dict[str, Model] = {}
        self._initialize_catalog()
        
        logger.info(f"ModelsCatalog initialized with {len(self.models)} models")
    
    def _initialize_catalog(self) -> None:
        """Initialize the catalog with top 50 models."""
        
        # Machine Learning Models
        self.models['linear_regression'] = Model(
            id='linear_regression',
            name='Linear Regression',
            category=ModelCategory.ML,
            complexity=ModelComplexity.SIMPLE,
            description='Linear regression for baseline predictions',
            use_case='Alpha prediction, feature importance',
            data_requirements=['Features', 'targets'],
            computational_requirements='Low'
        )
        
        self.models['ridge_regression'] = Model(
            id='ridge_regression',
            name='Ridge Regression',
            category=ModelCategory.ML,
            complexity=ModelComplexity.SIMPLE,
            description='L2-regularized linear regression',
            use_case='Regularized alpha prediction',
            data_requirements=['Features', 'targets'],
            computational_requirements='Low'
        )
        
        self.models['lasso_regression'] = Model(
            id='lasso_regression',
            name='Lasso Regression',
            category=ModelCategory.ML,
            complexity=ModelComplexity.SIMPLE,
            description='L1-regularized linear regression for feature selection',
            use_case='Feature selection, sparse models',
            data_requirements=['Features', 'targets'],
            computational_requirements='Low'
        )
        
        self.models['elastic_net'] = Model(
            id='elastic_net',
            name='Elastic Net',
            category=ModelCategory.ML,
            complexity=ModelComplexity.SIMPLE,
            description='Combined L1 and L2 regularization',
            use_case='Regularized alpha prediction',
            data_requirements=['Features', 'targets'],
            computational_requirements='Low'
        )
        
        self.models['random_forest'] = Model(
            id='random_forest',
            name='Random Forest',
            category=ModelCategory.ML,
            complexity=ModelComplexity.MEDIUM,
            description='Ensemble of decision trees',
            use_case='Alpha prediction, feature importance',
            data_requirements=['Features', 'targets'],
            computational_requirements='Medium'
        )
        
        self.models['gradient_boosting'] = Model(
            id='gradient_boosting',
            name='Gradient Boosting',
            category=ModelCategory.ML,
            complexity=ModelComplexity.MEDIUM,
            description='Gradient boosting machines',
            use_case='Alpha prediction',
            data_requirements=['Features', 'targets'],
            computational_requirements='Medium'
        )
        
        self.models['xgboost'] = Model(
            id='xgboost',
            name='XGBoost',
            category=ModelCategory.ML,
            complexity=ModelComplexity.MEDIUM,
            description='Extreme gradient boosting',
            use_case='Alpha prediction, competition',
            data_requirements=['Features', 'targets'],
            computational_requirements='Medium'
        )
        
        self.models['lightgbm'] = Model(
            id='lightgbm',
            name='LightGBM',
            category=ModelCategory.ML,
            complexity=ModelComplexity.MEDIUM,
            description='Light gradient boosting machine',
            use_case='Alpha prediction, large datasets',
            data_requirements=['Features', 'targets'],
            computational_requirements='Medium'
        )
        
        self.models['catboost'] = Model(
            id='catboost',
            name='CatBoost',
            category=ModelCategory.ML,
            complexity=ModelComplexity.MEDIUM,
            description='Categorical boosting',
            use_case='Alpha prediction with categorical features',
            data_requirements=['Features', 'targets'],
            computational_requirements='Medium'
        )
        
        self.models['svr'] = Model(
            id='svr',
            name='Support Vector Regression',
            category=ModelCategory.ML,
            complexity=ModelComplexity.MEDIUM,
            description='Support vector machines for regression',
            use_case='Alpha prediction',
            data_requirements=['Features', 'targets'],
            computational_requirements='Medium'
        )
        
        # Deep Learning Models
        self.models['mlp'] = Model(
            id='mlp',
            name='Multi-Layer Perceptron',
            category=ModelCategory.DEEP_LEARNING,
            complexity=ModelComplexity.MEDIUM,
            description='Feedforward neural network',
            use_case='Alpha prediction',
            data_requirements=['Features', 'targets'],
            computational_requirements='Medium (GPU recommended)'
        )
        
        self.models['lstm'] = Model(
            id='lstm',
            name='LSTM',
            category=ModelCategory.DEEP_LEARNING,
            complexity=ModelComplexity.COMPLEX,
            description='Long Short-Term Memory network',
            use_case='Time series prediction',
            data_requirements=['Time series data'],
            computational_requirements='High (GPU required)'
        )
        
        self.models['gru'] = Model(
            id='gru',
            name='GRU',
            category=ModelCategory.DEEP_LEARNING,
            complexity=ModelComplexity.COMPLEX,
            description='Gated Recurrent Unit network',
            use_case='Time series prediction',
            data_requirements=['Time series data'],
            computational_requirements='High (GPU required)'
        )
        
        self.models['transformer'] = Model(
            id='transformer',
            name='Transformer',
            category=ModelCategory.DEEP_LEARNING,
            complexity=ModelComplexity.VERY_COMPLEX,
            description='Self-attention based model',
            use_case='Sequence modeling, attention',
            data_requirements=['Sequence data'],
            computational_requirements='Very High (GPU required)'
        )
        
        self.models['autoencoder'] = Model(
            id='autoencoder',
            name='Autoencoder',
            category=ModelCategory.DEEP_LEARNING,
            complexity=ModelComplexity.COMPLEX,
            description='Neural network for dimensionality reduction',
            use_case='Feature extraction, anomaly detection',
            data_requirements=['Features'],
            computational_requirements='Medium (GPU recommended)'
        )
        
        # Time Series Models
        self.models['arima'] = Model(
            id='arima',
            name='ARIMA',
            category=ModelCategory.TIME_SERIES,
            complexity=ModelComplexity.MEDIUM,
            description='AutoRegressive Integrated Moving Average',
            use_case='Time series forecasting',
            data_requirements=['Time series data'],
            computational_requirements='Low'
        )
        
        self.models['sarima'] = Model(
            id='sarima',
            name='SARIMA',
            category=ModelCategory.TIME_SERIES,
            complexity=ModelComplexity.MEDIUM,
            description='Seasonal ARIMA',
            use_case='Seasonal time series forecasting',
            data_requirements=['Time series data'],
            computational_requirements='Low'
        )
        
        self.models['garch'] = Model(
            id='garch',
            name='GARCH',
            category=ModelCategory.TIME_SERIES,
            complexity=ModelComplexity.MEDIUM,
            description='Generalized Autoregressive Conditional Heteroskedasticity',
            use_case='Volatility modeling',
            data_requirements=['Returns data'],
            computational_requirements='Low'
        )
        
        self.models['egarch'] = Model(
            id='egarch',
            name='EGARCH',
            category=ModelCategory.TIME_SERIES,
            complexity=ModelComplexity.MEDIUM,
            description='Exponential GARCH',
            use_case='Volatility modeling with asymmetry',
            data_requirements=['Returns data'],
            computational_requirements='Low'
        )
        
        self.models['prophet'] = Model(
            id='prophet',
            name='Prophet',
            category=ModelCategory.TIME_SERIES,
            complexity=ModelComplexity.MEDIUM,
            description='Facebook Prophet for forecasting',
            use_case='Time series forecasting with seasonality',
            data_requirements=['Time series data'],
            computational_requirements='Low'
        )
        
        # Optimization Models
        self.models['mean_variance'] = Model(
            id='mean_variance',
            name='Mean-Variance Optimization',
            category=ModelCategory.OPTIMIZATION,
            complexity=ModelComplexity.SIMPLE,
            description='Markowitz mean-variance optimization',
            use_case='Portfolio optimization',
            data_requirements=['Expected returns', 'covariance matrix'],
            computational_requirements='Low'
        )
        
        self.models['black_litterman'] = Model(
            id='black_litterman',
            name='Black-Litterman',
            category=ModelCategory.OPTIMIZATION,
            complexity=ModelComplexity.MEDIUM,
            description='Black-Litterman portfolio optimization',
            use_case='Portfolio optimization with views',
            data_requirements=['Expected returns', 'covariance', 'views'],
            computational_requirements='Medium'
        )
        
        self.models['risk_parity'] = Model(
            id='risk_parity',
            name='Risk Parity',
            category=ModelCategory.OPTIMIZATION,
            complexity=ModelComplexity.SIMPLE,
            description='Risk parity portfolio construction',
            use_case='Portfolio optimization',
            data_requirements=['Covariance matrix'],
            computational_requirements='Low'
        )
        
        self.models['hierarchical_risk_parity'] = Model(
            id='hierarchical_risk_parity',
            name='Hierarchical Risk Parity',
            category=ModelCategory.OPTIMIZATION,
            complexity=ModelComplexity.COMPLEX,
            description='HRP with clustering',
            use_case='Portfolio optimization',
            data_requirements=['Returns data'],
            computational_requirements='Medium'
        )
        
        # Risk Models
        self.models['var'] = Model(
            id='var',
            name='Value at Risk (VaR)',
            category=ModelCategory.RISK,
            complexity=ModelComplexity.SIMPLE,
            description='Value at Risk calculation',
            use_case='Risk measurement',
            data_requirements=['Returns data'],
            computational_requirements='Low'
        )
        
        self.models['cvar'] = Model(
            id='cvar',
            name='Conditional VaR (CVaR)',
            category=ModelCategory.RISK,
            complexity=ModelComplexity.SIMPLE,
            description='Conditional Value at Risk',
            use_case='Risk measurement',
            data_requirements=['Returns data'],
            computational_requirements='Low'
        )
        
        self.models['expected_shortfall'] = Model(
            id='expected_shortfall',
            name='Expected Shortfall',
            category=ModelCategory.RISK,
            complexity=ModelComplexity.SIMPLE,
            description='Expected shortfall calculation',
            use_case='Risk measurement',
            data_requirements=['Returns data'],
            computational_requirements='Low'
        )
        
        self.models['cornish_fisher_var'] = Model(
            id='cornish_fisher_var',
            name='Cornish-Fisher VaR',
            category=ModelCategory.RISK,
            complexity=ModelComplexity.MEDIUM,
            description='Cornish-Fisher expansion for VaR',
            use_case='Risk measurement with non-normality',
            data_requirements=['Returns data'],
            computational_requirements='Low'
        )
        
        # Regime Models
        self.models['hmm'] = Model(
            id='hmm',
            name='Hidden Markov Model',
            category=ModelCategory.REGIME,
            complexity=ModelComplexity.MEDIUM,
            description='Hidden Markov Model for regime detection',
            use_case='Regime detection',
            data_requirements=['Returns data'],
            computational_requirements='Medium'
        )
        
        self.models['change_point'] = Model(
            id='change_point',
            name='Change Point Detection',
            category=ModelCategory.REGIME,
            complexity=ModelComplexity.MEDIUM,
            description='Change point detection algorithms',
            use_case='Regime change detection',
            data_requirements=['Time series data'],
            computational_requirements='Medium'
        )
        
        self.models['kalman_filter'] = Model(
            id='kalman_filter',
            name='Kalman Filter',
            category=ModelCategory.TIME_SERIES,
            complexity=ModelComplexity.MEDIUM,
            description='Kalman filter for state estimation',
            use_case='Time series filtering',
            data_requirements=['Time series data'],
            computational_requirements='Medium'
        )
        
        # Microstructure Models
        self.models['hawkes'] = Model(
            id='hawkes',
            name='Hawkes Process',
            category=ModelCategory.MICROSTRUCTURE,
            complexity=ModelComplexity.COMPLEX,
            description='Hawkes process for order flow clustering',
            use_case='Order flow modeling',
            data_requirements=['Order flow data'],
            computational_requirements='Medium'
        )
        
        self.models['order_book_model'] = Model(
            id='order_book_model',
            name='Order Book Model',
            category=ModelCategory.MICROSTRUCTURE,
            complexity=ModelComplexity.VERY_COMPLEX,
            description='Order book dynamics model',
            use_case='Order book prediction',
            data_requirements=['Order book data'],
            computational_requirements='High'
        )
        
        # Execution Models
        self.models['almgren_chriss'] = Model(
            id='almgren_chriss',
            name='Almgren-Chriss',
            category=ModelCategory.EXECUTION,
            complexity=ModelComplexity.MEDIUM,
            description='Almgren-Chriss optimal execution',
            use_case='Optimal execution',
            data_requirements=['Order data', 'market impact'],
            computational_requirements='Medium'
        )
        
        self.models['market_impact'] = Model(
            id='market_impact',
            name='Market Impact Model',
            category=ModelCategory.EXECUTION,
            complexity=ModelComplexity.MEDIUM,
            description='Market impact model (square root)',
            use_case='Execution cost estimation',
            data_requirements=['Trade data'],
            computational_requirements='Low'
        )
        
        # Factor Models
        self.models['capm'] = Model(
            id='capm',
            name='CAPM',
            category=ModelCategory.FACTOR,
            complexity=ModelComplexity.SIMPLE,
            description='Capital Asset Pricing Model',
            use_case='Factor modeling',
            data_requirements=['Returns', 'market returns'],
            computational_requirements='Low'
        )
        
        self.models['fama_french_3'] = Model(
            id='fama_french_3',
            name='Fama-French 3-Factor',
            category=ModelCategory.FACTOR,
            complexity=ModelComplexity.SIMPLE,
            description='Fama-French 3-factor model',
            use_case='Factor modeling',
            data_requirements=['Returns', 'factor data'],
            computational_requirements='Low'
        )
        
        self.models['fama_french_5'] = Model(
            id='fama_french_5',
            name='Fama-French 5-Factor',
            category=ModelCategory.FACTOR,
            complexity=ModelComplexity.SIMPLE,
            description='Fama-French 5-factor model',
            use_case='Factor modeling',
            data_requirements=['Returns', 'factor data'],
            computational_requirements='Low'
        )
        
        # Simulation Models
        self.models['monte_carlo'] = Model(
            id='monte_carlo',
            name='Monte Carlo Simulation',
            category=ModelCategory.SIMULATION,
            complexity=ModelComplexity.MEDIUM,
            description='Monte Carlo simulation',
            use_case='Risk simulation, pricing',
            data_requirements=['Distribution parameters'],
            computational_requirements='Medium'
        )
        
        self.models['agent_based'] = Model(
            id='agent_based',
            name='Agent-Based Simulation',
            category=ModelCategory.SIMULATION,
            complexity=ModelComplexity.VERY_COMPLEX,
            description='Agent-based market simulation',
            use_case='Market simulation',
            data_requirements=['Agent rules', 'market rules'],
            computational_requirements='Very High'
        )
        
        # ── Additional classical models ───────────────────────────────────────
        additional_models = [
            ('knn', 'K-Nearest Neighbors', ModelCategory.ML, ModelComplexity.SIMPLE),
            ('decision_tree', 'Decision Tree', ModelCategory.ML, ModelComplexity.SIMPLE),
            ('adaboost', 'AdaBoost', ModelCategory.ML, ModelComplexity.MEDIUM),
            ('bagging', 'Bagging', ModelCategory.ML, ModelComplexity.MEDIUM),
            ('sgd', 'Stochastic Gradient Descent', ModelCategory.ML, ModelComplexity.SIMPLE),
            ('bayesian_ridge', 'Bayesian Ridge', ModelCategory.ML, ModelComplexity.MEDIUM),
            ('gaussian_process', 'Gaussian Process', ModelCategory.ML, ModelComplexity.COMPLEX),
            ('isolation_forest', 'Isolation Forest', ModelCategory.ML, ModelComplexity.MEDIUM),
            ('local_outlier_factor', 'Local Outlier Factor', ModelCategory.ML, ModelComplexity.MEDIUM),
            ('one_class_svm', 'One-Class SVM', ModelCategory.ML, ModelComplexity.MEDIUM),
            ('cnn', 'Convolutional Neural Network', ModelCategory.DEEP_LEARNING, ModelComplexity.COMPLEX),
            ('rnn', 'Recurrent Neural Network', ModelCategory.DEEP_LEARNING, ModelComplexity.COMPLEX),
            ('attention', 'Attention Mechanism', ModelCategory.DEEP_LEARNING, ModelComplexity.COMPLEX),
            ('bert', 'BERT', ModelCategory.DEEP_LEARNING, ModelComplexity.VERY_COMPLEX),
            ('gan', 'Generative Adversarial Network', ModelCategory.DEEP_LEARNING, ModelComplexity.VERY_COMPLEX),
            ('vae', 'Variational Autoencoder', ModelCategory.DEEP_LEARNING, ModelComplexity.COMPLEX),
            ('state_space', 'State Space Model', ModelCategory.TIME_SERIES, ModelComplexity.COMPLEX),
            ('vector_autoregression', 'Vector Autoregression', ModelCategory.TIME_SERIES, ModelComplexity.MEDIUM),
            ('cointegration', 'Cointegration Model', ModelCategory.TIME_SERIES, ModelComplexity.MEDIUM),
            ('kalman_smoother', 'Kalman Smoother', ModelCategory.TIME_SERIES, ModelComplexity.MEDIUM),
            ('particle_filter', 'Particle Filter', ModelCategory.TIME_SERIES, ModelComplexity.COMPLEX),
            ('quadratic_programming', 'Quadratic Programming', ModelCategory.OPTIMIZATION, ModelComplexity.MEDIUM),
            ('convex_optimization', 'Convex Optimization', ModelCategory.OPTIMIZATION, ModelComplexity.COMPLEX),
            ('stochastic_optimization', 'Stochastic Optimization', ModelCategory.OPTIMIZATION, ModelComplexity.VERY_COMPLEX),
            ('robust_optimization', 'Robust Optimization', ModelCategory.OPTIMIZATION, ModelComplexity.COMPLEX),
            ('multi_objective', 'Multi-Objective Optimization', ModelCategory.OPTIMIZATION, ModelComplexity.COMPLEX),
            ('credit_var', 'Credit VaR', ModelCategory.RISK, ModelComplexity.MEDIUM),
            ('incremental_var', 'Incremental VaR', ModelCategory.RISK, ModelComplexity.MEDIUM),
            ('component_var', 'Component VaR', ModelCategory.RISK, ModelComplexity.MEDIUM),
            ('marginal_var', 'Marginal VaR', ModelCategory.RISK, ModelComplexity.MEDIUM),
            ('bayesian_hmm', 'Bayesian HMM', ModelCategory.REGIME, ModelComplexity.COMPLEX),
            ('switching_model', 'Markov Switching Model', ModelCategory.REGIME, ModelComplexity.COMPLEX),
            ('regression_tree', 'Regression Tree', ModelCategory.ML, ModelComplexity.SIMPLE),
        ]

        for i, (model_id, name, category, complexity) in enumerate(additional_models, start=40):
            self.models[model_id] = Model(
                id=model_id,
                name=name,
                category=category,
                complexity=complexity,
                description=f'Model for {name}',
                use_case='Use case TBD',
                data_requirements=['Data requirements TBD'],
                computational_requirements='Variable'
            )

        # ── HuggingFace Foundation Models (added 2026-06-02) ─────────────────
        # These models are RESEARCH-ONLY unless explicitly validated out-of-sample.
        # See ml/model_hub.py for instantiation, ml/*_forecaster.py for usage.

        self.models['chronos_t5_small'] = Model(
            id='chronos_t5_small',
            name='Chronos-T5-Small (Amazon)',
            category=ModelCategory.TIME_SERIES,
            complexity=ModelComplexity.COMPLEX,
            description=(
                'Amazon Chronos foundation model for time-series forecasting. '
                'Zero-shot probabilistic forecasting via language model architecture. '
                'Returns, volatility, volume, liquidity horizons.'
            ),
            use_case='Research: returns/vol/volume forecasting (5d, 20d). NOT direct signal.',
            data_requirements=['Univariate time series (min 30 obs)'],
            computational_requirements='Medium (CPU: ~2s/forecast, GPU: ~0.2s). ~250MB RAM.'
        )

        self.models['chronos_t5_large'] = Model(
            id='chronos_t5_large',
            name='Chronos-T5-Large (Amazon)',
            category=ModelCategory.TIME_SERIES,
            complexity=ModelComplexity.VERY_COMPLEX,
            description=(
                'Largest Chronos variant. Best accuracy for volatile financial series. '
                'Recommended for vol forecasting and liquidity prediction.'
            ),
            use_case='Research: high-quality vol/liquidity forecasting. NOT direct signal.',
            data_requirements=['Univariate time series (min 50 obs)'],
            computational_requirements='High (GPU recommended, ~700MB VRAM). ~4s CPU/forecast.'
        )

        self.models['timesfm_2_500m'] = Model(
            id='timesfm_2_500m',
            name='TimesFM 2.0-500M (Google)',
            category=ModelCategory.TIME_SERIES,
            complexity=ModelComplexity.VERY_COMPLEX,
            description=(
                'Google TimesFM 2.0 (500M parameters). Multi-horizon point forecasting. '
                'Patch-based architecture with variable context length. '
                'Strong on regime transitions and vol surface forecasting.'
            ),
            use_case='Research: multi-horizon prediction (1d/5d/20d), regime transitions. NOT direct signal.',
            data_requirements=['Univariate time series (min 20 obs)', 'Frequency specification'],
            computational_requirements='Very High (GPU strongly recommended, ~4GB VRAM). Slow on CPU.'
        )

        self.models['timesfm_1_200m'] = Model(
            id='timesfm_1_200m',
            name='TimesFM 1.0-200M (Google)',
            category=ModelCategory.TIME_SERIES,
            complexity=ModelComplexity.COMPLEX,
            description='Lighter TimesFM variant. More CPU-friendly at some accuracy cost.',
            use_case='Research: multi-horizon prediction (CPU-usable). NOT direct signal.',
            data_requirements=['Univariate time series (min 20 obs)'],
            computational_requirements='High (~1.5GB RAM/VRAM). ~10s CPU/forecast.'
        )

        self.models['patchtst'] = Model(
            id='patchtst',
            name='PatchTST (HuggingFace / NeuralForecast)',
            category=ModelCategory.TIME_SERIES,
            complexity=ModelComplexity.COMPLEX,
            description=(
                'Patch Time-Series Transformer. Treats time-series as image patches '
                'with channel-independent self-attention. Outperforms LSTM significantly '
                'on standard benchmarks. Supports both zero-shot and supervised fine-tuning.'
            ),
            use_case='Research: returns/vol forecasting. Walk-forward CV built in. NOT direct signal.',
            data_requirements=['Time series with DatetimeIndex (min 200 obs for supervised)'],
            computational_requirements='Medium (trainable on CPU, GPU recommended for large windows).'
        )

        self.models['finbert'] = Model(
            id='finbert',
            name='FinBERT (ProsusAI)',
            category=ModelCategory.DEEP_LEARNING,
            complexity=ModelComplexity.COMPLEX,
            description=(
                'BERT fine-tuned on financial text. Classifies sentiment as '
                'positive/negative/neutral. Very useful for earnings calls, '
                'news headlines, SEC filings. Strong accuracy on financial text.'
            ),
            use_case='Research: news/earnings/filing sentiment. Input to alpha research ONLY.',
            data_requirements=['Financial text (headlines, transcripts, filings)'],
            computational_requirements='Medium (~438MB). CPU inference viable. GPU for large batches.'
        )

        self.models['fingpt_sentiment'] = Model(
            id='fingpt_sentiment',
            name='FinGPT-Sentiment (LLaMA2-7B LoRA)',
            category=ModelCategory.DEEP_LEARNING,
            complexity=ModelComplexity.VERY_COMPLEX,
            description=(
                'FinGPT sentiment + summarization model. LLaMA2-7B with LoRA adapter '
                'fine-tuned on financial corpora. Handles longer documents. '
                'Summarizes risk factors, earnings narratives. Falls back to FinBERT on CPU.'
            ),
            use_case='Research: SEC filing summarization, earnings QA, risk extraction. NOT direct signal.',
            data_requirements=['Financial text (transcripts, filings, reports)'],
            computational_requirements='Very High (14GB VRAM FP16; 4-6GB with 4-bit quantization). CPU: very slow.'
        )

        self.models['xgboost_alpha'] = Model(
            id='xgboost_alpha',
            name='XGBoost Alpha Engine',
            category=ModelCategory.ML,
            complexity=ModelComplexity.MEDIUM,
            description=(
                'XGBoost configured for cross-sectional alpha generation with '
                'purged walk-forward validation (embargo=5d). '
                'Heavy regularization to prevent overfitting. IC/ICIR tracking.'
            ),
            use_case='Cross-sectional alpha generation, factor scoring. Validated with purged CV.',
            data_requirements=['Tabular features (no look-ahead), return targets'],
            computational_requirements='Medium. CPU-friendly. ~300 estimators.'
        )

        self.models['catboost_alpha'] = Model(
            id='catboost_alpha',
            name='CatBoost Alpha Engine',
            category=ModelCategory.ML,
            complexity=ModelComplexity.MEDIUM,
            description=(
                'CatBoost configured for noisy financial features. '
                'Native NaN handling, categorical feature support. '
                'Best for signals with sector/industry categoricals.'
            ),
            use_case='Cross-sectional alpha with noisy/categorical features. Purged CV included.',
            data_requirements=['Tabular features, categorical columns optional'],
            computational_requirements='Medium. CPU-friendly.'
        )

        self.models['lightgbm_alpha'] = Model(
            id='lightgbm_alpha',
            name='LightGBM Alpha Engine (Fixed)',
            category=ModelCategory.ML,
            complexity=ModelComplexity.MEDIUM,
            description=(
                'LightGBM properly configured with purged walk-forward CV. '
                'Fixes previous audit finding: "LightGBM trained with look-ahead features." '
                'Now uses PurgedTimeSeriesCV with 5-day embargo. Fast on large datasets.'
            ),
            use_case='Cross-sectional alpha, factor models, large-universe scoring. Purged CV.',
            data_requirements=['Tabular features (no future data), return targets'],
            computational_requirements='Low-Medium. Very fast. Best for >10K rows.'
        )

        self.models['tabular_ensemble'] = Model(
            id='tabular_ensemble',
            name='Tabular Ensemble (XGB + CatBoost + LGB)',
            category=ModelCategory.ML,
            complexity=ModelComplexity.COMPLEX,
            description=(
                'Weighted ensemble of XGBoost, CatBoost, LightGBM. '
                'Weights determined by inverse validation RMSE from purged walk-forward CV. '
                'Most robust tabular approach for Indian equity cross-sectional alpha.'
            ),
            use_case='Primary tabular alpha engine. Use after purged CV validation.',
            data_requirements=['Tabular features with DatetimeIndex'],
            computational_requirements='Medium. Requires all three boosting packages.'
        )
    
    def get_model(self, model_id: str) -> Optional[Model]:
        """Get a model by ID."""
        return self.models.get(model_id)
    
    def get_models_by_category(self, category: ModelCategory) -> List[Model]:
        """Get models by category."""
        return [m for m in self.models.values() if m.category == category]
    
    def get_models_by_complexity(self, complexity: ModelComplexity) -> List[Model]:
        """Get models by complexity."""
        return [m for m in self.models.values() if m.complexity == complexity]
    
    def print_catalog_report(self) -> None:
        """Print catalog report."""
        print("\n" + "="*80)
        print("TOP 50 MODELS CATALOG REPORT")
        print("="*80)
        
        print(f"\nTotal Models: {len(self.models)}")
        
        print(f"\nBy Category:")
        for category in ModelCategory:
            count = len(self.get_models_by_category(category))
            if count > 0:
                print(f"  {category.value}: {count}")
        
        print(f"\nBy Complexity:")
        for complexity in ModelComplexity:
            count = len(self.get_models_by_complexity(complexity))
            if count > 0:
                print(f"  {complexity.value}: {count}")
        
        print(f"\nSample Models by Category:")
        print(f"{'ID':<25} {'Name':<40} {'Category':<15} {'Complexity':<15}")
        print("-" * 100)
        for model in list(self.models.values())[:15]:
            print(f"{model.id:<25} {model.name:<40} {model.category.value:<15} {model.complexity.value:<15}")
        
        print("\n" + "="*80)


def sample_models_catalog():
    """Demonstrate models catalog."""
    print("=== Top 50 Models Catalog Demo ===\n")
    
    catalog = ModelsCatalog()
    catalog.print_catalog_report()
    
    print("\n=== Top 50 Models Catalog Demo Complete ===")
    print("Key capabilities:")
    print("- Catalog of top 50 models")
    print("- Classification by category (ML, deep learning, time series, etc.)")
    print("- Classification by complexity (simple, medium, complex, very complex)")
    print("- Use cases and data requirements")
    print("- Computational requirements for each model")


if __name__ == "__main__":
    sample_models_catalog()
