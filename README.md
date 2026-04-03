# Artificial Societies: generating synthetic answers varying model and personas


## Overview Workflow

1. Create a year-specific GSS extract with `scripts/00a_create_gss_extract_multiyear.R`.
2. Convert respondent demographics into natural-language personas with `scripts/00b_generate_personas.R`.
2. Create persona variations  `scripts/00c_create_variations.py`.
3. Query one or more LLMs with `scripts/01_generate_synthetic_GSS.py`.
4. Use the resulting CSV files for evaluations with JSD and homogenisation `scripts/02-evaluations_results.py`.


## Repository Structure

```
ArtificialSocieties/
├── generation/              # Data generation pipeline
│   ├── scripts/            # Python and R scripts for creating synthetic data
│   │   ├── 00a_create_gss_extract_multiyear.R  # Extract GSS data by year
│   │   ├── 00b_generate_personas.R             # Create natural language personas
│   │   ├── 00b_create_variations            # Create personas variations
│   │   ├── 01_generate_synthetic_GSS.py        # Query LLMs (OpenRouter)
│   └── evaluation/     # plots
│   |   └── homogenisation.png # visulising homogenisation 
│   |   └── jsd_heatmap.png # visulising jsd
│   |   └── jsd_homgenisation.png # summary graph(s)
│   ├── data/               # Small data files (personas, extracts)
│   │   ├── gss2024_personas.csv                # personas with all information
│   │   ├── gss7224_r1.dta               # input data
│   │   ├── personas_demographics_poliotical.csv  # personas with demographics and politial affiliation
│   │   └── personas_demographics       # Personas only demographics
│   └── synthetic_data/     # LLM-generated survey responses
│       └── year_2024/      # 9 CSV files, model x personas
├── alternative_survey            # Replication of work with a different survey and other persona variations (to test code robustness)
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Installation

### Prerequisites

- **Python 3.8+** for data generation

### Python Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


## Usage

cd ../..
```

#### Step 2: Query LLMs

```bash
# Navigate to the generation scripts directory (if not already there)
cd generation/scripts

# Set your OpenRouter API key
export OPENROUTER_API_KEY="your-key-here"

# Query all models with 100 personas (cheap! ~ 2 cents depending on models)
python 01_generate_synthetic_GSS.py --year 2024 --all-models --personas 100

# Or query specific models
python 01_generate_synthetic_GSS.py --year 2024 --models "gemini-2.5-flash-lite" --personas 100

# Return to project root
cd ../..
```


## Models Included

The repository includes results for 3 LLMs across families, but is completely flexible only constraint by availability on OpenRouter:

- **Google**: Gemini 2.5 Flash/Lite
- **DeepSeek**: v3,
- **QWEN**: Qwen

Plus **GSS-2024** human baseline for comparison.

## Acknowledgments

- General Social Survey (GSS) data from NORC at the University of Chicago
- LLM API access via OpenRouter

### API rate limits

The OpenRouter API has rate limits. The script includes retry logic with exponential backoff. For large runs, consider:
- Running overnight
- Using the `--max-workers` parameter to reduce concurrency
- Splitting across multiple days

## Development

To modify or extend this project:

1. **Add new models**: Edit `POPULAR_MODELS` in `generation/scripts/01_generate_synthetic_GSS.py`
2. **Add new questions**: Edit `GSS_QUESTIONS_*` dictionaries in the same file, questions are not bound to GSS, but must be in the same format (analysis is robust to other questions)

## Repository Metadata

- **Data Year**: 2024 (GSS wave)
- **Models**: 3 LLMs + human baseline
- **Survey Items**: 5 questions 
- **Personas**: 100 synthetic respondenses per model per persona-set
