from config import client # Import the pre-initialized OpenAI client
import pandas as pd # Needed for DataFrame type hinting and operations
import logging

def generate_column_descriptions(data: pd.DataFrame) -> dict:
    """Generates descriptions for each column in a DataFrame using simple fallback logic.

    Args:
        data (pandas.DataFrame): The DataFrame to analyze.

    Returns:
        dict: A dictionary mapping column names to their descriptions.
    """
    logging.info("Generating column descriptions using fallback logic (skipping GPT to avoid rate limits)...")
    descriptions = {}
    # Check if data is empty before proceeding
    if data.empty:
        logging.warning("DataFrame is empty, cannot generate column descriptions.")
        return descriptions

    for column in data.columns:
        # Get sample values, handling potential errors
        try:
            sample_values = data[column].dropna().astype(str).head(3).tolist()
            sample_text = ", ".join(sample_values)
            if not sample_text:
                 sample_text = "[No non-null values]" # Handle columns with all nulls
        except Exception as e:
            logging.warning(f"Could not get sample values for column '{column}': {e}")
            sample_text = "[Could not extract samples]"

        # Generate simple fallback description based on column name and sample values
        description = f"Column '{column}' with sample values: {sample_text}"
        descriptions[column] = description
        logging.debug(f"Generated fallback description for '{column}': {description}")

    logging.info(f"Finished generating descriptions for {len(descriptions)} columns.")
    return descriptions


def generate_summary(df: pd.DataFrame) -> dict:
    """Generates a summary of the DataFrame, including row count, column count, and column descriptions.

    Args:
        df (pandas.DataFrame): The DataFrame to summarize.

    Returns:
        dict: A dictionary containing the summary information.
    """
    logging.info("Generating data summary...")
    if df.empty:
        logging.warning("DataFrame is empty, generating empty summary.")
        return {
            "rows": 0,
            "columns": 0,
            "column_info": {}
        }

    meta = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_info": {}
    }

    # This calls the GPT description function
    descriptions = generate_column_descriptions(df)

    for col in df.columns:
        meta["column_info"][col] = {
            "dtype": str(df[col].dtype),
            # Use the description from the GPT call, or a fallback if generation failed for this column
            "description": descriptions.get(col, "Description unavailable")
        }
    logging.info("Finished generating data summary.")
    return meta