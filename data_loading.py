import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw real estate dataset
    """
    return pd.read_excel(filepath)

if __name__ == "__main__":
    df = load_data("house_price.xlsx")
    
