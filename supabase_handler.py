import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class SupabaseHandler:
    def __init__(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL or SUPABASE_KEY not found in environment variables.")
        self.supabase: Client = create_client(url, key)

    def fetch_data(self, table_name="Patient", limit=100000):
        """
        Fetches data from Supabase table and returns it as a Pandas DataFrame.
        """
        try:
            print(f"📡 Fetching data from Supabase table: '{table_name}'...")
            # Supabase pagination might be needed for very large datasets, 
            # but for 110k we can try a large range or multi-fetch.
            response = self.supabase.table(table_name).select("*").limit(limit).execute()
            
            if not response.data:
                print(f"⚠ No data found in table '{table_name}'.")
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            print(f"✅ Successfully fetched {len(df)} rows.")
            return df
        except Exception as e:
            print(f"❌ Error fetching data: {str(e)}")
            return pd.DataFrame()

    def list_tables(self):
        """
        Helper to check available tables (via a dummy query if needed)
        """
        # Supabase Python client doesn't have a direct 'list tables' like SQL
        # We'd usually check the public schema.
        pass

if __name__ == "__main__":
    handler = SupabaseHandler()
    # Test fetch
    df = handler.fetch_data()
    if not df.empty:
        print(df.head())
