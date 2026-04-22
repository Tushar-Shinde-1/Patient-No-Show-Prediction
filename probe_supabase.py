import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

def list_all_tables():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    supabase = create_client(url, key)

    print("🔍 Attempting to list all tables in 'public' schema...")
    try:
        # We can try to query pg_catalog or information_schema via RPC or direct select if permitted
        # Most 'anon' keys can't do this, but let's try.
        response = supabase.table("noshow_appointments").select("*").limit(1).execute()
        print("Success with noshow_appointments")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_all_tables()
