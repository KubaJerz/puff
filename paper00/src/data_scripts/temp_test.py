# Test database connectivity first
from utils import db_connection
with db_connection() as conn:
    print("Database connection successful!")

# Test a simple dataset creation
from utils import create_base_config, get_all_sessions_from_db
config = create_base_config()
sessions = get_all_sessions_from_db(['puff', 'puffs', 'puff segmentation model v2', 'puff segmentation model v1'])
print(f"Found {len(sessions)} sessions")