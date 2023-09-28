from dotenv import load_dotenv

import os
load_dotenv()

def get_postgres_str():
    POSTGRES_ADDRESS = os.getenv('POSTGRES_ADDRESS') ## INSERT YOUR DB ADDRESS IF IT'S NOT ON PANOPLY
    POSTGRES_PORT = os.getenv('POSTGRES_PORT')
    POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME') ## CHANGE THIS TO YOUR PANOPLY/POSTGRES USERNAME
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD') ## CHANGE THIS TO YOUR PANOPLY/POSTGRES PASSWORD
    POSTGRES_DBNAME = os.getenv('POSTGRES_DBNAME') ## CHANGE THIS TO YOUR DATABASE NAME

    ## A long string that contains the necessary Postgres login information
    postgres_str = ('postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'
    .format(username=POSTGRES_USERNAME,
    password=POSTGRES_PASSWORD,
    ipaddress=POSTGRES_ADDRESS,
    port=POSTGRES_PORT,
    dbname=POSTGRES_DBNAME))

    return postgres_str