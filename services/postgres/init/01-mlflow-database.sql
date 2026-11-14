-- Runs once, when the postgres-data volume is first created. POSTGRES_DB creates the
-- app's database (spatify: feedback and pipeline runs); MLflow's runs and model
-- registry get a database of their own.
CREATE DATABASE mlflow;
