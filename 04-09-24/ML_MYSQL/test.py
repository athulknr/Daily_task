#importing the necessary libraries
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import mysql.connector

#create an engine of sqlalchemy
def create_sqlalchemy_engine():
    engine = create_engine('mysql+pymysql://root:root123:@localhost/ai_project_db')
    return engine

#lets create or set up the connection using mysql connector
def create_connection(database=None):
    connection = mysql.connector.connect(
         host="localhost",  
         user="root",
         password="root123",
         database=database  # Pass the database name if provided, else connect without it
    )
    return connection

#setup the database as well as query over here
def setup_database():
    # Connect without specifying the database
    connection = create_connection()
    cursor = connection.cursor()

    # Create the database if it does not exist
    cursor.execute("CREATE DATABASE IF NOT EXISTS ai_project_db")
    connection.commit()

    # Now connect to the newly created database
    connection = create_connection(database="ai_project_db")
    cursor = connection.cursor()

    # Create the table if it does not exist
    cursor.execute("""CREATE TABLE IF NOT EXISTS 
                   dataset(
                   id INT AUTO_INCREMENT PRIMARY KEY,
                   feature1 FLOAT,
                   feature2 FLOAT,
                   label INT)
                   """)
    connection.commit()
    cursor.close()
    connection.close()

#fetch data
def fetch_data():
    engine = create_sqlalchemy_engine()
    query = "SELECT * FROM dataset"
    df = pd.read_sql(query, engine)
    return df

#train the model
def train_model(df):
    X = df[['feature1', 'feature2']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    return predictions

#store the predictions in mysql db
def store_predictions(predictions):
    connection = create_connection(database="ai_project_db")
    cursor = connection.cursor()

    # Query for creating the table if it does not exist
    cursor.execute("CREATE TABLE IF NOT EXISTS predictions(id INT, prediction INT)")
    
    # Inserting predictions into the table
    for i, pred in enumerate(predictions):
        cursor.execute("INSERT INTO predictions(id, prediction) VALUES (%s, %s)", (i+1, int(pred)))

    connection.commit()
    cursor.close()
    connection.close()

#insert sample data into the dataset table
def insert_sample_data():
    data = [
        (1.5, 2.5, 0),
        (2.5, 4.5, 0),
        (2.0, 3.1, 1),
        (1.8, 2.9, 0),
        (1.1, 2.1, 1)
    ]
    connection = create_connection(database="ai_project_db")
    cursor = connection.cursor()

    cursor.executemany("INSERT INTO dataset(feature1, feature2, label) VALUES (%s, %s, %s)", data)
    connection.commit()
    cursor.close()
    connection.close()

def main():
    setup_database()
    insert_sample_data()

    df = fetch_data()
    predictions = train_model(df)
    print("Predictions:", predictions)

    store_predictions(predictions)

if __name__ == "_main_":
    main()