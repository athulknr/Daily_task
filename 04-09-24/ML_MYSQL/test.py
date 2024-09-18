#importing the necessary libraries
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import mysql.connector

def create_sqlalchemy_engine():
    engine = create_engine('mysql+pymysql://root:@localhost/test_db')
    return engine

def create_connection():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="test_db"
    )
    return connection

def setup_database():
    connection = create_connection()
    cursor = connection.cursor()

    # Create the query for if the database is not exists
    cursor.execute("CREATE DATABASE IF NOT EXISTS test_db")
    # Use database
    cursor.execute("USE test_db")
    # Create the table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset(
            id INT AUTO_INCREMENT PRIMARY KEY,
            feature1 FLOAT,
            feature2 FLOAT,
            label INT
        )
    """)
    connection.commit()
    cursor.close()
    connection.close()

def fetch_data():
    engine = create_sqlalchemy_engine()
    query = "SELECT * FROM dataset"
    df = pd.read_sql(query, engine)
    return df

def train_model(df):
    x = df[['feature1', 'feature2']]
    y = df['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    return predictions

def store_predictions(predictions):
    connection = create_connection()
    cursor = connection.cursor()

    # Query for insert
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions(
            id INT PRIMARY KEY,
            prediction INT
        )
    """)

    for i, pred in enumerate(predictions):
        cursor.execute("INSERT INTO predictions(id, prediction) VALUES(%s, %s)", (i+1, int(pred)))

    connection.commit()
    cursor.close()
    connection.close()

def insert_sample_data():
    data = [
        (1.5, 2.5, 1),
        (1.6, 2.6, 2),
        (1.7, 2.7, 3),
        (1.8, 2.8, 4),
        (1.9, 2.9, 5),
    ]
    connection = create_connection()
    cursor = connection.cursor()

    cursor.executemany("INSERT INTO dataset(feature1, feature2, label) VALUES(%s, %s, %s)", data)
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

if __name__ == "__main__":
    main()
