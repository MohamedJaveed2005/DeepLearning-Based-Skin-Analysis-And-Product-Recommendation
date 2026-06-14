from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

# Create the database instance
db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True) # primary keys are required by SQLAlchemy
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(1000))

    # Method to securely hash and set the password
    def set_password(self, password):
        self.password = generate_password_hash(password)

    # Method to check a provided password against the stored hash
    def check_password(self, password):
        return check_password_hash(self.password, password)