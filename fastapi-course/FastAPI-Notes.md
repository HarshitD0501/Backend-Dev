# 📘 FastAPI Complete Notes — Module 3 & 4 & 5

> Covers: Building APIs, Pydantic Models, Validation, Async, Database Integration, CRUD Operations, Machine Learning
> These notes are written for deep understanding — not for cramming.

---

## Table of Contents

1. [Module 3: Building APIs](#module-3-building-apis)
   - [The Basics — Creating a FastAPI App](#1-the-basics--creating-a-fastapi-app)
   - [HTTP Methods & Endpoints](#2-http-methods--endpoints)
   - [Path Parameters vs Query Parameters](#3-path-parameters-vs-query-parameters)
   - [Pydantic Models — Data Validation](#4-pydantic-models--data-validation)
   - [Field Validation — Adding Rules](#5-field-validation--adding-rules)
   - [Response Model](#6-response-model)
   - [HTTPException — Error Handling](#7-httpexception--error-handling)
   - [In-Memory CRUD (No Database)](#8-in-memory-crud-no-database)
   - [Sync vs Async](#9-sync-vs-async)
2. [Module 4: Database Integration](#module-4-database-integration)
   - [Why a Database?](#1-why-a-database)
   - [SQLAlchemy — The ORM](#2-sqlalchemy--the-orm)
   - [Project Structure for DB CRUD](#3-project-structure-for-db-crud)
   - [database.py — The Connection](#4-databasepy--the-connection)
   - [models.py — The Table Blueprint](#5-modelspy--the-table-blueprint)
   - [schemas.py — The Data Guards](#6-schemaspy--the-data-guards)
   - [crud.py — The Workers](#7-crudpy--the-workers)
   - [main.py — The Receptionist](#8-mainpy--the-receptionist)
   - [Dependency Injection — get_db()](#9-dependency-injection--get_db)
3. [Module 5: Machine Learning Integration](#module-5-machine-learning-integration)
   - [The ML Workflow](#1-the-ml-workflow)
   - [Serializing Models (Pickle/Joblib)](#2-serializing-models-picklejoblib)
   - [Building the API](#3-building-the-api)
   - [Why Load Globally?](#4-why-load-globally)
4. [Step-by-Step: Build ANY CRUD API from Scratch](#step-by-step-build-any-crud-api-from-scratch)

---

# Module 3: Building APIs

## 1. The Basics — Creating a FastAPI App

```python
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def home():
    return {'message': 'Hello FastAPI!'}
```

**What's happening here?**

| Line | What it does |
|------|-------------|
| `app = FastAPI()` | Creates the application instance — this IS your API |
| `@app.get('/')` | A **decorator** that tells FastAPI: when someone visits `/`, run this function |
| `def home()` | A regular Python function that returns data |
| `return {'message': ...}` | FastAPI automatically converts this dict to JSON |

**Run it:**
```bash
uvicorn main:app --reload
#         ↑    ↑     ↑
#    filename  variable  auto-restart on file changes
```

---

## 2. HTTP Methods & Endpoints

HTTP methods tell the server **what action** you want to perform:

| Method | Decorator | Purpose | Example |
|--------|-----------|---------|---------|
| **GET** | `@app.get()` | Read/Fetch data | Get all users |
| **POST** | `@app.post()` | Create new data | Add a user |
| **PUT** | `@app.put()` | Update existing data (full) | Update all fields of a user |
| **PATCH** | `@app.patch()` | Update existing data (partial) | Update just the name |
| **DELETE** | `@app.delete()` | Remove data | Delete a user |

**Real-world analogy:**
- `GET` = "Show me the menu" (reading)
- `POST` = "I'd like to place an order" (creating)
- `PUT` = "Change my entire order" (full update)
- `DELETE` = "Cancel my order" (removing)

---

## 3. Path Parameters vs Query Parameters

### Path Parameters — Part of the URL

```python
@app.get('/users/{user_id}')
def get_user(user_id: int):
    return {'user_id': user_id}
```

- URL: `http://localhost:8000/users/42`
- `user_id` = 42
- Used for **identifying a specific resource**

### Query Parameters — After the `?` in URL

```python
@app.get('/search')
def search(q: str, limit: int = 10):
    return {'query': q, 'limit': limit}
```

- URL: `http://localhost:8000/search?q=python&limit=5`
- Used for **filtering, searching, or optional data**

### The Rule:
> **Path params** = WHICH resource (e.g., user #42)
> **Query params** = HOW to get it (e.g., sorted, filtered, limited)

---

## 4. Pydantic Models — Data Validation

Pydantic models define what data looks like and **automatically validate** it.

### Without Pydantic (Bad):
```python
@app.post('/users')
def create_user(name, email, age):
    # No validation! Someone could send age="banana" 🍌
    pass
```

### With Pydantic (Good):
```python
from pydantic import BaseModel

class User(BaseModel):
    name: str        # Must be a string
    email: str       # Must be a string
    age: int         # Must be an integer
```

**What happens when someone sends bad data?**
```json
// Sending: {"name": "Harshit", "email": "h@mail.com", "age": "banana"}
// FastAPI auto-responds with:
{
  "detail": [{
    "msg": "value is not a valid integer",
    "type": "type_error.integer",
    "loc": ["body", "age"]
  }]
}
```

FastAPI does this **automatically** — you write zero validation code!

### How Pydantic models work in an endpoint:
```python
@app.post('/users')
def create_user(user: User):    # ← FastAPI reads the JSON body and validates it
    return user                  # ← Returns validated data as JSON
```

When someone sends a POST request with JSON body, FastAPI:
1. Reads the raw JSON from the request body
2. Tries to create a `User` object from it
3. If validation fails → returns 422 error automatically
4. If validation passes → your function receives a clean `User` object

---

## 5. Field Validation — Adding Rules

Use `Field()` to add constraints beyond just the type:

```python
from pydantic import BaseModel, Field, StrictInt
from typing import Optional

class Employee(BaseModel):
    id: int = Field(..., gt=0, title='Employee ID')
    name: str = Field(..., min_length=3, max_length=30)
    department: str = Field(..., min_length=3, max_length=30)
    age: Optional[StrictInt] = Field(default=None, ge=21)
```

### Field() Parameters Explained:

| Parameter | Meaning | Example |
|-----------|---------|---------|
| `...` | **Required** field (no default) | `Field(...)` |
| `default=None` | **Optional** field with default value | `Field(default=None)` |
| `gt=0` | Greater than 0 | ID must be positive |
| `ge=21` | Greater than or equal to 21 | Age must be 21+ |
| `lt=100` | Less than 100 | |
| `le=100` | Less than or equal to 100 | |
| `min_length=3` | Minimum string length | Name at least 3 chars |
| `max_length=30` | Maximum string length | Name at most 30 chars |
| `title='...'` | Documentation label | Shows in Swagger docs |

### Special Types:

| Type | What it does |
|------|-------------|
| `Optional[int]` | Field can be `int` OR `None` |
| `StrictInt` | Only accepts pure integers, rejects `3.0` |
| `EmailStr` | Validates email format (needs `pip install pydantic[email]`) |

---

## 6. Response Model

Controls what data gets **sent back** to the user:

```python
@app.get('/user', response_model=User)
def get_user():
    return User(id=1, name='Bruce')
```

**Why use response_model?**
- Filters out internal/private fields
- Ensures consistent output format
- Shows up in Swagger docs
- Validates the response before sending

---

## 7. HTTPException — Error Handling

When something goes wrong, raise an HTTPException:

```python
from fastapi import HTTPException

@app.get('/users/{user_id}')
def get_user(user_id: int):
    user = find_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail='User Not Found')
    return user
```

### Common HTTP Status Codes:

| Code | Meaning | When to use |
|------|---------|-------------|
| 200 | OK | Successful GET/PUT |
| 201 | Created | Successful POST |
| 400 | Bad Request | Invalid input data |
| 404 | Not Found | Resource doesn't exist |
| 409 | Conflict | Duplicate entry |
| 422 | Unprocessable Entity | Validation error (auto by FastAPI) |
| 500 | Internal Server Error | Something crashed |

---

## 8. In-Memory CRUD (No Database)

In Module 3, the data is stored in a **Python list** (resets when server restarts):

```python
from fastapi import FastAPI, HTTPException
from models import Employee
from typing import List

employees_db: List[Employee] = []   # ← This is your "database" (just a list)

app = FastAPI()

# CREATE
@app.post('/employees')
def add_employee(new_emp: Employee):
    for emp in employees_db:
        if emp.id == new_emp.id:
            raise HTTPException(status_code=400, detail='Already exists')
    employees_db.append(new_emp)
    return new_emp

# READ ALL
@app.get('/employees')
def get_employees():
    return employees_db

# READ ONE
@app.get('/employees/{emp_id}')
def get_employee(emp_id: int):
    for emp in employees_db:
        if emp.id == emp_id:
            return emp
    raise HTTPException(status_code=404, detail='Not Found')

# UPDATE
@app.put('/employees/{emp_id}')
def update_employee(emp_id: int, updated: Employee):
    for i, emp in enumerate(employees_db):
        if emp.id == emp_id:
            employees_db[i] = updated
            return updated
    raise HTTPException(status_code=404, detail='Not Found')

# DELETE
@app.delete('/employees/{emp_id}')
def delete_employee(emp_id: int):
    for i, emp in enumerate(employees_db):
        if emp.id == emp_id:
            del employees_db[i]
            return {'message': 'Deleted'}
    raise HTTPException(status_code=404, detail='Not Found')
```

**The pattern in every operation:**
1. Loop through the list to find the item
2. If found → do the operation
3. If not found → raise 404

---

## 9. Sync vs Async

### Synchronous (Blocking) — One at a time
```python
import time

def task(name, seconds):
    time.sleep(seconds)    # BLOCKS everything

task('A', 2)  # Wait 2s...
task('B', 1)  # Then wait 1s...
task('C', 3)  # Then wait 3s...
# Total: 6 seconds 😴
```

### Asynchronous (Non-Blocking) — Concurrent
```python
import asyncio

async def task(name, seconds):
    await asyncio.sleep(seconds)    # Non-blocking, lets others run

await asyncio.gather(task('A', 2), task('B', 1), task('C', 3))
# Total: 3 seconds ⚡ (runs concurrently)
```

### In FastAPI:
```python
# Sync endpoint — fine for CPU-bound tasks or simple DB queries
@app.get("/users")
def get_users():
    return db.query(User).all()

# Async endpoint — better for I/O tasks (API calls, file reads)
@app.get("/wait")
async def wait():
    await asyncio.sleep(3)       # Doesn't block other requests
    return {"message": "Finished waiting!"}
```

**Rule of thumb:**
- Use `def` (sync) for database queries with SQLAlchemy (it's not async)
- Use `async def` for external API calls, file I/O, or long waits

---

---

# Module 4: Database Integration

## 1. Why a Database?

Module 3 stored data in a **Python list** → data is lost when server restarts.

Module 4 uses **SQLite + SQLAlchemy** → data persists in a file (`test.db`).

| Feature | Python List (Module 3) | Database (Module 4) |
|---------|----------------------|-------------------|
| Data persists? | ❌ Lost on restart | ✅ Saved to file |
| Can handle multiple users? | ❌ Race conditions | ✅ Transactions |
| Can search efficiently? | ❌ Loop through all | ✅ SQL indexes |
| Relationships? | ❌ Manual | ✅ Foreign keys |

---

## 2. SQLAlchemy — The ORM

**ORM = Object Relational Mapper**

Instead of writing raw SQL:
```sql
INSERT INTO employees (name, email) VALUES ('Harshit', 'h@gmail.com');
```

You write Python:
```python
employee = Employee(name='Harshit', email='h@gmail.com')
db.add(employee)
db.commit()
```

SQLAlchemy **translates your Python code into SQL** automatically.

### Key SQLAlchemy Operations:

| Python Code | SQL Equivalent |
|-------------|---------------|
| `db.query(Employee).all()` | `SELECT * FROM employees` |
| `db.query(Employee).filter(Employee.id == 1).first()` | `SELECT * FROM employees WHERE id = 1 LIMIT 1` |
| `db.add(emp)` | `INSERT INTO employees ...` (staged, not yet saved) |
| `db.commit()` | Actually saves all staged changes |
| `db.refresh(emp)` | Reloads the object from DB (gets auto-generated fields like `id`) |
| `db.delete(emp)` | `DELETE FROM employees WHERE id = ...` |

---

## 3. Project Structure for DB CRUD

```
crud-app/
├── database.py   → Connects to the database
├── models.py     → Defines the database TABLE (what columns exist)
├── schemas.py    → Defines the API DATA shapes (what JSON looks like)
├── crud.py       → Functions that do the actual DB operations
└── main.py       → FastAPI endpoints that tie everything together
```

### How they connect (Data Flow):

```
User sends JSON → schemas.py validates it
                      ↓
                  main.py receives clean data
                      ↓
                  crud.py uses models.py to talk to database.py
                      ↓
                  Database stores/returns data
                      ↓
                  schemas.py formats the response
                      ↓
                  User receives JSON response
```

---

## 4. database.py — The Connection

```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={'check_same_thread': False}
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()
```

### Line-by-line:

| Line | What it does | Analogy |
|------|-------------|---------|
| `SQLALCHEMY_DATABASE_URL` | Where the DB file lives | The **address** of the building |
| `engine` | The connection to the database | The **road** to the building |
| `check_same_thread=False` | SQLite-specific: allows multi-thread access | Needed because FastAPI handles multiple requests |
| `SessionLocal` | A factory that creates DB sessions | A **ticket machine** — each call gets a new ticket |
| `autoflush=False` | Don't auto-sync Python objects with DB | You control when changes are sent |
| `autocommit=False` | Don't auto-save — you must call `commit()` | You control when changes are permanent |
| `Base` | Parent class for all models | The **template** all tables inherit from |

### For different databases, just change the URL:
```python
# SQLite (file-based, simple)
"sqlite:///./test.db"

# PostgreSQL (production-grade)
"postgresql://user:password@localhost/dbname"

# MySQL
"mysql://user:password@localhost/dbname"
```

---

## 5. models.py — The Table Blueprint

```python
from sqlalchemy import Column, Integer, String
from database import Base

class Employee(Base):
    __tablename__ = 'employees'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
```

### This is NOT a Pydantic model! Key differences:

| Feature | SQLAlchemy Model (`models.py`) | Pydantic Schema (`schemas.py`) |
|---------|-------------------------------|-------------------------------|
| Purpose | Defines the **database table** | Defines the **API data shape** |
| Inherits from | `Base` (SQLAlchemy) | `BaseModel` (Pydantic) |
| Works with | Database (read/write rows) | JSON (validate input/output) |
| Example | `Column(Integer, primary_key=True)` | `id: int` |

### Column options:

| Option | Meaning |
|--------|---------|
| `primary_key=True` | This is the unique identifier for each row |
| `index=True` | Creates a DB index for faster searches |
| `unique=True` | No duplicate values allowed |
| `nullable=False` | Cannot be empty/null |
| `default=...` | Default value if not provided |

### How it becomes a real table:

```python
# In main.py — this line creates the actual table in the database
Base.metadata.create_all(bind=engine)
```

This reads ALL model classes that inherit from `Base` and creates their tables if they don't exist.

---

## 6. schemas.py — The Data Guards

```python
from pydantic import BaseModel, EmailStr

class EmployeeBase(BaseModel):     # Shared fields
    name: str
    email: EmailStr

class EmployeeCreate(EmployeeBase):  # What user sends to CREATE
    pass

class EmployeeUpdate(EmployeeBase):  # What user sends to UPDATE
    pass

class EmployeeOut(EmployeeBase):     # What API RETURNS
    id: int
    class Config:
        orm_mode = True
```

### Why separate schemas? This is the most confusing part, so let's break it down:

**The problem:** When you CREATE an item, you DON'T send an ID (the database auto-generates it).
But when the API RESPONDS, it SHOULD include the ID.

So you need different shapes for input vs output:

```
INPUT (Create/Update):              OUTPUT (Response):
┌─────────────────┐                 ┌─────────────────┐
│ name: "Harshit" │      →→→→      │ id: 1           │  ← Added by DB!
│ email: "h@m.c"  │   Database     │ name: "Harshit" │
└─────────────────┘                 │ email: "h@m.c"  │
                                    └─────────────────┘
```

### Inheritance chain:
```
EmployeeBase (name, email)
    ├── EmployeeCreate (inherits name, email — nothing new)
    ├── EmployeeUpdate (inherits name, email — nothing new)
    └── EmployeeOut (inherits name, email + ADDS id)
```

**Why have EmployeeCreate and EmployeeUpdate if they're the same as Base?**
Right now they're identical, but in a real app they might differ:
```python
class EmployeeCreate(EmployeeBase):
    password: str          # Need password when creating

class EmployeeUpdate(EmployeeBase):
    name: Optional[str]    # All fields optional when updating
    email: Optional[str]
```

### What is `orm_mode = True`?

By default, Pydantic reads data from **dicts**: `{"name": "Harshit"}`.

But SQLAlchemy returns **objects**: `employee.name` (attribute access, not dict).

`orm_mode = True` tells Pydantic: "Also read data from object attributes, not just dicts."

Without it, you'd get an error when trying to return a SQLAlchemy object from your endpoint.

---

## 7. crud.py — The Workers

Each function does ONE database operation:

### CREATE — Adding a new row
```python
def create_employee(db: Session, employee: schemas.EmployeeCreate):
    # 1. Convert Pydantic schema → SQLAlchemy model
    db_employee = models.Employee(
        name=employee.name,
        email=employee.email
    )
    # 2. Stage it (like "git add")
    db.add(db_employee)

    # 3. Save to database (like "git commit")
    db.commit()

    # 4. Reload from DB — now it has the auto-generated ID!
    db.refresh(db_employee)

    return db_employee
```

**Why `db.refresh()`?** When you do `db.add()` + `db.commit()`, the database auto-generates
the `id`. But your Python object doesn't know about it yet. `refresh()` reloads it from the DB
so `db_employee.id` is now populated.

### READ ALL — Get every row
```python
def get_employees(db: Session):
    return db.query(models.Employee).all()
    # SQL: SELECT * FROM employees
```

### READ ONE — Get specific row by ID
```python
def get_employee(db: Session, emp_id: int):
    return (
        db.query(models.Employee)
        .filter(models.Employee.id == emp_id)
        .first()   # Returns None if not found (not an error!)
    )
    # SQL: SELECT * FROM employees WHERE id = ? LIMIT 1
```

### UPDATE — Modify an existing row
```python
def update_employee(db: Session, emp_id: int, employee: schemas.EmployeeUpdate):
    # 1. Find the existing employee
    db_employee = db.query(models.Employee).filter(
        models.Employee.id == emp_id
    ).first()

    # 2. If found, update its fields
    if db_employee:
        db_employee.name = employee.name
        db_employee.email = employee.email
        db.commit()           # Save changes
        db.refresh(db_employee)  # Reload fresh data

    # 3. Returns None if not found, updated object if found
    return db_employee
```

### DELETE — Remove a row
```python
def delete_employee(db: Session, emp_id: int):
    db_employee = db.query(models.Employee).filter(
        models.Employee.id == emp_id
    ).first()

    if db_employee:
        db.delete(db_employee)  # Mark for deletion
        db.commit()             # Execute the deletion

    return db_employee
```

### The common pattern in every CRUD function:
```
1. Query the database  →  db.query(Model).filter(...).first()
2. Do something         →  db.add() / modify fields / db.delete()
3. Save                 →  db.commit()
4. Optionally reload    →  db.refresh()
5. Return the result
```

---

## 8. main.py — The Receptionist

```python
import models, schemas, crud
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from database import engine, SessionLocal, Base
from typing import List

# Creates all tables in the database (runs once at startup)
Base.metadata.create_all(bind=engine)

app = FastAPI()
```

### Each endpoint follows the same pattern:

```python
@app.post('/employees', response_model=schemas.EmployeeOut)
def create_employee(employee: schemas.EmployeeCreate, db: Session = Depends(get_db)):
    return crud.create_employee(db, employee)
```

**Breaking this down:**
1. `@app.post('/employees')` → POST requests to `/employees` trigger this
2. `response_model=schemas.EmployeeOut` → Response will include `id + name + email`
3. `employee: schemas.EmployeeCreate` → FastAPI validates the JSON body
4. `db: Session = Depends(get_db)` → Dependency injection (explained below)
5. `crud.create_employee(db, employee)` → Calls the worker function

### Error handling pattern:
```python
@app.get('/employees/{emp_id}', response_model=schemas.EmployeeOut)
def get_employee(emp_id: int, db: Session = Depends(get_db)):
    employee = crud.get_employee(db, emp_id)
    if employee is None:                    # crud returns None if not found
        raise HTTPException(status_code=404, detail='Employee Not Found')
    return employee
```

---

## 9. Dependency Injection — get_db()

```python
def get_db():
    db = SessionLocal()   # Open a database session
    try:
        yield db          # Give it to the endpoint function
    finally:
        db.close()        # ALWAYS close, even if there's an error
```

**What is `yield`?**

`yield` turns this into a **generator**. FastAPI uses it like this:
1. Before the endpoint runs → everything BEFORE `yield` executes (`db = SessionLocal()`)
2. `yield db` → gives the database session to the endpoint
3. After the endpoint finishes → everything AFTER `yield` executes (`db.close()`)

**What is `Depends(get_db)`?**

It tells FastAPI: "Before running this endpoint, call `get_db()`, get me a database session,
and clean it up when I'm done."

```python
def get_employee(emp_id: int, db: Session = Depends(get_db)):
    #                          ↑
    #    "Hey FastAPI, I need a database session please"
```

This is called **dependency injection** — instead of creating `db` inside every function,
FastAPI injects it for you and handles cleanup automatically.

---

# Module 5: Machine Learning Integration

How do we take a Python machine learning model (like scikit-learn) and put it on the web?

## 1. The ML Workflow

You **don't** train the model inside the API. That would be too slow!

1.  **Train** the model offline (in Jupyter / script).
2.  **Save** the model to a file (serialization).
3.  **Load** the model in your FastAPI app.
4.  **Predict** when a user sends data.

```
Training Phase (Offline)        API Phase (Online)
┌──────────────────────┐        ┌──────────────────────┐
│  Data → Train.py     │        │  User Request        │
│          ↓           │        │       ↓              │
│     Trained Model    │        │   FastAPI App        │
│          ↓           │        │       ↓              │
│    model.joblib      │───────>│ Load model.joblib    │
└──────────────────────┘        │       ↓              │
                                │   Predict & Return   │
                                └──────────────────────┘
```

---

## 2. Serializing Models (Pickle/Joblib)

**Serialization** = converting a Python object (like a trained model) into a stream of bytes (a file) so you can save it.

**Deserialization** = loading those bytes back into a Python object.

### Common Tools:

| Tool | Best For | Note |
|------|----------|------|
| **Pickle** | General Python objects | Standard library, but usage can be insecure (don't unpickle untrusted data!) |
| **Joblib** | Scikit-Learn models | Faster for large numpy arrays (what models are made of) |
| **TensorFlow/Keras** | Deep Learning | Use `.h5` or `.keras` formats |
| **PyTorch** | Deep Learning | Use `.pt` or `.pth` formats |

### Example (Saving):
```python
# train.py
import joblib
from sklearn.linear_model import LinearRegression

# ... train your model ...
model = LinearRegression().fit(X, y)

# Save it!
joblib.dump(model, 'model.joblib')
```

---

## 3. Building the API

### A. The Schema (Input Validation)
Just like CRUD, we use Pydantic to ensure the input features are correct.

```python
# schemas.py
class HousingFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: int = Field(..., gt=0)
    total_rooms: int
    median_income: float
    # ... all other features the model needs
```

### B. The Prediction Logic
Load the model **once** when the app starts, not every time a user requests needed.

```python
# predict.py
import joblib
import numpy as np

# LOAD ONCE (Global Scope)
model = joblib.load('model.joblib')

def make_prediction(data: dict):
    # Convert dictionary input to the 2D array the model expects
    # Example: [[-122.23, 37.88, 41, 880, ...]]
    features = np.array([[
        data['longitude'], 
        data['latitude'], 
        data['housing_median_age'], 
        # ... retain correct order!
    ]])
    
    # Predict returns an array like [452600.00], we want the first item
    return model.predict(features)[0]
```

### C. The API Endpoint
```python
# main.py
from fastapi import FastAPI
from schemas import HousingFeatures, PredictionOut
from predict import make_prediction

app = FastAPI()

@app.post("/predict")
def predict_price(features: HousingFeatures):
    # 1. Convert Pydantic model to dict
    data = features.model_dump()
    
    # 2. Run prediction
    price = make_prediction(data)
    
    # 3. Return result
    return {"predicted_price": price}
```

---

## 4. Why Load Globally?

**BAD implementation:**
```python
@app.post("/predict")
def predict(features: HousingFeatures):
    model = joblib.load('model.joblib')  # ⚠️ WRONG!
    return model.predict(...)
```
> **Why?** If your model is 500MB, you are reading 500MB from disk **every single time** a user makes a request. Your server will crash under load.

**GOOD implementation:**
```python
# Load at top level
model = joblib.load('model.joblib')  # ✅ Correct

@app.post("/predict")
def predict(features: HousingFeatures):
    return model.predict(...)
```
> **Why?** The model is loaded into memory **once** when the server starts. Requests just use the existing object. Fast! ⚡

---

# Step-by-Step: Build ANY CRUD API from Scratch

Here's a **generalized recipe** you can follow for any resource (Users, Products, Books, etc.):

## Step 1: Install Dependencies

```bash
pip install fastapi uvicorn sqlalchemy "pydantic[email]"
```

## Step 2: Create Project Structure

```
my-api/
├── database.py    # DB connection
├── models.py      # Table definitions
├── schemas.py     # Request/Response shapes
├── crud.py        # DB operations
└── main.py        # API endpoints
```

## Step 3: Setup Database Connection (`database.py`)

```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Change this URL for different databases
DATABASE_URL = "sqlite:///./app.db"

engine = create_engine(DATABASE_URL, connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()
```

> 💡 Replace `"sqlite:///./app.db"` with PostgreSQL/MySQL URL for production.

## Step 4: Define Your Table (`models.py`)

Replace `YourResource` and columns with whatever you're building:

```python
from sqlalchemy import Column, Integer, String, Float, Boolean
from database import Base

class YourResource(Base):
    __tablename__ = 'your_resources'

    id = Column(Integer, primary_key=True, index=True)
    # Add your columns here:
    title = Column(String, index=True)
    description = Column(String)
    price = Column(Float)
    is_active = Column(Boolean, default=True)
```

### Common Column Types:
| Type | Python Equivalent | Example |
|------|------------------|---------|
| `Integer` | `int` | id, age, quantity |
| `String` | `str` | name, email, title |
| `Float` | `float` | price, rating |
| `Boolean` | `bool` | is_active, is_admin |
| `DateTime` | `datetime` | created_at, updated_at |
| `Text` | `str` (long) | description, content |

## Step 5: Define Data Shapes (`schemas.py`)

```python
from pydantic import BaseModel
from typing import Optional

# Shared fields (used by Create, Update, and Out)
class ResourceBase(BaseModel):
    title: str
    description: Optional[str] = None
    price: float
    is_active: bool = True

# For creating — user sends this (no ID)
class ResourceCreate(ResourceBase):
    pass

# For updating — user sends this (no ID, fields optional)
class ResourceUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    is_active: Optional[bool] = None

# For responses — API returns this (with ID)
class ResourceOut(ResourceBase):
    id: int

    class Config:
        orm_mode = True
```

## Step 6: Write CRUD Operations (`crud.py`)

```python
from sqlalchemy.orm import Session
import models, schemas

# CREATE
def create_resource(db: Session, resource: schemas.ResourceCreate):
    db_resource = models.YourResource(**resource.model_dump())
    # ↑ model_dump() converts Pydantic → dict, then ** unpacks it into keyword args
    db.add(db_resource)
    db.commit()
    db.refresh(db_resource)
    return db_resource

# READ ALL
def get_resources(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.YourResource).offset(skip).limit(limit).all()

# READ ONE
def get_resource(db: Session, resource_id: int):
    return db.query(models.YourResource).filter(
        models.YourResource.id == resource_id
    ).first()

# UPDATE
def update_resource(db: Session, resource_id: int, resource: schemas.ResourceUpdate):
    db_resource = db.query(models.YourResource).filter(
        models.YourResource.id == resource_id
    ).first()
    if db_resource:
        # Only update fields that were provided (not None)
        update_data = resource.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_resource, key, value)
        db.commit()
        db.refresh(db_resource)
    return db_resource

# DELETE
def delete_resource(db: Session, resource_id: int):
    db_resource = db.query(models.YourResource).filter(
        models.YourResource.id == resource_id
    ).first()
    if db_resource:
        db.delete(db_resource)
        db.commit()
    return db_resource
```

> 💡 **Pro tips used here:**
> - `**resource.model_dump()` — converts Pydantic model to dict and unpacks as keyword arguments
> - `exclude_unset=True` — only includes fields the user actually sent (for partial updates)
> - `setattr(obj, key, value)` — dynamically sets attributes (avoids writing each field manually)

## Step 7: Wire Up Endpoints (`main.py`)

```python
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from database import engine, SessionLocal, Base
from typing import List
import models, schemas, crud

# Create all tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# CREATE
@app.post('/resources', response_model=schemas.ResourceOut, status_code=201)
def create(resource: schemas.ResourceCreate, db: Session = Depends(get_db)):
    return crud.create_resource(db, resource)

# READ ALL
@app.get('/resources', response_model=List[schemas.ResourceOut])
def read_all(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return crud.get_resources(db, skip=skip, limit=limit)

# READ ONE
@app.get('/resources/{resource_id}', response_model=schemas.ResourceOut)
def read_one(resource_id: int, db: Session = Depends(get_db)):
    resource = crud.get_resource(db, resource_id)
    if resource is None:
        raise HTTPException(status_code=404, detail='Resource not found')
    return resource

# UPDATE
@app.put('/resources/{resource_id}', response_model=schemas.ResourceOut)
def update(resource_id: int, resource: schemas.ResourceUpdate, db: Session = Depends(get_db)):
    db_resource = crud.update_resource(db, resource_id, resource)
    if db_resource is None:
        raise HTTPException(status_code=404, detail='Resource not found')
    return db_resource

# DELETE
@app.delete('/resources/{resource_id}')
def delete(resource_id: int, db: Session = Depends(get_db)):
    resource = crud.delete_resource(db, resource_id)
    if resource is None:
        raise HTTPException(status_code=404, detail='Resource not found')
    return {'detail': 'Resource deleted'}
```

## Step 8: Run and Test

```bash
cd my-api
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000/docs** for Swagger UI to test all endpoints!

---

## Quick Reference Cheat Sheet

```
ENDPOINT PATTERN:
┌──────────────────────────────────────────────────┐
│ @app.METHOD('/path', response_model=SchemaOut)   │
│ def handler(data: SchemaCreate, db = Depends()):│
│     result = crud.operation(db, data)            │
│     if not result: raise HTTPException(404)      │
│     return result                                │
└──────────────────────────────────────────────────┘

CRUD FUNCTION PATTERN:
┌──────────────────────────────────────────────────┐
│ def operation(db: Session, ...):                 │
│     1. Query the database                        │
│     2. Do something (add/modify/delete)          │
│     3. db.commit()                               │
│     4. db.refresh() if needed                    │
│     5. return result                             │
└──────────────────────────────────────────────────┘

FILE RESPONSIBILITIES:
┌──────────────┬───────────────────────────────────┐
│ database.py  │ Connection setup (engine, session) │
│ models.py    │ Table structure (columns, types)   │
│ schemas.py   │ API data shapes (input/output)     │
│ crud.py      │ DB operations (add/get/update/del) │
│ main.py      │ Endpoints + error handling         │
└──────────────┴───────────────────────────────────┘
```
