# Calculate the multiplication and sum of two numbers

input_variable = 10 #int(input(("please enter random number 1")))
input_variable_2 =10 #int(input(("please enter random number 2 ")))

print(input_variable+input_variable_2)
print(input_variable*input_variable_2)

# Print the Sum of a Current Number and a Previous number
for i in range(0,10):
    print( input_variable+input_variable_2, "and Previous number is ", input_variable)

print('this will be the 3rd in put that i m sharing with git ')

prices = [7, 1, 5, 3, 6, 4]

def max_profits(prices):
    if prices !=0:
        min_price = prices[0]
        max_profit = 0
        for price in prices[1:]:
            profit = price - min_price
            max_profit = max(max_profit, profit)
            min_price = min(min_price, price)
    else:
        return 0
    return max_profit
# max_sales = max_profits(prices)
print(max_profits(prices))


nums = [2, 7, 11, 15]
target = 9
def two_sum(nums, target):
    seen ={}
    for i, num in enumerate(nums):
        needed = target - num
        if needed in seen:
            return [seen[needed],i]
        seen[num] = i
        
print(two_sum([2, 7, 11, 15], 9)) 

<<<<<<< HEAD
def second_largest(nums):
    # Handle case if list has less than 2 elements
    if len(nums) < 2:
        return None
    
    first = second = float('-inf')  # Initialize with very small numbers
    
    for num in nums:
        if num > first:
            second = first   # update second before updating first
            first = num
        elif first > num > second:
            second = num
    
    return second if second != float('-inf') else None


# Example usage
nums = [12, 45, 2, 41, 31, 10]
print(second_largest(nums))  # Output: 41
=======



import random

# Dictionary of questions and answers
qa_bank = {
    "What is the capital of France?": "Paris",
    "Who developed the theory of relativity?": "Albert Einstein",
    "What is 9 x 9?": "81",
    "Which language is this program written in?": "Python",
    "What is the largest planet in our solar system?": "Jupiter",
    "Who painted the Mona Lisa?": "Leonardo da Vinci",
    "What does CPU stand for?": "Central Processing Unit",
    "What is the chemical symbol for water?": "H2O",
    "Who wrote 'Hamlet'?": "William Shakespeare",
    "What is the square root of 144?": "12"
}

# Function to get a random question and answer
def random_question():
    question = random.choice(list(qa_bank.keys()))
    answer = qa_bank[question]
    return question, answer

# Example usage
if __name__ == "__main__":
    for _ in range(5):  # generate 5 random Q&As
        q, a = random_question()
        print(f"Q: {q}")
        print(f"A: {a}\n")
        
        
# task_manager.py

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from datetime import datetime

# ---------------------------
# Database setup
# ---------------------------
DATABASE_URL = "sqlite:///./tasks.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ---------------------------
# Models
# ---------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)

    tasks = relationship("Task", back_populates="owner")


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String)
    completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="tasks")


# Create tables
Base.metadata.create_all(bind=engine)

# ---------------------------
# Pydantic Schemas
# ---------------------------
class TaskBase(BaseModel):
    title: str
    description: Optional[str] = None
    completed: Optional[bool] = False


class TaskCreate(TaskBase):
    pass


class TaskUpdate(TaskBase):
    completed: Optional[bool] = None


class TaskOut(TaskBase):
    id: int
    created_at: datetime
    updated_at: datetime
    owner_id: int

    class Config:
        orm_mode = True


class UserBase(BaseModel):
    username: str
    email: str


class UserCreate(UserBase):
    pass


class UserOut(UserBase):
    id: int
    tasks: List[TaskOut] = []

    class Config:
        orm_mode = True

# ---------------------------
# Dependency
# ---------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Task Manager API", version="1.0")

# Enable CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Routes - Users
# ---------------------------
@app.post("/users/", response_model=UserOut)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    db_email = db.query(User).filter(User.email == user.email).first()
    if db_email:
        raise HTTPException(status_code=400, detail="Email already exists")

    new_user = User(username=user.username, email=user.email)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.get("/users/", response_model=List[UserOut])
def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()


@app.get("/users/{user_id}", response_model=UserOut)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# ---------------------------
# Routes - Tasks
# ---------------------------
@app.post("/users/{user_id}/tasks/", response_model=TaskOut)
def create_task(user_id: int, task: TaskCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    new_task = Task(
        title=task.title,
        description=task.description,
        completed=task.completed,
        owner_id=user_id,
    )
    db.add(new_task)
    db.commit()
    db.refresh(new_task)
    return new_task


@app.get("/tasks/", response_model=List[TaskOut])
def get_tasks(db: Session = Depends(get_db)):
    return db.query(Task).all()


@app.get("/tasks/{task_id}", response_model=TaskOut)
def get_task(task_id: int, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.put("/tasks/{task_id}", response_model=TaskOut)
def update_task(task_id: int, task_update: TaskUpdate, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    task.title = task_update.title or task.title
    task.description = task_update.description or task.description
    if task_update.completed is not None:
        task.completed = task_update.completed
    task.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(task)
    return task


@app.delete("/tasks/{task_id}")
def delete_task(task_id: int, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    db.delete(task)
    db.commit()
    return {"detail": "Task deleted successfully"}

# 1) Linear search
# Time: O(n)   Space: O(1)
def linear_search(arr, target):
    for i, v in enumerate(arr):
        if v == target:
            return i
    return -1

# 2) Binary search (array must be sorted)
# Time: O(log n)   Space: O(1)
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

# 3) Bubble sort (educational)
# Time: O(n^2)   Space: O(1)
def bubble_sort(arr):
    n = len(arr)
    a = arr[:]  # copy so original not mutated
    for i in range(n):
        swapped = False
        for j in range(0, n - 1 - i):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:
            break
    return a

# 4) Merge sort (divide & conquer)
# Time: O(n log n)   Space: O(n)
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    # merge
    i = j = 0
    merged = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i]); i += 1
        else:
            merged.append(right[j]); j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

# 5) Quick sort (in-place, average O(n log n), worst O(n^2))
def quick_sort(arr):
    a = arr[:]  # avoid mutating original
    def _qs(lo, hi):
        if lo >= hi: return
        pivot = a[(lo+hi)//2]
        i, j = lo, hi
        while i <= j:
            while a[i] < pivot: i += 1
            while a[j] > pivot: j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                i += 1; j -= 1
        _qs(lo, j)
        _qs(i, hi)
    _qs(0, len(a)-1)
    return a

# 6) Fibonacci: naive recursion vs DP
# naive recursion: O(2^n) exponential (DON'T USE)
def fib_naive(n):
    if n <= 1: return n
    return fib_naive(n-1) + fib_naive(n-2)

# DP (bottom-up): O(n) time, O(1) space
def fib_dp(n):
    if n <= 1: return n
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b

# 7) Dijkstra (shortest path from source) using heap
# Time: O((V + E) log V) with adjacency list
import heapq
def dijkstra(adj, source):
    # adj: dict[node] = list of (neighbor, weight)
    dist = {u: float('inf') for u in adj}
    dist[source] = 0
    pq = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

# 8) 0/1 Knapsack (classic DP)
# Time: O(n * W)   Space: O(W) optimized
def knapsack(values, weights, W):
    n = len(values)
    dp = [0] * (W + 1)
    for i in range(n):
        # iterate weight backwards to avoid reuse
        for w in range(W, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[W]
    

# 1) Palindrome check (string)
# Time: O(n), Space: O(1)
def is_palindrome(s):
    return s == s[::-1]

# 2) Anagram check
# Time: O(n log n) if using sort, O(n) with counter
def are_anagrams(s1, s2):
    return sorted(s1) == sorted(s2)

# 3) Two Sum (find if two numbers add up to target)
# Time: O(n), Space: O(n)
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
    return None

# 4) BFS (Breadth-First Search) on graph
# Time: O(V + E), Space: O(V)
from collections import deque
def bfs(graph, start):
    visited = set([start])
    q = deque([start])
    order = []
    while q:
        node = q.popleft()
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
    return order

# 5) DFS (Depth-First Search) on graph
# Time: O(V + E), Space: O(V)
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited

# 6) Binary Tree Traversals
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

# Inorder Traversal (L, Root, R)
# Time: O(n), Space: O(h) (h = tree height)
def inorder(root):
    return inorder(root.left) + [root.val] + inorder(root.right) if root else []

# Preorder Traversal (Root, L, R)
def preorder(root):
    return [root.val] + preorder(root.left) + preorder(root.right) if root else []

# Postorder Traversal (L, R, Root)
def postorder(root):
    return postorder(root.left) + postorder(root.right) + [root.val] if root else []

# 7) Find factorial (recursive and iterative)
# Recursive: O(n), Iterative: O(n)
def factorial_recursive(n):
    return 1 if n == 0 else n * factorial_recursive(n-1)

def factorial_iterative(n):
    result = 1
    for i in range(2, n+1):
        result *= i
    return result
    
        

    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top IT Cities Scoring & Explorer (self-contained)
- 25 U.S. tech/IT cities with illustrative metrics
- Rank with custom weights (salary, cost-of-living, growth, postings, remote, ecosystem)
- Filter by state, thresholds; export CSV/JSON; optional bar chart
Usage examples:
  python it_cities.py --top 15
  python it_cities.py --w-salary 2.0 --w-col -1.5 --w-growth 1.2 --plot --top 12
  python it_cities.py --state TX --min-salary 70 --max-col 115 --export out.csv --export-json out.json
Note: Metrics are illustrative indices to let you practice ranking/analysis.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import argparse
import math
import csv
import json
import sys
import shutil
import statistics
from pathlib import Path

# Optional plotting (no seaborn; single-plot rule; avoid explicit colors)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False


@dataclass
class City:
    name: str
    state: str
    # Indices (roughly 0-100+). Higher is better except cost_of_living.
    # These are illustrative synthetic values for demo/learning.
    salary_index: float          # higher = better pay
    cost_of_living_index: float  # lower = cheaper
    postings_index: float        # relative job postings/activity
    growth_index: float          # recent/near-term growth momentum
    remote_share_index: float    # remote-friendly roles share
    ecosystem_index: float       # startups, VCs, R&D, anchor companies

CITIES: List[City] = [
    City("San Francisco", "CA", 100, 175, 92, 78, 68, 98),
    City("San Jose", "CA", 99, 170, 80, 75, 60, 97),
    City("New York", "NY", 95, 165, 96, 70, 62, 94),
    City("Seattle", "WA", 96, 150, 88, 74, 66, 93),
    City("Washington DC", "DC", 91, 150, 84, 72, 58, 90),
    City("Austin", "TX", 88, 120, 83, 88, 70, 89),
    City("Boston", "MA", 92, 145, 82, 73, 57, 92),
    City("Dallas–Fort Worth", "TX", 86, 115, 85, 77, 56, 84),
    City("Denver", "CO", 85, 125, 74, 79, 63, 83),
    City("Raleigh–Durham", "NC", 83, 110, 70, 82, 55, 82),
    City("Atlanta", "GA", 84, 112, 78, 80, 54, 81),
    City("San Diego", "CA", 87, 140, 69, 68, 52, 80),
    City("Charlotte", "NC", 82, 108, 68, 76, 50, 78),
    City("Miami", "FL", 81, 125, 72, 74, 59, 79),
    City("Baltimore", "MD", 79, 120, 63, 65, 48, 74),
    City("Houston", "TX", 83, 113, 76, 66, 49, 75),
    City("Philadelphia", "PA", 80, 120, 67, 63, 47, 73),
    City("Los Angeles", "CA", 88, 150, 79, 67, 58, 86),
    City("Plano", "TX", 81, 110, 66, 70, 46, 72),
    City("Jersey City", "NJ", 83, 135, 65, 60, 51, 76),
    City("Pittsburgh", "PA", 77, 105, 57, 62, 44, 70),
    City("Phoenix", "AZ", 79, 108, 71, 73, 53, 74),
    City("Chicago", "IL", 85, 120, 90, 69, 61, 88),
    City("San Antonio", "TX", 76, 105, 60, 64, 45, 69),
    City("Columbus", "OH", 80, 104, 64, 78, 52, 77),
]

def zscore(values: List[float]) -> List[float]:
    if len(values) <= 1:
        return [0.0 for _ in values]
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values)
    if stdev == 0:
        return [0.0 for _ in values]
    return [(v - mean) / stdev for v in values]

def minmax(values: List[float]) -> List[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]

def normalize_city_metrics(cities: List[City], method: str = "z") -> Dict[str, Dict[str, float]]:
    """Return normalized metrics per city keyed by city name."""
    metrics = {
        "salary_index": [c.salary_index for c in cities],
        "cost_of_living_index": [c.cost_of_living_index for c in cities],
        "postings_index": [c.postings_index for c in cities],
        "growth_index": [c.growth_index for c in cities],
        "remote_share_index": [c.remote_share_index for c in cities],
        "ecosystem_index": [c.ecosystem_index for c in cities],
    }
    normed = {}
    normalizer = zscore if method == "z" else minmax
    normed_metrics: Dict[str, List[float]] = {k: normalizer(v) for k, v in metrics.items()}

    # Invert cost-of-living (lower is better -> higher normalized score)
    if method == "z":
        normed_metrics["cost_of_living_index"] = [-x for x in normed_metrics["cost_of_living_index"]]
    else:
        # minmax 0(lowest COL) -> 1(highest COL), invert to make lower COL better
        normed_metrics["cost_of_living_index"] = [1 - x for x in normed_metrics["cost_of_living_index"]]

    for i, c in enumerate(cities):
        normed[c.name] = {
            "salary": normed_metrics["salary_index"][i],
            "col": normed_metrics["cost_of_living_index"][i],
            "postings": normed_metrics["postings_index"][i],
            "growth": normed_metrics["growth_index"][i],
            "remote": normed_metrics["remote_share_index"][i],
            "ecosys": normed_metrics["ecosystem_index"][i],
        }
    return normed

def compute_scores(
    cities: List[City],
    w_salary: float = 1.0,
    w_col: float = 1.0,
    w_growth: float = 1.0,
    w_postings: float = 1.0,
    w_remote: float = 0.6,
    w_ecosys: float = 1.0,
    norm: str = "z"
) -> List[Tuple[City, float, Dict[str, float]]]:
    normed = normalize_city_metrics(cities, method=norm)
    results = []
    for c in cities:
        nm = normed[c.name]
        score = (
            w_salary * nm["salary"] +
            w_col * nm["col"] +
            w_growth * nm["growth"] +
            w_postings * nm["postings"] +
            w_remote * nm["remote"] +
            w_ecosys * nm["ecosys"]
        )
        results.append((c, score, nm))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def tabulate_rows(rows: List[List[Any]], headers: List[str]) -> str:
    """
    Simple table printer (no external deps).
    Falls back to reasonable spacing based on terminal width.
    """
    # Compute column widths
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(cell)) for cell in col) for col in cols]
    def fmt_row(r): return " | ".join(str(cell).ljust(w) for cell, w in zip(r, widths))
    sep = "-+-".join("-" * w for w in widths)
    parts = [fmt_row(headers), sep]
    parts.extend(fmt_row(r) for r in rows)
    return "\n".join(parts)

def export_csv(path: Path, ranked: List[Tuple[City, float, Dict[str, float]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "city", "state", "score",
            "salary_index", "cost_of_living_index", "postings_index",
            "growth_index", "remote_share_index", "ecosystem_index",
            "norm_salary", "norm_col", "norm_postings", "norm_growth", "norm_remote", "norm_ecosys"
        ])
        for i, (c, score, nm) in enumerate(ranked, start=1):
            writer.writerow([
                i, c.name, c.state, round(score, 4),
                c.salary_index, c.cost_of_living_index, c.postings_index,
                c.growth_index, c.remote_share_index, c.ecosystem_index,
                round(nm["salary"], 4), round(nm["col"], 4), round(nm["postings"], 4),
                round(nm["growth"], 4), round(nm["remote"], 4), round(nm["ecosys"], 4)
            ])

def export_json(path: Path, ranked: List[Tuple[City, float, Dict[str, float]]]) -> None:
    data = []
    for i, (c, score, nm) in enumerate(ranked, start=1):
        row = {
            "rank": i,
            "city": c.name,
            "state": c.state,
            "score": score,
            "metrics": asdict(c),
            "normalized": nm,
        }
        data.append(row)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def plot_bar(ranked: List[Tuple[City, float, Dict[str, float]]], top: int) -> None:
    if not MATPLOTLIB_OK:
        print("Plotting skipped: matplotlib not available.", file=sys.stderr)
        return
    topn = ranked[:top]
    labels = [f"{c.name}, {c.state}" for c, _, _ in topn]
    scores = [s for _, s, _ in topn]
    plt.figure(figsize=(12, 6))
    plt.barh(labels, scores)                # no explicit colors/styles
    plt.gca().invert_yaxis()                 # highest on top
    plt.xlabel("Composite Score (higher is better)")
    plt.title(f"Top {top} IT Cities — Composite Score")
    plt.tight_layout()
    plt.show()

def filter_cities(
    cities: List[City],
    state: Optional[str] = None,
    min_salary: Optional[float] = None,
    max_col: Optional[float] = None,
    min_growth: Optional[float] = None,
    min_postings: Optional[float] = None,
    min_remote: Optional[float] = None,
    min_ecosys: Optional[float] = None,
) -> List[City]:
    res = []
    for c in cities:
        if state and c.state.upper() != state.upper():
            continue
        if min_salary is not None and c.salary_index < min_salary:
            continue
        if max_col is not None and c.cost_of_living_index > max_col:
            continue
        if min_growth is not None and c.growth_index < min_growth:
            continue
        if min_postings is not None and c.postings_index < min_postings:
            continue
        if min_remote is not None and c.remote_share_index < min_remote:
            continue
        if min_ecosys is not None and c.ecosystem_index < min_ecosys:
            continue
        res.append(c)
    return res

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank and explore U.S. IT cities.")
    p.add_argument("--top", type=int, default=25, help="How many results to show (default: 25)")
    p.add_argument("--state", type=str, help="Filter by 2-letter state (e.g., TX, CA)")
    p.add_argument("--min-salary", type=float, help="Minimum salary index filter")
    p.add_argument("--max-col", type=float, help="Maximum cost-of-living index filter")
    p.add_argument("--min-growth", type=float, help="Minimum growth index filter")
    p.add_argument("--min-postings", type=float, help="Minimum postings index filter")
    p.add_argument("--min-remote", type=float, help="Minimum remote-share index filter")
    p.add_argument("--min-ecosys", type=float, help="Minimum ecosystem index filter")

    # Weights (positive = more important; negative flips preference)
    p.add_argument("--w-salary", type=float, default=1.0, help="Weight for salary (default 1.0)")
    p.add_argument("--w-col", type=float, default=1.0, help="Weight for cost-of-living (default 1.0; higher favors cheaper places)")
    p.add_argument("--w-growth", type=float, default=1.0, help="Weight for growth momentum (default 1.0)")
    p.add_argument("--w-postings", type=float, default=1.0, help="Weight for job postings/activity (default 1.0)")
    p.add_argument("--w-remote", type=float, default=0.6, help="Weight for remote share (default 0.6)")
    p.add_argument("--w-ecosys", type=float, default=1.0, help="Weight for ecosystem depth (default 1.0)")

    p.add_argument("--norm", choices=["z", "minmax"], default="z", help="Normalization scheme (default: z)")
    p.add_argument("--export", type=str, help="Export ranked results to CSV file")
    p.add_argument("--export-json", type=str, help="Export ranked results to JSON file")
    p.add_argument("--plot", action="store_true", help="Show bar chart of scores (requires matplotlib)")
    return p.parse_args()

def main():
    args = parse_args()

    # Filter dataset
    filtered = filter_cities(
        CITIES,
        state=args.state,
        min_salary=args.min_salary,
        max_col=args.max_col,
        min_growth=args.min_growth,
        min_postings=args.min_postings,
        min_remote=args.min_remote,
        min_ecosys=args.min_ecosys,
    )

    if not filtered:
        print("No cities matched your filter criteria.", file=sys.stderr)
        sys.exit(1)

    ranked = compute_scores(
        filtered,
        w_salary=args.w_salary,
        w_col=args.w_col,
        w_growth=args.w_growth,
        w_postings=args.w_postings,
        w_remote=args.w_remote,
        w_ecosys=args.w_ecosys,
        norm=args.norm
    )

    top = max(1, min(args.top, len(ranked)))

    # Prepare rows for printing
    rows = []
    for i, (c, score, nm) in enumerate(ranked[:top], start=1):
        rows.append([
            i,
            f"{c.name}, {c.state}",
            round(score, 3),
            c.salary_index,
            c.cost_of_living_index,
            c.postings_index,
            c.growth_index,
            c.remote_share_index,
            c.ecosystem_index,
        ])

    headers = ["Rank", "City", "Score", "SalaryIdx", "COLIdx", "PostingsIdx", "GrowthIdx", "RemoteIdx", "EcosysIdx"]
    print(tabulate_rows(rows, headers))

    # Exports
    if args.export:
        export_csv(Path(args.export), ranked)
        print(f"\nCSV exported to: {args.export}")

    if args.export_json:
        export_json(Path(args.export_json), ranked)
        print(f"JSON exported to: {args.export_json}")

    # Plot
    if args.plot:
        plot_bar(ranked, top=top)

if __name__ == "__main__":
    main()

"""
Inventory Management Microservice
Technologies: FastAPI, SQLAlchemy, PostgreSQL, JWT Auth, Docker
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from jose import JWTError, jwt
from passlib.context import CryptContext
import uvicorn
import datetime

# -------------------- CONFIG --------------------
DATABASE_URL = "sqlite:///./inventory.db"  # Replace with PostgreSQL in prod
SECRET_KEY = "SUPER_SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(title="Inventory Management API")

# -------------------- MODELS --------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    price = Column(Float)
    quantity = Column(Integer)

Base.metadata.create_all(bind=engine)

# -------------------- SCHEMAS --------------------
class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ItemCreate(BaseModel):
    name: str
    description: str
    price: float
    quantity: int

class ItemOut(ItemCreate):
    id: int

# -------------------- UTILS --------------------
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: datetime.timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + (expires_delta or datetime.timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------- AUTH --------------------
@app.post("/register", response_model=Token)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_pw = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# -------------------- CRUD --------------------
@app.post("/items/", response_model=ItemOut)
def create_item(item: ItemCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_item = Item(**item.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@app.get("/items/", response_model=list[ItemOut])
def read_items(skip: int = 0, limit: int = 10, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Item).offset(skip).limit(limit).all()

@app.put("/items/{item_id}", response_model=ItemOut)
def update_item(item_id: int, item: ItemCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_item = db.query(Item).filter(Item.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")
    for key, value in item.dict().items():
        setattr(db_item, key, value)
    db.commit()
    db.refresh(db_item)
    return db_item

@app.delete("/items/{item_id}")
def delete_item(item_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_item = db.query(Item).filter(Item.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Item not found")
    db.delete(db_item)
    db.commit()
    return {"msg": "Item deleted successfully"}

# -------------------- MAIN --------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


import os
import json
import time
import random
import asyncio
import logging
import threading
from typing import List, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# ------------------------- CONFIG & LOGGING -------------------------
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s"
)

CONFIG_FILE = "config.json"

# ------------------------- CONFIG MANAGER -------------------------
class ConfigManager:
    """Loads and saves application configuration."""

    def __init__(self, path: str):
        self.path = path
        self.config = self.load_config()

    def load_config(self) -> Dict:
        if os.path.exists(self.path):
            with open(self.path, 'r') as file:
                logging.info("Loaded existing config.")
                return json.load(file)
        else:
            logging.warning("No config found. Using default.")
            return {"version": 1.0, "theme": "dark", "users": []}

    def save_config(self):
        with open(self.path, 'w') as file:
            json.dump(self.config, file, indent=4)
        logging.info("Config saved successfully.")

# ------------------------- USER MODEL -------------------------
class User:
    """Represents a system user."""

    def __init__(self, username: str, email: str):
        self.username = username
        self.email = email
        self.created_at = datetime.now()

    def to_dict(self) -> Dict:
        return {
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at.isoformat()
        }

# ------------------------- USER SERVICE -------------------------
class UserService:
    """Handles user creation, retrieval, and persistence."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    def add_user(self, username: str, email: str):
        user = User(username, email)
        self.config_manager.config["users"].append(user.to_dict())
        self.config_manager.save_config()
        logging.info(f"User {username} added.")

    def list_users(self) -> List[Dict]:
        return self.config_manager.config.get("users", [])

# ------------------------- ASYNC API SIMULATION -------------------------
async def fake_api_call(user: Dict) -> Dict:
    """Simulates an async API call."""
    await asyncio.sleep(random.uniform(0.5, 2.0))
    logging.info(f"Fetched data for {user['username']}")
    return {"username": user["username"], "data": random.randint(1, 100)}

async def fetch_all_users_data(users: List[Dict]):
    """Fetches data for all users asynchronously."""
    tasks = [fake_api_call(user) for user in users]
    results = await asyncio.gather(*tasks)
    logging.info("All user data fetched.")
    return results

# ------------------------- THREADING EXAMPLE -------------------------
def background_task(name: str):
    """Runs a background computation."""
    logging.info(f"Background task {name} started.")
    time.sleep(random.randint(1, 4))
    logging.info(f"Background task {name} completed.")

def run_background_tasks():
    threads = []
    for i in range(3):
        t = threading.Thread(target=background_task, args=(f"Task-{i}",))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    logging.info("All background tasks done.")

# ------------------------- MAIN APP -------------------------
class App:
    """Main application orchestrating all components."""

    def __init__(self):
        self.config_manager = ConfigManager(CONFIG_FILE)
        self.user_service = UserService(self.config_manager)

    def run(self):
        # Add random users
        for i in range(5):
            self.user_service.add_user(f"user{i}", f"user{i}@example.com")

        # List users
        users = self.user_service.list_users()
        print("Registered Users:")
        for u in users:
            print(f"- {u['username']} ({u['email']})")

        # Run background tasks in parallel
        print("\nRunning background tasks...")
        run_background_tasks()

        # Fetch async data
        print("\nFetching async data for all users...")
        results = asyncio.run(fetch_all_users_data(users))
        print("API Results:")
        for r in results:
            print(f"{r['username']}: {r['data']}")

# ------------------------- ENTRY POINT -------------------------
if __name__ == "__main__":
    app = App()
    app.run()
    print("\nAll operations completed. Check app.log for details.")


