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
    
