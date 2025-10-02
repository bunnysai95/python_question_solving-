# from fastapi import FastAPI #, HTTPException, status,Depands
# from fastapi.middleware.cors import CORSMiddleware
# from tortoise.contrib.fastapi import register_tortoise


# from settings import Settings
# from models import User
# from security import hash_password, verify_password, create_access_token, get_current_username
# # from security import hash_password, verify_password, create_access_token, decode_access_token, get_current_username
# from schemas import RegisterIn, UserOut, LoginIn, TokenOut, MeOut, ProfileOut
# # from schemas import RegisterIn,UserOut, LoginIn, TokenOut,MeOut,ProfileOut

# # forms after login page

# from typing import Literal
# from datatime import date
# from pathlib import Path
# from uuid import uuid4
# import re

# from fastapi import UploadFile, File, Form

# app = FastAPI(title = settings.APP_NAME)

# # CORS so Vite url calling 
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins = [
#         "http://localhost:5173",
#         "http://127.0.0.1:5173",
#     ],
#     allow_origin_regex = r"^https://.*\.github\.dev$",
#     allow_credentials = True,
#     allow_methods = ["*"],
#     allow_headers = ["*"],
# )

# # routes
# @app.post("/api/register", response_model = UserOut, status_code = status.HTTP_201_CREATED)
# async def register():
#     exists = await User.filter(username = payload.username).exists()
#     if exists:
#         raise HTTPException(status_code = 409, detail = "username already taken")    
#     user = await User.created(
#         username = payload.username,
#         first_name = payload.firstname,
#         last_name = payload.lastname,
#         dob = payload.dob,
#         phone = payload.phone,
#         password_hash = hash_password(payload.password),
#     )

#     return UserOut(
#         id=user.id,
#         username=user.username,
#         firstName=user.first_name,
#         lastName=user.last_name,
#         dob=user.dob,
#         phone=user.phone,
#     )
    
# # comit for 1 day purpose delte later  below this line 
# # 1) Linear search
# # Time: O(n)   Space: O(1)
# def linear_search(arr, target):
#     for i, v in enumerate(arr):
#         if v == target:
#             return i
#     return -1

# # 2) Binary search (array must be sorted)
# # Time: O(log n)   Space: O(1)
# def binary_search(arr, target):
#     lo, hi = 0, len(arr) - 1
#     while lo <= hi:
#         mid = (lo + hi) // 2
#         if arr[mid] == target:
#             return mid
#         if arr[mid] < target:
#             lo = mid + 1
#         else:
#             hi = mid - 1
#     return -1

# # 3) Bubble sort (educational)
# # Time: O(n^2)   Space: O(1)
# def bubble_sort(arr):
#     n = len(arr)
#     a = arr[:]  # copy so original not mutated
#     for i in range(n):
#         swapped = False
#         for j in range(0, n - 1 - i):
#             if a[j] > a[j + 1]:
#                 a[j], a[j + 1] = a[j + 1], a[j]
#                 swapped = True
#         if not swapped:
#             break
#     return a

# # 4) Merge sort (divide & conquer)
# # Time: O(n log n)   Space: O(n)
# def merge_sort(arr):
#     if len(arr) <= 1:
#         return arr
#     mid = len(arr) // 2
#     left = merge_sort(arr[:mid])
#     right = merge_sort(arr[mid:])
#     # merge
#     i = j = 0
#     merged = []
#     while i < len(left) and j < len(right):
#         if left[i] <= right[j]:
#             merged.append(left[i]); i += 1
#         else:
#             merged.append(right[j]); j += 1
#     merged.extend(left[i:])
#     merged.extend(right[j:])
#     return merged

# # 5) Quick sort (in-place, average O(n log n), worst O(n^2))
# def quick_sort(arr):
#     a = arr[:]  # avoid mutating original
#     def _qs(lo, hi):
#         if lo >= hi: return
#         pivot = a[(lo+hi)//2]
#         i, j = lo, hi
#         while i <= j:
#             while a[i] < pivot: i += 1
#             while a[j] > pivot: j -= 1
#             if i <= j:
#                 a[i], a[j] = a[j], a[i]
#                 i += 1; j -= 1
#         _qs(lo, j)
#         _qs(i, hi)
#     _qs(0, len(a)-1)
#     return a

# # 6) Fibonacci: naive recursion vs DP
# # naive recursion: O(2^n) exponential (DON'T USE)
# def fib_naive(n):
#     if n <= 1: return n
#     return fib_naive(n-1) + fib_naive(n-2)

# # DP (bottom-up): O(n) time, O(1) space
# def fib_dp(n):
#     if n <= 1: return n
#     a, b = 0, 1
#     for _ in range(2, n+1):
#         a, b = b, a + b
#     return b

# # 7) Dijkstra (shortest path from source) using heap
# # Time: O((V + E) log V) with adjacency list
# import heapq
# def dijkstra(adj, source):
#     # adj: dict[node] = list of (neighbor, weight)
#     dist = {u: float('inf') for u in adj}
#     dist[source] = 0
#     pq = [(0, source)]
#     while pq:
#         d, u = heapq.heappop(pq)
#         if d > dist[u]: continue
#         for v, w in adj[u]:
#             nd = d + w
#             if nd < dist[v]:
#                 dist[v] = nd
#                 heapq.heappush(pq, (nd, v))
#     return dist

# # 8) 0/1 Knapsack (classic DP)
# # Time: O(n * W)   Space: O(W) optimized
# def knapsack(values, weights, W):
#     n = len(values)
#     dp = [0] * (W + 1)
#     for i in range(n):
#         # iterate weight backwards to avoid reuse
#         for w in range(W, weights[i] - 1, -1):
#             dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
#     return dp[W]


# -------------------------------------------------------------

from FastAPI import FastAPI, HTTPException, status, Depends
from FastAPI.middleware.cors import CORSMiddleware
from tortoise.contrib.FastAPI import register_tortoise
from settings import settings
from models import USER, Profile
from schemas import RegisterIn, UserOut,LOginIn, TokenOut,Meout,PorfileOut,ChatRequest, ChatRespone
from security import hash_password, verify_password, create_acess_token, get_current_username

# for chat bot 
import asyncio
from typing import List 
# forms after login page 
fromo typing import Literal 
from datetime import date 
from pathlib import path 
from uuid import uuid4
import re 
from fastapi import UploadFile, File, Form
app = FastAPI(title = setting.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_orgins =[
        "https://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_orgins_regex =r"^https://.*\.github\.dev$",
    allow_credentials = True,
    allow_headers= ["*"],
    allow_methods =["*"],
)
# routes
@app.post("/api/register", response_model= UserOut, status_code = status.HTTP_201_CREATED)
async def(payload:RegisterIn):
    exists = awaite User.filter(username = payload.username).exists()
    if exists:
        raise HTTPException(status_code = 409, detail = "Username already taken ")

    user = await user.create(
        username = payload.username,
        first_name = payload.firsrname,
        last_name = payload.lastname,
        dob = payload.dob,
        phone =payload.phone,
        password_hash = hash_password(payload.password)
    )
    return Userout(
        id = user.id,
        username = user.username,
        firstName = user.first_name,
        lastName = user.last_name,
        dob = user.dob,
        phone = user.phone,
    )
@app.post("/api/login", response_model= TokenOut)
async def login(payload:LoginIn):
    user = await.User.get_or_none(username = payload.username)
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPEXception(status_code =401 , detail = "Invalid username or password")
    token = create_acess_token(sub = user.username)
    return TokenOut()

@app.get("api/me", response_model = MeOut)
async def me(current_username: str = Depends(get_current_username)):
    user = await user.get_or_none(username = current_username)
    if not user:
        raise HTTPException(status_code = 404, detail = "user not found")
    return meout(username = user.username, firstname = user.first-name, lastname = user.last_name)

@app.get("/health")
def health():
    return {"ok": True}

# db creating

register_tortoise(
    app,
    db_url = settings.DB_URL,
    modules = {"models":["models"]},
    generate_schemas = True,
    add_exception_handlers = True,
)

def _safe_filename(name:str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]","_", name)
UPLOAD_DIR = Path("upload")
UPLOAD_DIR.mkdir(exists_ok = True)

#profile creation
@app.post()
async def():
    return ProfileOut()