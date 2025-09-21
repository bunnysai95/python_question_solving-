import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import datetime
import os
import sqlite3
from dataclasses import dataclass, asdict
from typing import List, Optional
from enum import Enum
import threading
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class Priority(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    URGENT = "Urgent"

class Status(Enum):
    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    CANCELLED = "Cancelled"

@dataclass
class Task:
    id: int
    title: str
    description: str
    priority: Priority
    status: Status
    due_date: str
    created_date: str
    completed_date: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class DatabaseManager:
    def __init__(self, db_path="tasks.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                due_date TEXT,
                created_date TEXT NOT NULL,
                completed_date TEXT,
                tags TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def save_task(self, task: Task) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        tags_str = json.dumps(task.tags) if task.tags else "[]"
        
        if task.id == 0:  # New task
            cursor.execute('''
                INSERT INTO tasks (title, description, priority, status, due_date, created_date, completed_date, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (task.title, task.description, task.priority.value, task.status.value,
                  task.due_date, task.created_date, task.completed_date, tags_str))
            task_id = cursor.lastrowid
        else:  # Update existing task
            cursor.execute('''
                UPDATE tasks SET title=?, description=?, priority=?, status=?, due_date=?, completed_date=?, tags=?
                WHERE id=?
            ''', (task.title, task.description, task.priority.value, task.status.value,
                  task.due_date, task.completed_date, tags_str, task.id))
            task_id = task.id
        
        conn.commit()
        conn.close()
        return task_id
    
    def load_tasks(self) -> List[Task]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM tasks')
        rows = cursor.fetchall()
        conn.close()
        
        tasks = []
        for row in rows:
            tags = json.loads(row[8]) if row[8] else []
            task = Task(
                id=row[0],
                title=row[1],
                description=row[2],
                priority=Priority(row[3]),
                status=Status(row[4]),
                due_date=row[5],
                created_date=row[6],
                completed_date=row[7],
                tags=tags
            )
            tasks.append(task)
        return tasks
    
    def delete_task(self, task_id: int):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM tasks WHERE id=?', (task_id,))
        conn.commit()
        conn.close()

class NotificationSystem:
    def __init__(self):
        self.notifications_enabled = True
        self.email_config = {}
    
    def check_due_dates(self, tasks: List[Task]):
        if not self.notifications_enabled:
            return
        
        today = datetime.datetime.now().date()
        overdue_tasks = []
        due_soon_tasks = []
        
        for task in tasks:
            if task.status != Status.COMPLETED and task.due_date:
                try:
                    due_date = datetime.datetime.strptime(task.due_date, "%Y-%m-%d").date()
                    if due_date < today:
                        overdue_tasks.append(task)
                    elif due_date <= today + datetime.timedelta(days=3):
                        due_soon_tasks.append(task)
                except ValueError:
                    continue
        
        if overdue_tasks or due_soon_tasks:
            self.show_notification(overdue_tasks, due_soon_tasks)
    
    def show_notification(self, overdue_tasks: List[Task], due_soon_tasks: List[Task]):
        notification_window = tk.Toplevel()
        notification_window.title("Task Notifications")
        notification_window.geometry("400x300")
        
        frame = ttk.Frame(notification_window, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        if overdue_tasks:
            ttk.Label(frame, text="Overdue Tasks:", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
            for i, task in enumerate(overdue_tasks):
                ttk.Label(frame, text=f"• {task.title} (Due: {task.due_date})", foreground="red").grid(row=i+1, column=0, sticky=tk.W)
        
        if due_soon_tasks:
            start_row = len(overdue_tasks) + 2 if overdue_tasks else 0
            ttk.Label(frame, text="Due Soon:", font=('Arial', 12, 'bold')).grid(row=start_row, column=0, sticky=tk.W, pady=(10, 5))
            for i, task in enumerate(due_soon_tasks):
                ttk.Label(frame, text=f"• {task.title} (Due: {task.due_date})", foreground="orange").grid(row=start_row+i+1, column=0, sticky=tk.W)
        
        ttk.Button(frame, text="Close", command=notification_window.destroy).grid(row=start_row+len(due_soon_tasks)+2, column=0, pady=(20, 0))

class TaskManager:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.notification_system = NotificationSystem()
        self.tasks = self.db_manager.load_tasks()
        self.filtered_tasks = self.tasks.copy()
        
    def add_task(self, task: Task) -> int:
        task_id = self.db_manager.save_task(task)
        task.id = task_id
        self.tasks.append(task)
        self.apply_filters()
        return task_id
    
    def update_task(self, task: Task):
        self.db_manager.save_task(task)
        # Update in memory
        for i, t in enumerate(self.tasks):
            if t.id == task.id:
                self.tasks[i] = task
                break
        self.apply_filters()
    
    def delete_task(self, task_id: int):
        self.db_manager.delete_task(task_id)
        self.tasks = [t for t in self.tasks if t.id != task_id]
        self.apply_filters()
    
    def get_task_by_id(self, task_id: int) -> Optional[Task]:
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def apply_filters(self, status_filter=None, priority_filter=None, search_text=""):
        self.filtered_tasks = self.tasks.copy()
        
        if status_filter and status_filter != "All":
            self.filtered_tasks = [t for t in self.filtered_tasks if t.status.value == status_filter]
        
        if priority_filter and priority_filter != "All":
            self.filtered_tasks = [t for t in self.filtered_tasks if t.priority.value == priority_filter]
        
        if search_text:
            search_lower = search_text.lower()
            self.filtered_tasks = [t for t in self.filtered_tasks 
                                 if search_lower in t.title.lower() or 
                                    search_lower in t.description.lower()]
    
    def export_tasks(self, filename: str):
        tasks_dict = [asdict(task) for task in self.tasks]
        # Convert enum values to strings
        for task_dict in tasks_dict:
            task_dict['priority'] = task_dict['priority'].value if isinstance(task_dict['priority'], Priority) else task_dict['priority']
            task_dict['status'] = task_dict['status'].value if isinstance(task_dict['status'], Status) else task_dict['status']
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(tasks_dict, f, indent=2, ensure_ascii=False)
    
    def import_tasks(self, filename: str):
        with open(filename, 'r', encoding='utf-8') as f:
            tasks_dict = json.load(f)
        
        for task_dict in tasks_dict:
            task = Task(
                id=0,  # Will be assigned new ID
                title=task_dict['title'],
                description=task_dict['description'],
                priority=Priority(task_dict['priority']),
                status=Status(task_dict['status']),
                due_date=task_dict['due_date'],
                created_date=task_dict['created_date'],
                completed_date=task_dict.get('completed_date'),
                tags=task_dict.get('tags', [])
            )
            self.add_task(task)
    
    def get_statistics(self):
        total = len(self.tasks)
        completed = len([t for t in self.tasks if t.status == Status.COMPLETED])
        pending = len([t for t in self.tasks if t.status == Status.PENDING])
        in_progress = len([t for t in self.tasks if t.status == Status.IN_PROGRESS])
        
        overdue = 0
        today = datetime.datetime.now().date()
        for task in self.tasks:
            if task.status != Status.COMPLETED and task.due_date:
                try:
                    due_date = datetime.datetime.strptime(task.due_date, "%Y-%m-%d").date()
                    if due_date < today:
                        overdue += 1
                except ValueError:
                    continue
        
        return {
            'total': total,
            'completed': completed,
            'pending': pending,
            'in_progress': in_progress,
            'overdue': overdue,
            'completion_rate': (completed / total * 100) if total > 0 else 0
        }

class TaskDialog:
    def __init__(self, parent, task=None):
        self.parent = parent
        self.task = task
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Add Task" if task is None else "Edit Task")
        self.dialog.geometry("500x600")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.create_widgets()
        
        if task:
            self.populate_fields()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        ttk.Label(main_frame, text="Title:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.title_var = tk.StringVar()
        title_entry = ttk.Entry(main_frame, textvariable=self.title_var, width=50)
        title_entry.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Description
        ttk.Label(main_frame, text="Description:").grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        self.description_text = tk.Text(main_frame, width=50, height=8)
        self.description_text.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Priority
        ttk.Label(main_frame, text="Priority:").grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        self.priority_var = tk.StringVar()
        priority_combo = ttk.Combobox(main_frame, textvariable=self.priority_var, 
                                    values=[p.value for p in Priority], state="readonly")
        priority_combo.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        priority_combo.set(Priority.MEDIUM.value)
        
        # Status
        ttk.Label(main_frame, text="Status:").grid(row=4, column=1, sticky=tk.W, padx=(20, 0), pady=(0, 5))
        self.status_var = tk.StringVar()
        status_combo = ttk.Combobox(main_frame, textvariable=self.status_var,
                                  values=[s.value for s in Status], state="readonly")
        status_combo.grid(row=5, column=1, sticky=(tk.W, tk.E), padx=(20, 0), pady=(0, 15))
        status_combo.set(Status.PENDING.value)
        
        # Due Date
        ttk.Label(main_frame, text="Due Date (YYYY-MM-DD):").grid(row=6, column=0, sticky=tk.W, pady=(0, 5))
        self.due_date_var = tk.StringVar()
        due_date_entry = ttk.Entry(main_frame, textvariable=self.due_date_var, width=20)
        due_date_entry.grid(row=7, column=0, sticky=tk.W, pady=(0, 15))
        
        # Tags
        ttk.Label(main_frame, text="Tags (comma-separated):").grid(row=8, column=0, sticky=tk.W, pady=(0, 5))
        self.tags_var = tk.StringVar()
        tags_entry = ttk.Entry(main_frame, textvariable=self.tags_var, width=50)
        tags_entry.grid(row=9, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=10, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(button_frame, text="Save", command=self.save_task).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.LEFT)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
    
    def populate_fields(self):
        if self.task:
            self.title_var.set(self.task.title)
            self.description_text.insert('1.0', self.task.description)
            self.priority_var.set(self.task.priority.value)
            self.status_var.set(self.task.status.value)
            self.due_date_var.set(self.task.due_date)
            self.tags_var.set(', '.join(self.task.tags))
    
    def save_task(self):
        title = self.title_var.get().strip()
        if not title:
            messagebox.showerror("Error", "Title is required!")
            return
        
        description = self.description_text.get('1.0', tk.END).strip()
        priority = Priority(self.priority_var.get())
        status = Status(self.status_var.get())
        due_date = self.due_date_var.get().strip()
        tags_text = self.tags_var.get().strip()
        tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()]
        
        # Validate due date
        if due_date:
            try:
                datetime.datetime.strptime(due_date, "%Y-%m-%d")
            except ValueError:
                messagebox.showerror("Error", "Invalid date format! Use YYYY-MM-DD")
                return
        
        completed_date = None
        if status == Status.COMPLETED and (not self.task or self.task.status != Status.COMPLETED):
            completed_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elif self.task and self.task.completed_date:
            completed_date = self.task.completed_date
        
        if self.task:
            # Update existing task
            self.task.title = title
            self.task.description = description
            self.task.priority = priority
            self.task.status = status
            self.task.due_date = due_date
            self.task.completed_date = completed_date
            self.task.tags = tags
            self.result = self.task
        else:
            # Create new task
            self.result = Task(
                id=0,
                title=title,
                description=description,
                priority=priority,
                status=status,
                due_date=due_date,
                created_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                completed_date=completed_date,
                tags=tags
            )
        
        self.dialog.destroy()

class TaskManagerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Advanced Task Manager")
        self.root.geometry("1200x800")
        
        self.task_manager = TaskManager()
        
        self.create_widgets()
        self.refresh_task_list()
        
        # Start notification checker
        self.check_notifications()
    
    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Configure main grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_frame, padding="10")
        left_panel.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons
        ttk.Button(left_panel, text="Add Task", command=self.add_task).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(left_panel, text="Edit Task", command=self.edit_task).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(left_panel, text="Delete Task", command=self.delete_task).pack(fill=tk.X, pady=(0, 15))
        
        # Filters
        ttk.Label(left_panel, text="Filter by Status:").pack(anchor=tk.W)
        self.status_filter_var = tk.StringVar(value="All")
        status_filter = ttk.Combobox(left_panel, textvariable=self.status_filter_var,
                                   values=["All"] + [s.value for s in Status], state="readonly")
        status_filter.pack(fill=tk.X, pady=(0, 10))
        status_filter.bind('<<ComboboxSelected>>', self.apply_filters)
        
        ttk.Label(left_panel, text="Filter by Priority:").pack(anchor=tk.W)
        self.priority_filter_var = tk.StringVar(value="All")
        priority_filter = ttk.Combobox(left_panel, textvariable=self.priority_filter_var,
                                     values=["All"] + [p.value for p in Priority], state="readonly")
        priority_filter.pack(fill=tk.X, pady=(0, 10))
        priority_filter.bind('<<ComboboxSelected>>', self.apply_filters)
        
        ttk.Label(left_panel, text="Search:").pack(anchor=tk.W)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(left_panel, textvariable=self.search_var)
        search_entry.pack(fill=tk.X, pady=(0, 15))
        search_entry.bind('<KeyRelease>', self.apply_filters)
        
        # Statistics
        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=(0, 10))
        ttk.Label(left_panel, text="Statistics:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.stats_label = ttk.Label(left_panel, text="", justify=tk.LEFT)
        self.stats_label.pack(anchor=tk.W, pady=(5, 15))
        
        # Import/Export
        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, pady=(0, 10))
        ttk.Button(left_panel, text="Export Tasks", command=self.export_tasks).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(left_panel, text="Import Tasks", command=self.import_tasks).pack(fill=tk.X, pady=(0, 5))
        
        # Top panel - Title and refresh
        top_panel = ttk.Frame(main_frame)
        top_panel.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 10))
        top_panel.columnconfigure(0, weight=1)
        
        ttk.Label(top_panel, text="Task List", font=('Arial', 16, 'bold')).grid(row=0, column=0, sticky=tk.W)
        ttk.Button(top_panel, text="Refresh", command=self.refresh_task_list).grid(row=0, column=1, sticky=tk.E)
        
        # Task list
        list_frame = ttk.Frame(main_frame)
        list_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Treeview with scrollbars
        columns = ('ID', 'Title', 'Priority', 'Status', 'Due Date', 'Tags')
        self.task_tree = ttk.Treeview(list_frame, columns=columns, show='headings', selectmode='extended')
        
        # Configure columns
        self.task_tree.heading('ID', text='ID')
        self.task_tree.heading('Title', text='Title')
        self.task_tree.heading('Priority', text='Priority')
        self.task_tree.heading('Status', text='Status')
        self.task_tree.heading('Due Date', text='Due Date')
        self.task_tree.heading('Tags', text='Tags')
        
        self.task_tree.column('ID', width=50)
        self.task_tree.column('Title', width=200)
        self.task_tree.column('Priority', width=100)
        self.task_tree.column('Status', width=100)
        self.task_tree.column('Due Date', width=100)
        self.task_tree.column('Tags', width=150)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.task_tree.yview)
        h_scrollbar = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL, command=self.task_tree.xview)
        self.task_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.task_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Double-click to edit
        self.task_tree.bind('<Double-1>', lambda e: self.edit_task())
        
        # Context menu
        self.create_context_menu()
    
    def create_context_menu(self):
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Edit", command=self.edit_task)
        self.context_menu.add_command(label="Delete", command=self.delete_task)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Mark as Completed", command=self.mark_completed)
        
        self.task_tree.bind("<Button-3>", self.show_context_menu)
    
    def show_context_menu(self, event):
        if self.task_tree.selection():
            self.context_menu.post(event.x_root, event.y_root)
    
    def add_task(self):
        dialog = TaskDialog(self.root)
        self.root.wait_window(dialog.dialog)
        
        if dialog.result:
            self.task_manager.add_task(dialog.result)
            self.refresh_task_list()
    
    def edit_task(self):
        selection = self.task_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a task to edit.")
            return
        
        task_id = int(self.task_tree.item(selection[0])['values'][0])
        task = self.task_manager.get_task_by_id(task_id)
        
        if task:
            dialog = TaskDialog(self.root, task)
            self.root.wait_window(dialog.dialog)
            
            if dialog.result:
                self.task_manager.update_task(dialog.result)
                self.refresh_task_list()
    
    def delete_task(self):
        selection = self.task_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select task(s) to delete.")
            return
        
        if messagebox.askyesno("Confirm", f"Delete {len(selection)} task(s)?"):
            for item in selection:
                task_id = int(self.task_tree.item(item)['values'][0])
                self.task_manager.delete_task(task_id)
            self.refresh_task_list()
    
    def mark_completed(self):
        selection = self.task_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a task to mark as completed.")
            return
        
        for item in selection:
            task_id = int(self.task_tree.item(item)['values'][0])
            task = self.task_manager.get_task_by_id(task_id)
            if task:
                task.status = Status.COMPLETED
                task.completed_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.task_manager.update_task(task)
        
        self.refresh_task_list()
    
    def apply_filters(self, event=None):
        status_filter = self.status_filter_var.get()
        priority_filter = self.priority_filter_var.get()
        search_text = self.search_var.get()
        
        self.task_manager.apply_filters(status_filter, priority_filter, search_text)
        self.refresh_task_list()
    
    def refresh_task_list(self):
        # Clear existing items
        for item in self.task_tree.get_children():
            self.task_tree.delete(item)
        
        # Add filtered tasks
        for task in self.task_manager.filtered_tasks:
            tags_str = ', '.join(task.tags)
            self.task_tree.insert('', 'end', values=(
                task.id,
                task.title,
                task.priority.value,
                task.status.value,
                task.due_date,
                tags_str
            ))
        
        # Update statistics
        self.update_statistics()
    
    def update_statistics(self):
        stats = self.task_manager.get_statistics()
        stats_text = f"Total: {stats['total']}\n"
        stats_text += f"Completed: {stats['completed']}\n"
        stats_text += f"Pending: {stats['pending']}\n"
        stats_text += f"In Progress: {stats['in_progress']}\n"
        stats_text += f"Overdue: {stats['overdue']}\n"
        stats_text += f"Completion Rate: {stats['completion_rate']:.1f}%"
        
        self.stats_label.config(text=stats_text)
    
    def export_tasks(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.task_manager.export_tasks(filename)
                messagebox.showinfo("Success", f"Tasks exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export tasks: {str(e)}")
    
    def import_tasks(self):
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.task_manager.import_tasks(filename)
                self.refresh_task_list()
                messagebox.showinfo("Success", f"Tasks imported from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import tasks: {str(e)}")
    
    def check_notifications(self):
        def notification_worker():
            while True:
                self.task_manager.notification_system.check_due_dates(self.task_manager.tasks)
                time.sleep(3600)  # Check every hour
        
        notification_thread = threading.Thread(target=notification_worker, daemon=True)
        notification_thread.start()
    
    def run(self):
        self.root.mainloop()

class AdvancedTaskAnalyzer:
    def __init__(self, tasks: List[Task]):
        self.tasks = tasks
    
    def get_productivity_trends(self):
        """Analyze productivity trends over time"""
        completed_tasks = [t for t in self.tasks if t.status == Status.COMPLETED and t.completed_date]
        
        daily_completion = {}
        for task in completed_tasks:
            try:
                date = datetime.datetime.strptime(task.completed_date, "%Y-%m-%d %H:%M:%S").date()
                daily_completion[date] = daily_completion.get(date, 0) + 1
            except ValueError:
                continue
        
        return daily_completion
    
    def get_priority_distribution(self):
        """Get distribution of tasks by priority"""
        priority_count = {p.value: 0 for p in Priority}
        for task in self.tasks:
            priority_count[task.priority.value] += 1
        return priority_count
    
    def get_overdue_analysis(self):
        """Analyze overdue tasks patterns"""
        today = datetime.datetime.now().date()
        overdue_tasks = []
        
        for task in self.tasks:
            if task.status != Status.COMPLETED and task.due_date:
                try:
                    due_date = datetime.datetime.strptime(task.due_date, "%Y-%m-%d").date()
                    if due_date < today:
                        days_overdue = (today - due_date).days
                        overdue_tasks.append({
                            'task': task,
                            'days_overdue': days_overdue
                        })
                except ValueError:
                    continue
        
        return sorted(overdue_tasks, key=lambda x: x['days_overdue'], reverse=True)
    
    def get_tag_analysis(self):
        """Analyze tag usage and task completion by tags"""
        tag_stats = {}
        
        for task in self.tasks:
            for tag in task.tags:
                if tag not in tag_stats:
                    tag_stats[tag] = {'total': 0, 'completed': 0}
                
                tag_stats[tag]['total'] += 1
                if task.status == Status.COMPLETED:
                    tag_stats[tag]['completed'] += 1
        
        # Calculate completion rates
        for tag, stats in tag_stats.items():
            stats['completion_rate'] = (stats['completed'] / stats['total']) * 100 if stats['total'] > 0 else 0
        
        return tag_stats
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        report = []
        report.append("=== TASK MANAGEMENT ANALYSIS REPORT ===\n")
        
        # Basic statistics
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks if t.status == Status.COMPLETED])
        pending_tasks = len([t for t in self.tasks if t.status == Status.PENDING])
        in_progress_tasks = len([t for t in self.tasks if t.status == Status.IN_PROGRESS])
        
        report.append("OVERVIEW:")
        report.append(f"Total Tasks: {total_tasks}")
        report.append(f"Completed: {completed_tasks} ({(completed_tasks/total_tasks)*100:.1f}%)")
        report.append(f"Pending: {pending_tasks} ({(pending_tasks/total_tasks)*100:.1f}%)")
        report.append(f"In Progress: {in_progress_tasks} ({(in_progress_tasks/total_tasks)*100:.1f}%)")
        report.append("")
        
        # Priority distribution
        priority_dist = self.get_priority_distribution()
        report.append("PRIORITY DISTRIBUTION:")
        for priority, count in priority_dist.items():
            percentage = (count/total_tasks)*100 if total_tasks > 0 else 0
            report.append(f"{priority}: {count} tasks ({percentage:.1f}%)")
        report.append("")
        
        # Overdue analysis
        overdue_analysis = self.get_overdue_analysis()
        report.append("OVERDUE TASKS:")
        if overdue_analysis:
            report.append(f"Total overdue: {len(overdue_analysis)}")
            report.append("Most overdue tasks:")
            for item in overdue_analysis[:5]:  # Top 5 most overdue
                task = item['task']
                days = item['days_overdue']
                report.append(f"  • {task.title} - {days} days overdue ({task.priority.value})")
        else:
            report.append("No overdue tasks!")
        report.append("")
        
        # Tag analysis
        tag_stats = self.get_tag_analysis()
        if tag_stats:
            report.append("TAG ANALYSIS:")
            sorted_tags = sorted(tag_stats.items(), key=lambda x: x[1]['completion_rate'], reverse=True)
            for tag, stats in sorted_tags[:10]:  # Top 10 tags
                report.append(f"{tag}: {stats['completed']}/{stats['total']} completed ({stats['completion_rate']:.1f}%)")
            report.append("")
        
        # Productivity trends
        trends = self.get_productivity_trends()
        if trends:
            report.append("RECENT PRODUCTIVITY:")
            recent_dates = sorted(trends.keys(), reverse=True)[:7]  # Last 7 days with completions
            for date in recent_dates:
                report.append(f"{date}: {trends[date]} tasks completed")
        
        return "\n".join(report)

class ReportWindow:
    def __init__(self, parent, task_manager):
        self.parent = parent
        self.task_manager = task_manager
        
        self.window = tk.Toplevel(parent)
        self.window.title("Task Analysis Report")
        self.window.geometry("800x600")
        self.window.transient(parent)
        
        self.create_widgets()
        self.generate_report()
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title and refresh button
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        title_frame.columnconfigure(0, weight=1)
        
        ttk.Label(title_frame, text="Task Analysis Report", font=('Arial', 16, 'bold')).grid(row=0, column=0, sticky=tk.W)
        ttk.Button(title_frame, text="Refresh", command=self.generate_report).grid(row=0, column=1, sticky=tk.E)
        ttk.Button(title_frame, text="Export Report", command=self.export_report).grid(row=0, column=2, sticky=tk.E, padx=(5, 0))
        
        # Report text area
        text_frame = ttk.Frame(main_frame)
        text_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.report_text = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=scrollbar.set)
        
        self.report_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
    
    def generate_report(self):
        analyzer = AdvancedTaskAnalyzer(self.task_manager.tasks)
        report = analyzer.generate_report()
        
        self.report_text.delete('1.0', tk.END)
        self.report_text.insert('1.0', report)
    
    def export_report(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.report_text.get('1.0', tk.END))
                messagebox.showinfo("Success", f"Report exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export report: {str(e)}")

class TaskManagerApp(TaskManagerGUI):
    def __init__(self):
        super().__init__()
        self.add_advanced_features()
    
    def add_advanced_features(self):
        # Add menu bar
        self.create_menu_bar()
        
        # Add status bar
        self.create_status_bar()
        
        # Configure window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_menu_bar(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Task", command=self.add_task, accelerator="Ctrl+N")
        file_menu.add_separator()
        file_menu.add_command(label="Import Tasks", command=self.import_tasks, accelerator="Ctrl+I")
        file_menu.add_command(label="Export Tasks", command=self.export_tasks, accelerator="Ctrl+E")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing, accelerator="Ctrl+Q")
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Edit Task", command=self.edit_task, accelerator="Ctrl+E")
        edit_menu.add_command(label="Delete Task", command=self.delete_task, accelerator="Delete")
        edit_menu.add_separator()
        edit_menu.add_command(label="Mark Completed", command=self.mark_completed, accelerator="Ctrl+M")
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Refresh", command=self.refresh_task_list, accelerator="F5")
        view_menu.add_command(label="Analysis Report", command=self.show_analysis_report, accelerator="Ctrl+R")
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Backup Database", command=self.backup_database)
        tools_menu.add_command(label="Settings", command=self.show_settings)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-n>', lambda e: self.add_task())
        self.root.bind('<Control-i>', lambda e: self.import_tasks())
        self.root.bind('<Control-e>', lambda e: self.export_tasks())
        self.root.bind('<Control-q>', lambda e: self.on_closing())
        self.root.bind('<Control-m>', lambda e: self.mark_completed())
        self.root.bind('<Control-r>', lambda e: self.show_analysis_report())
        self.root.bind('<F5>', lambda e: self.refresh_task_list())
        self.root.bind('<Delete>', lambda e: self.delete_task())
    
    def create_status_bar(self):
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=(0, 5))
        
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side=tk.LEFT)
        
        # Clock
        self.clock_label = ttk.Label(self.status_bar, text="")
        self.clock_label.pack(side=tk.RIGHT)
        self.update_clock()
    
    def update_clock(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.clock_label.config(text=current_time)
        self.root.after(1000, self.update_clock)
    
    def show_analysis_report(self):
        ReportWindow(self.root, self.task_manager)
    
    def backup_database(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"tasks_backup_{timestamp}.db"
        
        try:
            import shutil
            shutil.copy2(self.task_manager.db_manager.db_path, backup_filename)
            messagebox.showinfo("Success", f"Database backed up to {backup_filename}")
            self.status_label.config(text=f"Database backed up to {backup_filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to backup database: {str(e)}")
    
    def show_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        frame = ttk.Frame(settings_window, padding="20")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Notification settings
        ttk.Label(frame, text="Notification Settings", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        notifications_var = tk.BooleanVar(value=self.task_manager.notification_system.notifications_enabled)
        ttk.Checkbutton(frame, text="Enable notifications", variable=notifications_var).grid(row=1, column=0, sticky=tk.W)
        
        # Database settings
        ttk.Label(frame, text="Database Settings", font=('Arial', 12, 'bold')).grid(row=2, column=0, sticky=tk.W, pady=(20, 10))
        ttk.Label(frame, text=f"Database location: {self.task_manager.db_manager.db_path}").grid(row=3, column=0, sticky=tk.W)
        
        # Save button
        def save_settings():
            self.task_manager.notification_system.notifications_enabled = notifications_var.get()
            messagebox.showinfo("Settings", "Settings saved successfully!")
            settings_window.destroy()
        
        ttk.Button(frame, text="Save", command=save_settings).grid(row=4, column=0, pady=(20, 0))
    
    def show_about(self):
        about_text = """Advanced Task Manager v1.0

A comprehensive task management application built with Python and Tkinter.

Features:
• Create, edit, and delete tasks
• Priority and status management
• Due date tracking with notifications
• Tag-based organization
• Advanced filtering and search
• Import/export functionality
• Detailed analytics and reporting
• Database backup and restore

Created with Python, Tkinter, and SQLite.
"""
        messagebox.showinfo("About Task Manager", about_text)
    
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.root.destroy()
    
    def refresh_task_list(self):
        super().refresh_task_list()
        task_count = len(self.task_manager.filtered_tasks)
        total_count = len(self.task_manager.tasks)
        if task_count == total_count:
            self.status_label.config(text=f"Showing {task_count} tasks")
        else:
            self.status_label.config(text=f"Showing {task_count} of {total_count} tasks")

def main():
    """Main function to run the Task Manager application"""
    try:
        app = TaskManagerApp()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
