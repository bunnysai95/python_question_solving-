from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request, "best/home.html")

def update(request):
    return render(request, "best/update.html")