from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request, "best/home.html")

def update(request):
    return render(request, "best/update.html")
def application(request):
    if request.method=='POST':
        date = datetime.datetime.now()
        date1=datetime.datetime.strftime(date, "%Y-%m-%d")
        fname         = request.POST.get("fname")
        lname         = request.POST.get("lname")
        dob           = request.POST.get('dob')
        board         = request.POST.get('board')
        father        = request.POST.get("father")
        mother        = request.POST.get("mother")
        qualification = request.POST.get('qualification')
        sname         = request.POST.get('sname')
        saddress      = request.POST.get('Saddress')
        haddress      = request.POST.get('Haddress')
        state         = request.POST.get('state')
        anum          = request.POST.get('anum', '')
        phonenum      = request.POST.get('num')
        email         = request.POST.get('email')
        number        = '19'+'{:06d}'.format(random.randrange(1, 999999))
        username      = (state + board + qualification+ number)
        status        = request.POST.get('hidden1')
        referral      = request.POST.get('referral', '')
        if ApplicationFormClass.objects.filter(emailID=email).exists():
            messages.error(request,"Email ID Already Registered.Use Another One.")
        # elif status=='offline':
        #     af = ApplicationFormClass(firstName = fname, lastName = lname, date_of_birth = dob, board = board, fatherName = father, motherName = mother, qualification = qualification, schoolName = sname, schoolAddress = saddress, homeAddress = haddress, aadharNumber = anum, phoneNumber = phonenum, emailID = email, state = state, username = username, status=status, referral=referral)
        #     af.save()
        #     sub="Email From Best Scholarship"
        #     msg = "UserName:"+username+"."+"\n"+"password:"+str(dob)
        #     send_mail(sub,msg,'dontreply@ibest.co.in',[email])
        #     return HttpResponse('mail sent successfully')
        else:
            af = ApplicationFormClass(date=date1, firstName = fname, lastName = lname, date_of_birth = dob, board = board, fatherName = father, motherName = mother, qualification = qualification, schoolName = sname, schoolAddress = saddress, homeAddress = haddress, aadharNumber = anum, phoneNumber = phonenum, emailID = email, state = state, username = username, status=status, referral=referral)
            af.save()
            return redirect('/pay')
    return render(request, 'best/Register.html')


def admin_login(reuest):
    return render(reuest, 'best/admin_login.html')
def about(request):
    return render(request, 'best/about.html') 