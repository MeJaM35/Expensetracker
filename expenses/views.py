from django.shortcuts import render, redirect,HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from .models import Category, Expense
from django.contrib import messages
from django.contrib.auth.models import User
from django.core.paginator import Paginator
import json
from django.http import JsonResponse
from userpreferences.models import UserPreference
import datetime
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from django.contrib.sessions.models import Session
from datetime import date
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages
import datetime
from .models import ExpenseLimit
from django.core.mail import send_mail
from django.conf import settings
from django.db.models import Sum, Avg
from django.utils import timezone
from datetime import timedelta
import json


data = pd.read_csv('dataset.csv')

# Preprocessing
stop_words = set(stopwords.words('english'))

def home(request):
    return render(request, 'expenses/landing.html')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    return ' '.join(tokens)

data['clean_description'] = data['description'].apply(preprocess_text)

# Feature extraction
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['clean_description'])

# Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, data['category'])
@login_required(login_url='/authentication/login')
def search_expenses(request):
    if request.method == 'POST':
        search_str = json.loads(request.body).get('searchText')
        expenses = Expense.objects.filter(
            amount__istartswith=search_str, owner=request.user) | Expense.objects.filter(
            date__istartswith=search_str, owner=request.user) | Expense.objects.filter(
            description__icontains=search_str, owner=request.user) | Expense.objects.filter(
            category__icontains=search_str, owner=request.user)
        data = expenses.values()
        return JsonResponse(list(data), safe=False)


@login_required(login_url='/authentication/login')
def index(request):
    categories = Category.objects.all()
    expenses = Expense.objects.filter(owner=request.user)

    sort_order = request.GET.get('sort')

    if sort_order == 'amount_asc':
        expenses = expenses.order_by('amount')
    elif sort_order == 'amount_desc':
        expenses = expenses.order_by('-amount')
    elif sort_order == 'date_asc':
        expenses = expenses.order_by('date')
    elif sort_order == 'date_desc':
        expenses = expenses.order_by('-date')

    paginator = Paginator(expenses, 5)
    page_number = request.GET.get('page')
    page_obj = Paginator.get_page(paginator, page_number)
    try:
        currency = UserPreference.objects.get(user=request.user).currency
    except:
        currency=None

    total = page_obj.paginator.num_pages
    context = {
        'expenses': expenses,
        'page_obj': page_obj,
        'currency': currency,
        'total': total,
        'sort_order': sort_order,

    }
    return render(request, 'expenses/index.html', context)

daily_expense_amounts = {}

@login_required(login_url='/authentication/login')
def add_expense(request):
    categories = Category.objects.all()
    context = {
        'categories': categories,
        'values': request.POST
    }
    if request.method == 'GET':
        return render(request, 'expenses/add_expense.html', context)

    if request.method == 'POST':
        amount = request.POST['amount']
        date_str = request.POST.get('expense_date')
        
        if not amount:
            messages.error(request, 'Amount is required')
            return render(request, 'expenses/add_expense.html', context)
        description = request.POST['description']
        date = request.POST['expense_date']
        predicted_category = request.POST['category']

        if not description:
            messages.error(request, 'description is required')
            return render(request, 'expenses/add_expense.html', context)
        
        initial_predicted_category = request.POST.get('initial_predicted_category')
        if predicted_category != initial_predicted_category:
            new_data = {
            'description': description,
            'category': predicted_category,
        }

        update_url = 'http://127.0.0.1:8000/api/update-dataset/'
        response = requests.post(update_url, json={'new_data': new_data})

        try:
            date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            today = datetime.date.today()

            if date > today:
                messages.error(request, 'Date cannot be in the future')
                return render(request, 'expenses/add_expense.html', context)
            
            user = request.user
            expense_limits = ExpenseLimit.objects.filter(owner=user)
            if expense_limits.exists():
                daily_expense_limit = expense_limits.first().daily_expense_limit
            else:
                daily_expense_limit = 5000  

            
            total_expenses_today = get_expense_of_day(user) + float(amount)
            if total_expenses_today > daily_expense_limit:
                subject = 'Daily Expense Limit Exceeded'
                message = f'Hello {user.username},\n\nYour expenses for today have exceeded your daily expense limit. Please review your expenses.'
                from_email = settings.EMAIL_HOST_USER
                to_email = [user.email]
                send_mail(subject, message, from_email, to_email, fail_silently=False)
                messages.warning(request, 'Your expenses for today exceed your daily expense limit')

            Expense.objects.create(owner=request.user, amount=amount, date=date,
                                   category=predicted_category, description=description)
            messages.success(request, 'Expense saved successfully')
            return redirect('expenses')
        except ValueError:
            messages.error(request, 'Invalid date format')
            return render(request, 'expenses/add_expense.html', context)


@login_required(login_url='/authentication/login')
def expense_edit(request, id):
    expense = Expense.objects.get(pk=id)
    categories = Category.objects.all()
    context = {
        'expense': expense,
        'values': expense,
        'categories': categories
    }
    if request.method == 'GET':
        return render(request, 'expenses/edit-expense.html', context)
    if request.method == 'POST':
        amount = request.POST['amount']
        date_str = request.POST.get('expense_date')

        if not amount:
            messages.error(request, 'Amount is required')
            return render(request, 'expenses/edit-expense.html', context)
        description = request.POST['description']
        date = request.POST['expense_date']
        category = request.POST['category']

        if not description:
            messages.error(request, 'description is required')
            return render(request, 'expenses/edit-expense.html', context)

        try:
            # Convert the date string to a datetime object and validate the date
            date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            today = datetime.date.today()

            if date > today:
                messages.error(request, 'Date cannot be in the future')
                return render(request, 'expenses/add_expense.html', context)

            expense.owner = request.user
            expense.amount = amount
            expense. date = date
            expense.category = category
            expense.description = description

            expense.save()
            messages.success(request, 'Expense saved successfully')

            return redirect('expenses')
        except ValueError:
            messages.error(request, 'Invalid date format')
            return render(request, 'expenses/edit_income.html', context)

        # expense.owner = request.user
        # expense.amount = amount
        # expense. date = date
        # expense.category = category
        # expense.description = description

        # expense.save()

        # messages.success(request, 'Expense updated  successfully')

        # return redirect('expenses')

@login_required(login_url='/authentication/login')
def delete_expense(request, id):
    expense = Expense.objects.get(pk=id)
    expense.delete()
    messages.success(request, 'Expense removed')
    return redirect('expenses')

@login_required(login_url='/authentication/login')
def expense_category_summary(request):
    todays_date = datetime.date.today()
    six_months_ago = todays_date-datetime.timedelta(days=30*6)
    expenses = Expense.objects.filter(owner=request.user,
                                      date__gte=six_months_ago, date__lte=todays_date)
    finalrep = {}

    def get_category(expense):
        return expense.category
    category_list = list(set(map(get_category, expenses)))

    def get_expense_category_amount(category):
        amount = 0
        filtered_by_category = expenses.filter(category=category)

        for item in filtered_by_category:
            amount += item.amount
        return amount

    for x in expenses:
        for y in category_list:
            finalrep[y] = get_expense_category_amount(y)

    return JsonResponse({'expense_category_data': finalrep}, safe=False)

@login_required(login_url='/authentication/login')
def stats_view(request):
    # Get date range from request, default to 30 days
    date_range = request.GET.get('date_range', '30')
    
    # Calculate date range
    end_date = timezone.now().date()
    start_date = end_date - timedelta(days=int(date_range))
    
    # Get expenses within date range
    expenses = Expense.objects.filter(
        owner=request.user,
        date__gte=start_date,
        date__lte=end_date
    )
    
    # Calculate total expenses with fallback to 0
    total_expenses = expenses.aggregate(
        total=Sum('amount')
    )['total'] or 0.00
    
    # Calculate monthly average (handle case when there are no expenses)
    if total_expenses > 0:
        monthly_average = total_expenses / (int(date_range) / 30)
    else:
        monthly_average = 0.00
    
    # Get expense limit for context
    expense_limit = ExpenseLimit.objects.filter(owner=request.user).first()
    daily_limit = expense_limit.daily_expense_limit if expense_limit else 5000
    
    # Get today's expenses
    today_expenses = get_expense_of_day(request.user)
    
    # Get top categories with category names
    top_categories = expenses.values('category')\
        .annotate(total=Sum('amount'))\
        .order_by('-total')[:5]
    
    # Prepare chart data
    categories = []
    amounts = []
    chart_colors = [
        '#FF6384', '#36A2EB', '#FFCE56',
        '#4BC0C0', '#9966FF', '#FF9F40'
    ]
    
    for category in top_categories:
        categories.append(category['category'])
        amounts.append(float(category['total']))
    
    chart_data = {
        'labels': categories,
        'datasets': [{
            'label': 'Expenses by Category',
            'data': amounts,
            'backgroundColor': chart_colors[:len(categories)]
        }]
    }
    
    # Get currency preference
    try:
        currency = UserPreference.objects.get(user=request.user).currency
    except UserPreference.DoesNotExist:
        currency = '₹'  # Default to INR symbol
    
    context = {
        'total_expenses': total_expenses,
        'monthly_average': round(monthly_average, 2),
        'daily_limit': daily_limit,
        'today_expenses': today_expenses,
        'top_categories': [
            {
                'name': cat['category'],
                'total': cat['total'],
                'percentage': (cat['total'] / total_expenses * 100) if total_expenses > 0 else 0
            } 
            for cat in top_categories
        ],
        'date_range': date_range,
        'chart_data': json.dumps(chart_data),
        'currency': currency,
        'limit_percentage': (today_expenses / daily_limit * 100) if daily_limit > 0 else 0
    }
    
    return render(request, 'expenses/stats.html', context)

@login_required(login_url='/authentication/login')
def predict_category(description):
    predict_category_url = 'http://localhost:8000/api/predict-category/'  # Use the correct URL path
    data = {'description': description}
    response = requests.post(predict_category_url, data=data)

    if response.status_code == 200:
        # Get the predicted category from the response
        predicted_category = response.json().get('predicted_category')
        return predicted_category
    else:
        # Handle the case where the prediction request failed
        return None
    

def set_expense_limit(request):
    if request.method == "POST":
        daily_expense_limit = request.POST.get('daily_expense_limit')
        
        existing_limit = ExpenseLimit.objects.filter(owner=request.user).first()
        
        if existing_limit:
            existing_limit.daily_expense_limit = daily_expense_limit
            existing_limit.save()
        else:
            ExpenseLimit.objects.create(owner=request.user, daily_expense_limit=daily_expense_limit)
        
        messages.success(request, "Daily Expense Limit Updated Successfully!")
        return HttpResponseRedirect('/preferences/')
    else:
        return HttpResponseRedirect('/preferences/')
    
def get_expense_of_day(user):
    current_date=date.today()
    expenses=Expense.objects.filter(owner=user,date=current_date)
    total_expenses=sum(expense.amount for expense in expenses)
    return total_expenses


