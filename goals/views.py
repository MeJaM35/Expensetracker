from django.shortcuts import render, redirect, get_object_or_404
from .models import Goal
from userincome.models import UserIncome
from expenses.models import Expense
from .forms import GoalForm, AddAmountForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.mail import send_mail
from django.db.models import Sum, Avg, Count, Q, F, ExpressionWrapper, FloatField
from django.db.models.functions import TruncMonth
from django.utils import timezone
import json
from decimal import Decimal
from huggingface_hub import InferenceClient
from django.conf import settings
import logging
from django.core.cache import cache
from datetime import timedelta
import numpy as np
from collections import defaultdict

# Update to use Together AI provider through huggingface_hub
TOGETHER_API_KEY = ""  # Replace with your actual API key

def generate_ai_recommendations(user):
    # Check if recommendations are cached for this user
    cache_key = f'ai_recommendations_{user.id}'
    cached_recommendations = cache.get(cache_key)
    
    if cached_recommendations:
        return cached_recommendations
    
    try:
        # Establish time periods for current and historical analysis
        current_date = timezone.now().date()
        current_month = current_date.month
        current_year = current_date.year
        
        # Define historical period - last 6 months
        six_months_ago = current_date - timedelta(days=180)
        three_months_ago = current_date - timedelta(days=90)
        
        # --- CURRENT FINANCIAL SITUATION ---
        
        # Get monthly income and expenses in a single query each
        monthly_income = float(UserIncome.objects.filter(
            owner=user, 
            date__month=current_month,
            date__year=current_year
        ).aggregate(Sum('amount'))['amount__sum'] or 0.0)
        
        monthly_expenses = float(Expense.objects.filter(
            owner=user, 
            date__month=current_month,
            date__year=current_year
        ).aggregate(Sum('amount'))['amount__sum'] or 0.0)
        
        net_cash_flow = monthly_income - monthly_expenses
        
        # Get current top expense categories
        current_top_expenses = Expense.objects.filter(
            owner=user,
            date__month=current_month,
            date__year=current_year
        ).values('category').annotate(
            total=Sum('amount')
        ).order_by('-total')[:5]
        
        top_expenses = [{
            "category": e['category'], 
            "amount": float(e['total'])
        } for e in current_top_expenses]
        
        # --- HISTORICAL DATA ANALYSIS ---
        
        # 1. Calculate monthly spending trends over time
        monthly_expenses_trend = Expense.objects.filter(
            owner=user,
            date__gte=six_months_ago
        ).annotate(
            month=TruncMonth('date')
        ).values('month').annotate(
            total=Sum('amount')
        ).order_by('month')
        
        # Calculate month-over-month changes
        expense_trend = []
        prev_amount = None
        
        for month_data in monthly_expenses_trend:
            current_amount = float(month_data['total'])
            month_str = month_data['month'].strftime('%b %Y')
            
            if prev_amount is not None:
                change_pct = ((current_amount - prev_amount) / prev_amount * 100) if prev_amount > 0 else 0
                trend_direction = "increase" if change_pct > 0 else "decrease"
                expense_trend.append({
                    "month": month_str,
                    "amount": current_amount,
                    "change_percent": round(change_pct, 1),
                    "direction": trend_direction
                })
            else:
                expense_trend.append({
                    "month": month_str,
                    "amount": current_amount
                })
            
            prev_amount = current_amount
        
        # 2. Analyze category-specific spending patterns
        category_trends = defaultdict(list)
        
        for category in [e['category'] for e in current_top_expenses]:
            cat_expenses = Expense.objects.filter(
                owner=user,
                category=category,
                date__gte=six_months_ago
            ).annotate(
                month=TruncMonth('date')
            ).values('month').annotate(
                total=Sum('amount')
            ).order_by('month')
            
            for item in cat_expenses:
                category_trends[category].append({
                    "month": item['month'].strftime('%b %Y'),
                    "amount": float(item['total'])
                })
        
        # 3. Goal-specific analysis - track contribution patterns and correlate with spending
        
        # Get all user goals
        goals = Goal.objects.filter(owner=user)
        goals_analysis = []
        
        # Process goals in chunks for efficiency
        chunk_size = 10
        for i in range(0, goals.count(), chunk_size):
            chunk = goals[i:i+chunk_size]
            
            for goal in chunk:
                progress = goal.calculate_progress()
                
                # Calculate additional goal metrics
                days_to_deadline = (goal.end_date - current_date).days if goal.end_date else None
                monthly_contribution_needed = (goal.amount_to_save - goal.current_saved_amount) / (days_to_deadline / 30) if days_to_deadline and days_to_deadline > 0 else None
                feasibility_score = None
                
                if monthly_contribution_needed and net_cash_flow > 0:
                    feasibility_score = min(10, round((net_cash_flow / monthly_contribution_needed) * 10))
                
                # Look for expense categories that might impact this goal
                # Simple keyword matching between goal name and expense categories
                goal_keywords = goal.name.lower().split()
                related_categories = []
                
                if goal_keywords:
                    for expense_cat in top_expenses:
                        category_name = expense_cat['category'].lower()
                        if any(keyword in category_name for keyword in goal_keywords):
                            related_categories.append(expense_cat['category'])
                
                # Add goal analysis with enhanced metrics
                goals_analysis.append({
                    "name": goal.name,
                    "progress": float(progress['saved_percentage']),
                    "daily_required": float(progress['daily_savings_required']),
                    "status": "behind" if progress['daily_savings_required'] > Decimal('0') else "on_track",
                    "days_to_deadline": days_to_deadline,
                    "monthly_needed": float(monthly_contribution_needed) if monthly_contribution_needed else None,
                    "feasibility_score": feasibility_score,
                    "related_categories": related_categories
                })
        
        # 4. Identify discretionary vs. essential spending
        # Simple categorization based on common expense categories
        essential_categories = ['Rent', 'Mortgage', 'Groceries', 'Utilities', 'Healthcare', 'Insurance', 'Transportation']
        
        essential_spending = float(Expense.objects.filter(
            owner=user,
            date__gte=three_months_ago,
            category__in=essential_categories
        ).aggregate(Sum('amount'))['amount__sum'] or 0.0)
        
        discretionary_spending = float(Expense.objects.filter(
            owner=user,
            date__gte=three_months_ago
        ).exclude(
            category__in=essential_categories
        ).aggregate(Sum('amount'))['amount__sum'] or 0.0)
        
        # 5. Savings rate calculation
        total_income_3m = float(UserIncome.objects.filter(
            owner=user, 
            date__gte=three_months_ago
        ).aggregate(Sum('amount'))['amount__sum'] or 0.0)
        
        total_expenses_3m = float(Expense.objects.filter(
            owner=user, 
            date__gte=three_months_ago
        ).aggregate(Sum('amount'))['amount__sum'] or 0.0)
        
        savings_rate = ((total_income_3m - total_expenses_3m) / total_income_3m * 100) if total_income_3m > 0 else 0
        
        # Create a comprehensive context for the model
        financial_summary = {
            # Current situation
            "monthly_income": monthly_income,
            "monthly_expenses": monthly_expenses,
            "net_cash_flow": net_cash_flow,
            "top_expenses": top_expenses[:3],
            
            # Historical analysis
            "expense_trend": expense_trend[-3:],  # Last 3 months
            "category_trends": {k: v[-2:] for k, v in category_trends.items()},  # Last 2 months per category
            
            # Goal-specific analysis
            "goals_summary": goals_analysis[:5],  # Limit to 5 goals
            
            # Additional insights
            "essential_vs_discretionary": {
                "essential": essential_spending,
                "discretionary": discretionary_spending,
                "ratio": round(essential_spending / discretionary_spending, 2) if discretionary_spending > 0 else "N/A"
            },
            "savings_rate": round(savings_rate, 1)
        }
        
        # Create system message with focus on goal-centric recommendations
        system_message = (
            "You are a financial advisor specializing in helping users achieve their financial goals. "
            "Provide actionable, goal-oriented financial recommendations based on the user's current "
            "financial situation and historical spending patterns. "
            "Focus on: "
            "1. Specific ways to accelerate progress toward their financial goals "
            "2. Identifying expense categories that could be reduced to free up money for goals "
            "3. Suggesting optimal savings allocation across different goals based on priority and feasibility "
            "4. Recommending specific behavioral changes based on spending patterns "
            "Format your response as 3-5 bullet points. "
            "Be direct, clear, and specific. "
            "Use INR as currency and include exact amounts when possible. "
            "Don't use phrases like 'Based on the data' - just give the recommendations directly."
        )
        
        # Format user content with enhanced historical context
        user_content = (
            f"Monthly Income: ₹{monthly_income:.2f}\n"
            f"Monthly Expenses: ₹{monthly_expenses:.2f}\n"
            f"Net Monthly Cash Flow: ₹{net_cash_flow:.2f}\n\n"
            
            f"Top Expense Categories:\n{json.dumps(top_expenses, indent=2)}\n\n"
            
            f"Expense Trends (Last {len(expense_trend)} Months):\n{json.dumps(expense_trend, indent=2)}\n\n"
            
            f"Savings Rate (3-month): {financial_summary['savings_rate']}%\n"
            f"Essential vs. Discretionary Spending Ratio: {financial_summary['essential_vs_discretionary']['ratio']}\n\n"
            
            f"Goals Progress:\n{json.dumps(financial_summary['goals_summary'], indent=2)}\n\n"
            
            "Please provide specific financial recommendations to help me achieve my goals faster. "
            "Focus on actionable advice based on my spending patterns and goal progress."
        )
        
        # Initialize the InferenceClient with Together AI provider
        client = InferenceClient(
            provider="together",
            api_key=TOGETHER_API_KEY,
        )
        
        # Create the messages for the chat completion with enhanced context
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        
        # Make the API call with Mixtral model
        completion = client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=messages,
            max_tokens=512,
            temperature=0.3,
            top_p=0.9,
            frequency_penalty=0.5,
        )
        
        # Extract and clean the response
        recommendations = completion.choices[0].message.content
        
        # Post-process the response to remove common self-talk patterns
        recommendations = recommendations.replace("As an AI", "As a financial advisor")
        recommendations = recommendations.replace("Let me analyze", "")
        recommendations = recommendations.replace("Based on the information provided", "")
        recommendations = recommendations.replace("I'll provide", "Here's")
        
        # Remove any leading text that appears to be self-talk or reasoning
        lines = recommendations.split('\n')
        cleaned_lines = []
        processing_started = False
        
        for line in lines:
            # Skip initial reasoning/thinking lines
            if not processing_started:
                # Start processing when we see a bullet point or direct advice
                if line.strip().startswith('•') or line.strip().startswith('-') or line.strip().startswith('1.'):
                    processing_started = True
                # Also accept lines that don't seem like self-talk
                elif line.strip() and not any(x in line.lower() for x in ["let me", "i think", "based on", "analyzing", "i'll", "i will", "i can", "first", "here are", "looking at"]):
                    processing_started = True
            
            if processing_started:
                cleaned_lines.append(line)
        
        # If we filtered out everything (unlikely), use original response
        if not cleaned_lines:
            recommendations = recommendations
        else:
            recommendations = '\n'.join(cleaned_lines)
        
        # Cache the recommendations for 6 hours
        cache.set(cache_key, recommendations, 60*60*6)
        
        return recommendations
    except Exception as e:
        logging.error(f"Error generating recommendations: {str(e)}")
        return "Unable to generate recommendations at this time. Please try again later."

@login_required
def add_goal(request):
    if request.method == 'POST':
        form = GoalForm(request.POST)
        if form.is_valid():
            goal = form.save(commit=False)  # Delay saving to add owner
            goal.owner = request.user
            goal.save()
            return redirect('list_goals')
        else:
            # If form is invalid, re-render with error messages
            return render(request, 'goals/add_goals.html', {'form': form})
    else:
        # For non-POST requests, show the form (optional)
        form = GoalForm()
        return render(request, 'goals/add_goals.html', {'form': form})

@login_required(login_url='/authentication/login')
def list_goals(request):

    # goals = Goal.objects.all()
    goals = Goal.objects.filter(owner=request.user)
    ai_recommendations = generate_ai_recommendations(request.user)
    
    context = {
        'goals': goals,
        'ai_recommendations': ai_recommendations
    }
    add_amount_form = AddAmountForm() 
    return render(request, 'goals/list_goals.html', context)


@login_required(login_url='/authentication/login')
def add_amount(request, goal_id):
    goal = get_object_or_404(Goal, pk=goal_id)

    if request.method == 'POST':
        form = AddAmountForm(request.POST)
        if form.is_valid():
            additional_amount = form.cleaned_data['additional_amount']
            amount_required = goal.amount_to_save - goal.current_saved_amount

            if additional_amount > amount_required:
                messages.error(request, f'The maximum amount needed to achieve goal is : {amount_required}.')
            else:
                goal.current_saved_amount += additional_amount
                goal.save()

                # Check if the goal is achieved
                if goal.current_saved_amount == goal.amount_to_save:
                    # Send congratulatory email to the user
                        
                        send_congratulatory_email(request.user.email, goal)
                        messages.success(request, 'Congratulations! You have achieved your goal.')

                        # Disable the "Add Amount" button
                        goal.is_achieved = True
                        goal.delete()
               
                else:
                    messages.success(request, f'Amount successfully added. Total saved amount: {goal.current_saved_amount}.')
                    messages.info(request, f'Amount required to reach goal: {amount_required}.')

        return redirect('list_goals')

    # Redirect to list_goals if the request method is not POST
    return redirect('list_goals')

def send_congratulatory_email(email, goal):
    subject = 'Congratulations on achieving your goal!'
    message = f'Dear User,\n\nCongratulations on achieving your goal "{goal.name}". You have successfully saved {goal.amount_to_save}.\n\nKeep up the good work!\n\nBest regards,\nThe Goal Tracker Team, \nWealthWizard Team'
    send_mail(subject, message, '<your email>', [email])
    
    



def delete_goal(request, goal_id):
    try:
        goal = Goal.objects.get(id=goal_id,owner=request.user)
        goal.delete()
        return redirect('list_goals')
    except Goal.DoesNotExist:
        pass