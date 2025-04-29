from django.shortcuts import render, redirect, get_object_or_404
from .models import Goal
from userincome.models import UserIncome
from expenses.models import Expense
from .forms import GoalForm, AddAmountForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.mail import send_mail
from django.db.models import Sum, Avg
from django.utils import timezone
import json
from decimal import Decimal
import requests
from django.conf import settings
import logging
from django.core.cache import cache
import re

# Set up logging
logger = logging.getLogger(__name__)

def get_user_financial_data(user):
    """
    Gather and process user's financial data for AI recommendations.
    Returns a dictionary with income, expenses, and goals information.
    """
    # Calculate total monthly income
    current_month = timezone.now().month
    current_year = timezone.now().year
    monthly_income = UserIncome.objects.filter(
        owner=user, 
        date__month=current_month, 
        date__year=current_year
    ).aggregate(total=Sum('amount'))['total'] or 0
    
    # Fallback to average income if no data for current month
    if monthly_income == 0:
        monthly_income = UserIncome.objects.filter(owner=user).aggregate(avg=Avg('amount'))['avg'] or 0
        
    # Calculate total monthly expenses
    monthly_expenses = Expense.objects.filter(
        owner=user, 
        date__month=current_month, 
        date__year=current_year
    ).aggregate(total=Sum('amount'))['total'] or 0
    
    # Get top expense categories
    top_categories = list(Expense.objects.filter(
        owner=user,
        date__month=current_month,
        date__year=current_year
    ).values('category').annotate(total=Sum('amount')).order_by('-total')[:3])
    
    top_categories_formatted = [
        {"category": item['category'], "amount": float(item['total'])} 
        for item in top_categories
    ]
    
    # Get goals progress
    goals = Goal.objects.filter(owner=user)
    goals_summary = []
    
    for goal in goals:
        # Calculate percentage complete
        percent_complete = (goal.current_saved_amount / goal.amount_to_save * 100) if goal.amount_to_save > 0 else 0
        
        # Determine if goal is on track
        days_passed = (timezone.now().date() - goal.start_date).days
        total_days = (goal.target_date - goal.start_date).days
        expected_progress = (days_passed / total_days * 100) if total_days > 0 else 0
        
        status = "on_track" if percent_complete >= expected_progress else "behind"
        
        goals_summary.append({
            "name": goal.name,
            "target": float(goal.amount_to_save),
            "current": float(goal.current_saved_amount),
            "percent_complete": round(percent_complete, 1),
            "status": status
        })
    
    return {
        "total_income": float(monthly_income),
        "total_expenses": float(monthly_expenses),
        "top_expense_categories": top_categories_formatted,
        "goals_summary": goals_summary
    }

def get_rule_based_recommendations(financial_data):
    """
    Generate rule-based recommendations when AI recommendations are unavailable.
    This serves as a fallback for when the API call fails.
    """
    recommendations = []
    
    # Recommendation based on income vs expenses
    income = financial_data['total_income']
    expenses = financial_data['total_expenses']
    
    if expenses > income * 0.9:
        recommendations.append("Your expenses are over 90% of your income. Consider reviewing your budget to reduce expenses and increase your savings rate.")
    
    # Recommendation based on top expense categories
    top_categories = financial_data['top_expense_categories']
    if top_categories and len(top_categories) > 0:
        highest_category = top_categories[0]
        recommendations.append(f"Your highest spending category is {highest_category['category']}. Look for ways to reduce spending in this area to accelerate your progress toward your goals.")
    
    # Generic recommendation for goals
    recommendations.append("Set up automated transfers to your savings accounts right after you receive your income to ensure consistent progress toward your financial goals.")
    
    # Ensure we have at least 3 recommendations
    generic_recommendations = [
        "Consider using the 50/30/20 rule: allocate 50% of income to needs, 30% to wants, and 20% to savings and debt repayment.",
        "Track all expenses diligently to identify spending patterns and opportunities for savings.",
        "Review and adjust your financial goals quarterly to ensure they remain relevant and achievable."
    ]
    
    while len(recommendations) < 3:
        recommendations.append(generic_recommendations[len(recommendations) - 3])
    
    return recommendations[:3]  # Return top 3 recommendations

def generate_ai_recommendations(user):
    try:
        logger.info(f"Generating AI recommendations for user {user.id}")
        api_key = settings.HUGGINGFACE_API_KEY
        
        # Get user's financial data
        financial_data = get_user_financial_data(user)
        
        # Create detailed prompt
        prompt = (
            f"As a financial advisor, suggest 3 actionable tips to help this person reach their savings goals faster. Be specific and concise.\n\n"
            f"Monthly Income: ₹{financial_data['total_income']:.2f}\n"
            f"Monthly Expenses: ₹{financial_data['total_expenses']:.2f}\n"
            f"Top Expense Categories: {json.dumps(financial_data['top_expense_categories'], indent=2)}\n"
            f"Savings Goals Progress: {json.dumps(financial_data['goals_summary'], indent=2)}\n\n"
        )
        
        # Use Hugging Face API
        # Properly encode the model name for URL construction
        import urllib.parse
        encoded_model = urllib.parse.quote(settings.HUGGINGFACE_MODEL)
        API_URL = f"{settings.HUGGINGFACE_API_URL}/{encoded_model}"
        logger.info(f"Using Hugging Face API URL: {API_URL}")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        # Make API request to Hugging Face
        logger.info("Sending request to Hugging Face API")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        
        # If unauthorized, use rule-based recommendations
        if response.status_code == 401:
            logger.warning("Unauthorized access to Hugging Face API, using rule-based recommendations")
            return get_rule_based_recommendations(financial_data)
        
        # Log status code for debugging  
        logger.info(f"Hugging Face API response status code: {response.status_code}")
        
        response.raise_for_status()
        
        # Parse the response from Hugging Face
        result = response.json()
        logger.info(f"API response received: {len(str(result))} characters")
        
        # Extract text from Hugging Face response (structure is different from Together AI)
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and 'generated_text' in result[0]:
            recommendations = result[0]['generated_text'].strip()
        else:
            # Fallback if response format is unexpected
            recommendations = str(result).strip()
            
        # Clean up the recommendations (remove any AI self-references)
        recommendations = re.sub(r'(?i)As an AI|As a language model|As an assistant|As a financial advisor', '', recommendations)
        recommendations = re.sub(r'(?i)Here are (3|three) tips|Here are some tips|I recommend', '', recommendations)
        recommendations = recommendations.strip()
        
        # Split into list items if not already formatted
        if not recommendations.startswith("1.") and not recommendations.startswith("-"):
            sentences = re.split(r'(?<=[.!?])\s+', recommendations)
            recommendations = "\n".join([f"{i+1}. {sentence}" for i, sentence in enumerate(sentences) if sentence.strip()])
        
        # Return the cleaned recommendations
        recommendation_list = [r.strip() for r in re.split(r'(?:\r?\n)|(?:^d+\.)|(?:^-)', recommendations) if r.strip()]
        return recommendation_list[:3]  # Return top 3 recommendations
        
    except Exception as e:
        # Log the error and return rule-based recommendations
        logger.error(f"Error generating AI recommendations: {str(e)}")
        return get_rule_based_recommendations(financial_data)

def generate_rule_based_recommendations(income, expenses, cash_flow, top_expenses, goals):
    """Generate recommendations using rule-based logic rather than AI"""
    recommendations = ["## Financial Recommendations\n"]

    # Rule 1: Basic cash flow analysis
    if cash_flow < 0:
        recommendations.append("1. **Reduce Expenses**: Your expenses (₹{:.2f}) exceed your income (₹{:.2f}) by ₹{:.2f}. Find ways to cut back on non-essential spending.".format(
            expenses, income, abs(cash_flow)))
    elif cash_flow < income * 0.2:  # Less than 20% savings rate
        recommendations.append("1. **Increase Savings**: Your current savings rate is only {:.1f}% of your income. Aim to save at least 20% of your monthly income.".format(
            (cash_flow / income * 100) if income > 0 else 0))
    else:
        recommendations.append("1. **Maintain Savings**: You're saving {:.1f}% of your income, which is excellent. Consider investing part of this surplus for long-term growth.".format(
            (cash_flow / income * 100) if income > 0 else 0))

    # Rule 2: Top expense category analysis
    if top_expenses and len(top_expenses) > 0:
        top_category = top_expenses[0]['category']
        top_amount = top_expenses[0]['amount']
        if expenses > 0:
            percentage = (top_amount / expenses) * 100
            if percentage > 40:
                recommendations.append("2. **Review '{}' Spending**: This category makes up {:.1f}% of your total expenses (₹{:.2f}). Look for ways to reduce this significant expense.".format(
                    top_category, percentage, top_amount))
            else:
                recommendations.append("2. **Monitor '{}' Spending**: This is your highest expense category at ₹{:.2f}. Track it closely to identify potential savings.".format(
                    top_category, top_amount))

    # Rule 3: Goals analysis
    behind_goals = [g for g in goals if g.get('status') == 'behind']
    if behind_goals:
        recommendations.append("3. **Prioritize Goals**: You have {} goals that are behind schedule. Consider allocating more funds to these goals or adjusting their timelines.".format(len(behind_goals)))
    elif goals:
        on_track = [g for g in goals if g.get('status') == 'on_track']
        recommendations.append("3. **Goal Progress**: You have {} goals on track. Keep up with your current savings plan to achieve them on time.".format(len(on_track)))
    
    # Rule 4: General savings advice
    if cash_flow > 0 and len(goals) == 0:
        recommendations.append("4. **Set Savings Goals**: You have a positive cash flow but no savings goals. Consider creating specific financial goals for your future.")
    
    # Rule 5: Emergency fund check - generic as we don't have this data
    recommendations.append("5. **Emergency Fund**: Ensure you maintain an emergency fund covering 3-6 months of expenses (₹{:.2f} - ₹{:.2f}).".format(
        expenses * 3, expenses * 6))

    return "\n\n".join(recommendations)

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