from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Chat
from expenses.models import Expense
from goals.models import Goal
from userincome.models import UserIncome
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.template.loader import render_to_string
from django.db import models
import requests
import json
from huggingface_hub import InferenceClient
from django.core.cache import cache
from django.db.models import Sum, Avg, Count
from datetime import timedelta
from django.utils import timezone
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Update to use Together AI provider through huggingface_hub
TOGETHER_API_KEY = ""  # Replace with your actual API key

# Function to get optimized context with advanced techniques for handling large datasets
def get_combined_context(user):
    # Use caching for expensive context gathering (10-minute cache)
    cache_key = f'finassist_context_{user.id}'
    cached_context = cache.get(cache_key)
    
    if cached_context:
        return cached_context
    
    try:
        # Get recent date range to limit queries
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=60)  # Consider last 60 days of data
        
        # Get the most recent expenses with optimized query
        expenses = Expense.objects.filter(
            owner=user, 
            date__range=[start_date, end_date]
        ).order_by('-date')[:20]
        
        # Get active goals with essential fields only
        goals = Goal.objects.filter(owner=user).only(
            'name', 'amount_to_save', 'current_saved_amount', 'end_date'
        )
        
        # Get recent incomes with essential fields only
        incomes = UserIncome.objects.filter(
            owner=user, 
            date__range=[start_date, end_date]
        ).order_by('-date')[:20]

        # Calculate expense statistics with optimized query - load in chunks for large datasets
        expense_stats = {}
        
        # Process top categories in one optimized query
        top_categories = Expense.objects.filter(
            owner=user,
            date__range=[start_date, end_date]
        ).values('category').annotate(
            total=Sum('amount'),
            count=Count('id'),
            avg=Avg('amount')
        ).order_by('-total')[:5]
        
        # Calculate monthly spending trend
        current_month = end_date.month
        prev_month = (end_date.replace(day=1) - timedelta(days=1)).month
        
        current_month_spending = Expense.objects.filter(
            owner=user, 
            date__month=current_month
        ).aggregate(Sum('amount'))['amount__sum'] or 0
        
        prev_month_spending = Expense.objects.filter(
            owner=user, 
            date__month=prev_month
        ).aggregate(Sum('amount'))['amount__sum'] or 0
        
        spending_trend = None
        if prev_month_spending > 0:
            change_pct = ((current_month_spending - prev_month_spending) / prev_month_spending) * 100
            spending_trend = {
                'change_percent': round(change_pct, 1),
                'direction': 'up' if change_pct > 0 else 'down'
            }
        
        # Create a more structured and optimized context - JSON serializable
        context = {
            "expense_summary": {
                "recent_expenses": [
                    {
                        "amount": float(exp.amount),
                        "date": exp.date.isoformat(),
                        "description": exp.description,
                        "category": exp.category
                    } 
                    for exp in expenses
                ],
                "top_categories": [
                    {
                        "category": cat["category"],
                        "total": float(cat["total"]),
                        "count": cat["count"],
                        "avg": float(cat["avg"])
                    }
                    for cat in top_categories
                ],
                "spending_trend": spending_trend
            },
            "goals": [
                {
                    "name": goal.name,
                    "amount_to_save": float(goal.amount_to_save),
                    "current_saved_amount": float(goal.current_saved_amount),
                    "end_date": goal.end_date.isoformat() if goal.end_date else None
                }
                for goal in goals
            ],
            "income_summary": {
                "recent_income": [
                    {
                        "amount": float(inc.amount),
                        "date": inc.date.isoformat(),
                        "source": inc.source,
                        "description": inc.description
                    }
                    for inc in incomes
                ]
            }
        }
        
        # Cache the context for 10 minutes to reduce database load
        cache.set(cache_key, context, 60 * 10)
        return context
        
    except Exception as e:
        logger.error(f"Error getting context: {str(e)}")
        # Return minimal context if there's an error
        return {
            "expense_summary": {"recent_expenses": [], "top_categories": []},
            "goals": [],
            "income_summary": {"recent_income": []}
        }

# Function to generate chatbot response with improved handling of large context
def generate_response(user_input, context):
    # Create a more optimized summary of the context for the prompt
    # Focus on most relevant data for the specific user query
    
    # Extract keywords from user query to prioritize relevant context
    keywords = extract_query_keywords(user_input.lower())
    
    # Prepare context based on detected query intent
    expenses_summary = context.get('expense_summary', {})
    goals_summary = context.get('goals', [])
    income_summary = context.get('income_summary', {}).get('recent_income', [])
    
    # Construct optimized context based on query keywords
    relevant_context = {}
    
    # Add expenses data if relevant to query
    if any(kw in keywords for kw in ['expense', 'spend', 'cost', 'budget', 'money', 'paid']):
        relevant_context['expenses'] = expenses_summary.get('recent_expenses', [])[:3]
        relevant_context['top_categories'] = expenses_summary.get('top_categories', [])[:3]
        relevant_context['spending_trend'] = expenses_summary.get('spending_trend')
    
    # Add goals data if relevant to query
    if any(kw in keywords for kw in ['goal', 'save', 'target', 'aim', 'plan']):
        relevant_context['goals'] = [
            {"name": g["name"], "progress": round((g["current_saved_amount"] / g["amount_to_save"]) * 100 if g["amount_to_save"] else 0, 1)}
            for g in goals_summary[:2]
        ]
    
    # Add income data if relevant to query
    if any(kw in keywords for kw in ['income', 'earn', 'salary', 'money', 'receive']):
        relevant_context['income'] = income_summary[:3]
    
    # If no specific focus is detected or for general financial queries, include a balanced summary
    if not relevant_context or any(kw in keywords for kw in ['overall', 'financial', 'summary', 'situation', 'advice']):
        relevant_context = {
            'expenses': expenses_summary.get('recent_expenses', [])[:2],
            'top_categories': expenses_summary.get('top_categories', [])[:2],
            'goals': [{"name": g["name"], "progress": round((g["current_saved_amount"] / g["amount_to_save"]) * 100 if g["amount_to_save"] else 0, 1)} for g in goals_summary[:1]],
            'income': income_summary[:2],
            'spending_trend': expenses_summary.get('spending_trend')
        }
    
    # Format the context to be more concise and useful
    context_prompt = format_context_for_model(relevant_context)
    
    # Prepare the system message for the model - enhanced instructions for cleaner responses
    system_message = (
        "You are a financial assistant providing direct advice about personal finances. "
        "Respond clearly and concisely without any meta-commentary, preambles, or self-reflection. "
        "Do not include phrases like 'As an AI' or 'Based on the data provided'. "
        "Do not preface your answers with explanations about your reasoning process. "
        "Simply provide knowledgeable financial advice based on the user's data and question. "
        "Always use INR as currency in your responses. "
        "If asked about data you don't have, state that you don't have that information rather than making assumptions. "
        "Keep responses focused, actionable, and within 3-4 paragraphs maximum."
    )
    
    try:
        # Initialize the InferenceClient with Together AI provider
        client = InferenceClient(
            provider="together",
            api_key=TOGETHER_API_KEY,
        )
        
        # Create the messages for the chat completion with user context added to the user message
        messages = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": f"Here is my financial data:\n{context_prompt}\n\nMy question is: {user_input}"
            }
        ]
        
        # Make the API call with a better model for direct responses and tuned parameters
        completion = client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=messages,
            max_tokens=512,
            temperature=0.3,
            top_p=0.9,
            frequency_penalty=0.5,
        )
        
        # Extract and clean the response using enhanced post-processing
        assistant_response = clean_model_response(completion.choices[0].message.content)
        
        return assistant_response
        
    except Exception as e:
        logger.error(f"Error with model: {str(e)}")
        return "I apologize, but I'm having trouble processing your request right now. Please try again later."

# Helper function to extract keywords from user query
def extract_query_keywords(query):
    # Remove common stop words to focus on meaningful terms
    stop_words = ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 
                 'should', 'can', 'could', 'may', 'might', 'must', 'i', 'you', 'he', 
                 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their']
    
    words = query.lower().split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return keywords

# Helper function to format context for model consumption
def format_context_for_model(context_data):
    output = []
    
    # Format expenses if available
    if 'expenses' in context_data and context_data['expenses']:
        expenses = context_data['expenses']
        output.append("Recent Expenses:")
        for i, exp in enumerate(expenses[:3]):
            output.append(f"- {exp.get('category', 'Uncategorized')}: ₹{exp.get('amount', 0)} on {exp.get('date', 'N/A')[:10]} ({exp.get('description', 'No description')})")
    
    # Format top expense categories if available
    if 'top_categories' in context_data and context_data['top_categories']:
        categories = context_data['top_categories']
        output.append("\nTop Expense Categories:")
        for i, cat in enumerate(categories[:3]):
            output.append(f"- {cat.get('category', 'Unknown')}: ₹{cat.get('total', 0)} ({cat.get('count', 0)} transactions)")
    
    # Format spending trend if available
    if 'spending_trend' in context_data and context_data['spending_trend']:
        trend = context_data['spending_trend']
        if trend:
            direction = "increased" if trend.get('direction') == 'up' else "decreased"
            output.append(f"\nSpending has {direction} by {abs(trend.get('change_percent', 0))}% compared to last month.")
    
    # Format goals if available
    if 'goals' in context_data and context_data['goals']:
        goals = context_data['goals']
        output.append("\nGoals Progress:")
        for i, goal in enumerate(goals[:2]):
            output.append(f"- {goal.get('name', 'Unnamed goal')}: {goal.get('progress', 0)}% complete")
    
    # Format income if available
    if 'income' in context_data and context_data['income']:
        income = context_data['income']
        output.append("\nRecent Income:")
        for i, inc in enumerate(income[:2]):
            output.append(f"- ₹{inc.get('amount', 0)} from {inc.get('source', 'Unknown')} on {inc.get('date', 'N/A')[:10]}")
    
    return "\n".join(output)

# Enhanced post-processing for model responses
def clean_model_response(response):
    if not response:
        return "Sorry, I couldn't generate a response. Please try again."
    
    # Replace common AI self-references
    response = re.sub(r'(?i)As an AI|As a language model|As an assistant', 'As your financial assistant', response)
    
    # Remove reasoning/thinking lines
    response = re.sub(r'(?i)Let me analyze|Let me think|Based on the (data|information) provided', '', response)
    
    # Process by paragraphs for better control
    lines = response.split('\n')
    cleaned_lines = []
    processing_started = False
    
    for line in lines:
        # Skip initial reasoning/thinking lines
        if not processing_started:
            # Start processing when we see a clear response beginning
            if line.strip() and not any(x in line.lower() for x in [
                "let me", "i think", "based on", "analyzing", "i'll", "i will", "i can", 
                "first", "here are", "looking at", "after reviewing", "upon examining"
            ]):
                processing_started = True
        
        if processing_started:
            # Also clean up any mid-text reasoning
            line = re.sub(r'(?i)I would recommend|I suggest|I advise', 'Recommendation:', line)
            cleaned_lines.append(line)
    
    # If we filtered out everything (unlikely), use original response
    if not cleaned_lines:
        # Try a simpler filtering approach as fallback
        cleaned_response = re.sub(r'(?i)^(.*?)(to answer your question|regarding your question|in response to your question)', '', response)
        if cleaned_response.strip():
            return cleaned_response.strip()
        return response
    
    return '\n'.join(cleaned_lines)

@csrf_exempt
@login_required
def chatbot_view(request):
    # Optimize chat history retrieval - limit to most recent 50 entries
    chat_history = Chat.objects.filter(user=request.user).order_by('-timestamp')[:50]

    if request.method == 'POST':
        user_message = request.POST.get('message')

        if user_message:
            try:
                # Use the optimized context function
                context = get_combined_context(request.user)
                
                # Generate the assistant's response
                assistant_message = generate_response(user_message, context)
                
                # Save to database - consider async processing for high-traffic scenarios
                Chat.objects.create(user=request.user, message=user_message, response=assistant_message)
                
                # For AJAX requests, return JSON response with minimal HTML rendering
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    # Only get the most recent 10 entries for the updated display
                    updated_chat_history = Chat.objects.filter(user=request.user).order_by('-timestamp')[:10]
                    
                    # Render the updated chat history to HTML
                    chat_history_html = render_to_string(
                        'finassist/partials/chat_history.html',
                        {'chat_history': updated_chat_history},
                        request=request
                    )
                    
                    return JsonResponse({
                        'response': assistant_message,
                        'updated_data': {
                            'chat_history': chat_history_html
                        }
                    })
                
                # For regular requests, return the full page with limited history
                return render(request, 'finassist/chatbot.html', {'chat_history': chat_history})
                
            except Exception as e:
                logger.error(f"Error in chatbot view: {str(e)}")
                return JsonResponse({
                    'response': "I'm having trouble processing your request. Please try again later.",
                    'error': str(e)
                }, status=500) if request.headers.get('X-Requested-With') == 'XMLHttpRequest' else render(
                    request, 'finassist/chatbot.html', 
                    {'chat_history': chat_history, 'error': "An error occurred. Please try again."}
                )

    # GET request - render the initial page with limited history
    return render(request, 'finassist/chatbot.html', {'chat_history': chat_history})