from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Chat
from expenses.models import Expense
from goals.models import Goal
from userincome.models import UserIncome
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from openai import OpenAI
from django.template.loader import render_to_string

client = OpenAI(
	base_url="https://api-inference.huggingface.co/v1/",
    api_key="",
)

# Function to get additional context
def get_combined_context(user):
    expenses = Expense.objects.filter(owner=user)
    goals = Goal.objects.filter(owner=user)
    incomes = UserIncome.objects.filter(owner=user)

    context = {
        "expenses": list(expenses.values("amount", "date", "description", "category")),
        "goals": list(goals.values("name", "amount_to_save", "current_saved_amount", "end_date")),
        "incomes": list(incomes.values("amount", "date", "source", "description")),
    }
    return context

# Function to generate chatbot response using Hugging Face Inference Client
def generate_response(user_input, context):
    # Prepare the context and user input for the LLM
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful financial assistant. Use the following context to respond to user queries:\n"
                f"User's Expenses: {context.get('expenses', [])}\n"
                f"User's Goals: {context.get('goals', [])}\n"
                f"User's Incomes: {context.get('incomes', [])}\n"
            ),
        },
        {"role": "user", "content": user_input},
    ]

    # Call the LLM using the Hugging Face Inference Client
    try:
        completion = client.chat.completions.create(
            model="microsoft/Phi-3.5-mini-instruct",
            messages=messages,
            max_tokens=500,
        )
        return  completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

@csrf_exempt
@login_required
def chatbot_view(request):
    chat_history = Chat.objects.filter(user=request.user).order_by('-timestamp')

    if request.method == 'POST':
        user_message = request.POST.get('message')

        if user_message:
            # Fetch context data
            expenses = Expense.objects.filter(owner=request.user)
            goals = Goal.objects.filter(owner=request.user)
            incomes = UserIncome.objects.filter(owner=request.user)

            context = {
                'expenses': expenses,
                'goals': goals,
                'incomes': incomes,
            }

            # Generate the assistant's response
            assistant_message = generate_response(user_message, context)

            # Save to database
            Chat.objects.create(user=request.user, message=user_message, response=assistant_message)

            # For AJAX requests, return JSON response
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                # Get updated chat history
                updated_chat_history = Chat.objects.filter(user=request.user).order_by('-timestamp')
                
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

            # For regular requests, return the full page
            return render(request, 'finassist/chatbot.html', {'chat_history': chat_history})

    # GET request - render the initial page
    return render(request, 'finassist/chatbot.html', {'chat_history': chat_history})