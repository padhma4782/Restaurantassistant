# Import the libraries
import pandas as pd
import chardet
import os, json, ast
from IPython.display import display, HTML
from openai import AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
Azure_OpenAIAPI_KEY = os.environ["Azure_OpenAIAPI_KEY"]
AZURE_ENDPOINT=os.environ["AZURE_ENDPOINT"]
API_VERSION="2024-02-01"

# Set the display width to control the output width
pd.set_option('display.width', 100)
# Read the dataset and read the Laptop Dataset
# Read the CSV with the detected encoding
restaurants_df = pd.read_csv('zomato.csv', encoding='MacRoman')
#restaurants_df
with open('zomato.csv', 'rb') as f:
    result = chardet.detect(f.read())
#print(result['encoding'])

# Initialize OpenAI client
openai = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=Azure_OpenAIAPI_KEY,
    api_version=API_VERSION
)

def initialize_conversation():
    '''
    Returns a list [{"role": "system", "content": system_message}]
    '''

    delimiter = "####"

    example_user_dict = {'City': "Chennai",
                        'Cuisines':"Italian",
                        'Average Cost for two': "1000",
                        'Rating': "3"}
    example_user_req = {'City': "_",
                        'Cuisines': "_",
                        'Average Cost for two': "_",
                        'Rating': "_"
                        }

    system_message = f"""
    You are an intelligent local restaurant guide and your goal is to suggest the best restaurant for a user.
    You need to ask relevant questions and understand the user profile by analysing the user's responses.
    You final objective is to fill the values for the different keys ('City','Cuisines','Average Cost for two','Rating') in the python dictionary and be confident of the values.
    These key value pairs define the user's profile.
    The python dictionary looks like this
    {{'City': 'values','Cuisines': 'values',''Average Cost for two': 'values','Rating': 'values'}}
    The value for 'Average Cost for two'and Rating should be a numerical value extracted from the user's response.
    The values for Rating should be in the integer range from 1 to 5 based on the restaurant rating, as stated by user.
    All the values in the example dictionary are only representative values.
    {delimiter}
    Here are some instructions around the values for the different keys. If you do not follow this, you'll be heavily penalised:
    - The values for only Rating should be in the integer range from 1 to 5 as stated by user.
    - The value for 'Average Cost for two' should be a numerical value extracted from the user's response.
    - Do not randomly assign values to any of the keys.
    - The values need to be inferred from the user's response.
    {delimiter}

    To fill the dictionary, you need to have the following chain of thoughts:
    Follow the chain-of-thoughts below and only output the final updated python dictionary for the keys as described in {example_user_req}. \n
    {delimiter}
    Thought 1: Ask a question to understand the user's preference and requirements. \n
    If their primary preference of City and Cuisine is unclear. Ask followup questions to understand their needs.
    You are trying to fill the values of all the keys {{'City','Cuisines','Average Cost for two','Rating'}} in the python dictionary by understanding the user requirements.
    Identify the keys for which you can fill the values confidently using the understanding. \n
    Remember the instructions around the values for the different keys.
    If the necessary information has been extracted, only then proceed to the next step. \n
    Otherwise, rephrase the question to capture their profile clearly. \n

    {delimiter}
    Thought 2: Now, you are trying to fill the values for the rest of the keys which you couldn't in the previous step.
    Remember the instructions around the values for the different keys.
    Ask questions you might have for all the keys to strengthen your understanding of the user's profile.
    If yes, move to the next Thought. If no, ask question on the keys whose values you are unsure of. \n
    It is a good practice to ask question with a sound logic as opposed to directly citing the key you want to understand value for.
    {delimiter}

    {delimiter}
    Thought 3: Check if you have correctly updated the values for the different keys in the python dictionary.
    If you are not confident about any of the values, ask clarifying questions.
    {delimiter}

    {delimiter}
    Here is a sample conversation between the user and assistant:
    User: "Hi, I am an tourist."
    Assistant: "Great! As a tourist, you likely to prefer local Cuisine. Understanding the specific type of Cuisine work will help me tailor my recommendations accordingly."
    User: "I primarily prefer local Cuisine."
    Assistant: "Thank you for the information. To ensure I have a complete understanding of your needs, I have one more question: Could you kindly let me know your budget for the meal that you are looking for and resturant type on a scale of less than 1000, 1000-2000, more than 2000"
    User: "my max budget is 1000 inr"
    Assistant: "Thank you for providing that information. Let me know if my understanding is correct until now. Understanding your rating preference will help determine the exact restaurant."
    User: "Yes, I am looking 3 star rated restaurant"
    Assistant: "{example_user_dict}"
    {delimiter}

    Start with a short welcome message and encourage the user to share their requirements.
    """
    conversation = [{"role": "system", "content": system_message}]
    return conversation

# Define a Chat Completions API call
# Retry up to 6 times with exponential backoff, starting at 1 second and maxing out at 20 seconds delay
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_chat_completions(input, json_format = False):
    MODEL = 'gpt-4o'

    system_message_json_output = """<<. Return output in JSON format to the key output.>>"""

    # If the output is required to be in JSON format
    if json_format == True:
        # Append the input prompt to include JSON response as specified by OpenAI
        input[0]['content'] += system_message_json_output

        # JSON return type specified
        chat_completion_json = openai.chat.completions.create(
            model = MODEL,
            messages = input,
            response_format = { "type": "json_object"},
            seed = 1234)

        output = json.loads(chat_completion_json.choices[0].message.content)

    # No JSON return type specified
    else:
        chat_completion = openai.chat.completions.create(
            model = MODEL,
            messages = input,
            seed = 2345)

        output = chat_completion.choices[0].message.content

    return output

# Define a function called moderation_check that takes user_input as a parameter.

def moderation_check(user_input):
    # Call the OpenAI API to perform moderation on the user's input.
    #response = openai.moderations.create(input=user_input)
    #print(response)

    response = openai.chat.completions.create(
            model = "gpt-4o",
            messages= [{'role' : 'user', 'content' : str(user_input)}],
            temperature=0,
            max_tokens=900
        )
    response = response.prompt_filter_results[0]['content_filter_results']
    #print("moderation response: " + str(response))

    #Code for 'hate'
    if response.get('hate')['severity'] not in 'safe':
      return 'Flagged'

    #Code for 'jailbreak'
    if str(response.get('jailbreak')['detected']).lower() not in 'false':
      return 'Flagged'

    #Code for 'self_harm'
    if response.get('self_harm')['severity'] not in 'safe':
      return 'Flagged'

    #Code for 'sexual'
    if response.get('sexual')['severity'] not in 'safe':
      return 'Flagged'

    #Code for 'violence'
    if response.get('violence')['severity'] not in 'safe':
      return 'Flagged'

    return 'Not Flagged'

def intent_confirmation_layer(response_assistant):

    delimiter = "####"

    prompt = f"""
    You are a senior evaluator who has an eye for detail.The input text will contain a user requirement captured through 4 keys.
    You are provided an input. You need to evaluate if the input text has the following keys:
    {{
    'City': 'values',
    'Cuisines':'values',
    'Average Cost for two':'values',
    'Rating': 'values'
    }}

    you need to evaluate if the keys have the the values.
    Only output a one-word string in JSON format at the key 'result' - Yes/No.
    Thought 1 - Output a string 'Yes' if the values are correctly filled for all keys, otherwise output 'No'.
    Thought 2 - If the answer is No, mention the reason in the key 'reason'.
    Thought 3 - Think carefully before the answering.
    """

    messages=[{"role": "system", "content":prompt },
              {"role": "user", "content":f"""Here is the input: {response_assistant}""" }]

    response = openai.chat.completions.create(
                                    model="gpt-4o",
                                    messages = messages,
                                    response_format={ "type": "json_object" },
                                    seed = 1234
                                    # n = 5
                                    )

    json_output = json.loads(response.choices[0].message.content)

    return json_output

def dictionary_present(response):
    delimiter = "####"

    user_req = {'City': 'text value',
                'Cuisines': 'text value',
                'Average Cost for two': 'number value',
                'Rating': 'decimal value'
                }

    prompt = f"""You are a python expert. You are provided an input.
            You have to check if there is a python dictionary present in the string.
            It will have the following format {user_req}.
            Your task is to just extract the relevant values from the input and return only the python dictionary in JSON format.
            The output should match the format as {user_req}.

            {delimiter}
            Make sure that the value of budget is also present in the user input. ###
            The output should contain the exact keys and values as present in the input.
            Ensure the keys and values are in the given format:
            {{
            'City': 'text value ',
            'Cuisines':'text value',
            'Average Cost for two':'number value',
            'Rating':'integer value'
            }}

            Here are some sample input output pairs for better understanding:
            {delimiter}

            input 1: {{'City':     'delhi', 'Cuisines': 'indian', 'Average Cost for two': '1,000', 'Rating':'2'}}
            output 1: {{'GPU intensity': 'low', 'Cuisines': 'indian','Average Cost for two': '1000','Rating': '2'}}

            input 2: Here is your user profile 'City': 'chennai','Cuisines': 'chinese','Average Cost for two': '2000 INR','Rating': '4'
            output 2: {{'City': 'chennai','Cuisines':'chinese','Average Cost for two': '2000','Rating': '4'}}
            {delimiter}
            """
    messages = [{"role": "system", "content":prompt },
                {"role": "user", "content":f"""Here is the user input: {response}""" }]

    confirmation = get_chat_completions(messages, json_format = True)

    return confirmation


#Compare restaurants with user requirement

def compare_restaurants_with_user(user_req_string):

    #with open('zomato.csv', 'rb') as f:
    #restaurants_df = chardet.detect(f.read())
      #restaurants_df = pd.read_csv('zomato.csv')
      user_requirements = user_req_string
      #print("****User Requirements inside  compare_restaurants_with_user ***** :", user_requirements)
      # Extracting the budget value from user_requirements and converting it to an integer
      # Get the 'Average Cost for two' as a string first
      budget_string = user_requirements.get('output', {}).get('Average Cost for two', '0')
      # Now apply replace and convert to integer
      User_BudgetRating = int(budget_string.replace(',', '').split()[0]) if isinstance(budget_string, str) else int(budget_string)
      #User_BudgetRating = int(user_requirements.get('output', {}).get('Average Cost for two', '0').replace(',', '').split()[0]) # this should be User_BudgetRating
      User_City = user_requirements.get('output', {}).get('City')

      if User_City is not None:
        User_City =  User_City.lower()

      else:
        User_City = ""

      User_Cuisines = user_requirements.get('output', {}).get('Cuisines') # this should be 'Cuisines' to align with the dict key
      if User_Cuisines is not None:
         User_Cuisines = User_Cuisines.lower()
      else:
         User_Cuisines = ""

      User_Restaurantrating = user_requirements.get('output', {}).get('Rating') # It is used in the filtering logic
      if User_Restaurantrating is not None:
         User_Restaurantrating = int(User_Restaurantrating)
      else:
         User_Restaurantrating = 0

      #print("****User City inside  compare_restaurants_with_user ***** :", User_City)
      #print("****User_Cuisines inside  compare_restaurants_with_user ***** :", User_Cuisines)
      #print("****User_Restaurantrating inside  compare_restaurants_with_user ***** :", User_Restaurantrating)
      #print("****User_BudgetRating  compare_restaurants_with_user ***** :", User_BudgetRating)


    # Filtering restaurants based on user requirements

    # # Creating a copy of the DataFrame and filtering restaurant based on the user requirement
      filtered_restaurants = restaurants_df.copy()
      filtered_restaurants['City'] = filtered_restaurants['City'].str.lower()
      filtered_restaurants['Cuisines'] = filtered_restaurants['Cuisines'].str.lower()

      filtered_restaurants = filtered_restaurants[(filtered_restaurants['City'].str.contains(User_City, na=False)) & (filtered_restaurants['Cuisines'].str.contains(User_Cuisines, na=False))& (filtered_restaurants['Aggregate rating'] >= User_Restaurantrating) & (filtered_restaurants['Average Cost for two'] < User_BudgetRating)].copy()
      #print("****after filtering restaurants*****", filtered_restaurants)
      # Sorting restaurants by ranking in descending order and selecting the top 3 products
      filtered_restaurants = filtered_restaurants.sort_values('Aggregate rating', ascending=False).head(3)
      # Sorting restaurants by ranking in descending order and selecting the top 3 products
      #filtered_restaurants = filtered_restaurants.sort_values('Average Cost for two', ascending=False).head(3) # sort by Average Cost for two
      #filtered_restaurants_json = filtered_restaurants

      return filtered_restaurants

# Define a Chat Completions API call
# Retry up to 6 times with exponential backoff, starting at 1 second and maxing out at 20 seconds delay
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_chat_completions(input, json_format = False):
    MODEL = 'gpt-4o'

    system_message_json_output = """<<. Return output in JSON format to the key output.>>"""

    # If the output is required to be in JSON format
    if json_format == True:
        # Append the input prompt to include JSON response as specified by OpenAI
        input[0]['content'] += system_message_json_output

        # JSON return type specified
        chat_completion_json = openai.chat.completions.create(
            model = MODEL,
            messages = input,
            response_format = { "type": "json_object"},
            seed = 1234)

        output = json.loads(chat_completion_json.choices[0].message.content)

    # No JSON return type specified
    else:
        chat_completion = openai.chat.completions.create(
            model = MODEL,
            messages = input,
            seed = 2345)

        output = chat_completion.choices[0].message.content

    return output


def dialogue_mgmt_system():
    conversation = initialize_conversation()

    introduction = get_chat_completions(conversation)

    display(introduction)

    top_3_restaurants = None

    user_input = ''

    while(user_input != "exit"):

        user_input = input("")

        moderation = moderation_check(user_input)
        if moderation == 'Flagged':
            display("Sorry, this message has been flagged. Please restart your conversation.")
            break

        if top_3_restaurants is None:

            conversation.append({"role": "user", "content": user_input})

            response_assistant = get_chat_completions(conversation)
            print(response_assistant)
            moderation = moderation_check(response_assistant)
            if moderation == 'Flagged':
                display("Sorry, this message has been flagged. Please restart your conversation.")
                break


            confirmation = intent_confirmation_layer(response_assistant)

            print("Intent Confirmation Yes/No:",confirmation.get('result'))

            if "No" in confirmation.get('result'):
                conversation.append({"role": "assistant", "content": str(response_assistant)})
                print("\n" + str(response_assistant) + "\n")

            else:
                #print("\n" + str(response_assistant) + "\n")
                #print('\n' + "Variables extracted!" + '\n')
                response = dictionary_present(response_assistant)
                print("Thank you for providing all the information. Kindly wait, while I fetch the products: \n")
                top_3_restaurants = compare_restaurants_with_user(response)
                #print("*****after getting top3 restaurants*****", top_3_restaurants)
                conversation_reco =[]
                #conversation_reco = [{"role": "user", "content": "This is my user profile" + str(response)}]
                if top_3_restaurants is not None:
                  # Convert DataFrame rows to dictionaries with role and content
                  conversation_reco = [{'role': 'user', 'content': row.to_json()} for _, row in top_3_restaurants.iterrows()]  # Change here
                  conversation_reco.append({"role": "user", "content": "This is my user profile" + str(response)})
                  recommendation = get_chat_completions(conversation_reco)
                  moderation = moderation_check(recommendation)
                  if moderation == 'Flagged':
                    display("Sorry, this message has been flagged. Please restart your conversation.")
                    break
                  conversation_reco.append({"role": "assistant", "content": str(recommendation)})
                  print(str(recommendation) + '\n')
                else:
                   print("****Sorry there are no match******")
                   break
        else:
            conversation_reco.append({"role": "user", "content": user_input})

            response_asst_reco = get_chat_completions(conversation_reco)

            moderation = moderation_check(response_asst_reco)
            if moderation == 'Flagged':
                print("Sorry, this message has been flagged. Please restart your conversation.")
                break

            print('\n' + response_asst_reco + '\n')
            conversation.append({"role": "assistant", "content": response_asst_reco})
dialogue_mgmt_system()